"""
Mouse connection probability model validation: multi-sample connection probability estimation,
power-law fitting, and multi-model comparison (power-law / exponential / Gaussian kernel / logistic).
"""
import pandas as pd
import os
import numpy as np
import cupy as cp
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
from scipy.optimize import curve_fit
from sklearn.utils import resample
from sklearn.metrics import log_loss

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = True
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

def clear_gpu_memory():
    """Release CuPy default memory pools (device and pinned) to reclaim GPU memory between large batches."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

# Script directory and data/output paths
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, "synapses_pni_2.csv")

output_dir = os.path.join(current_dir, "visual_cortex_result")
os.makedirs(output_dir, exist_ok=True)

print("\n=== Starting multi-sample analysis ===")
sample_sizes = [30000, 50000, 70000, 100000]
results_by_sample = {}

for sample_size in sample_sizes:
    print(f"\n=== Processing sample size: {sample_size} ===")
    # Read CSV slice (skip first 2M rows, take nrows rows)
    print(f"Loading first {sample_size} rows from CSV...")
    df = pd.read_csv(csv_file, skiprows=range(1, 2000001), nrows=sample_size)

    print("\n=== Starting coordinate processing ===")
    # Extract pre/post coordinates (columns 3–5, 8–10) and cast to float32
    print("Extracting and converting coordinates...")
    pre_coords = df.iloc[:, [2, 3, 4]].values.astype(np.float32)  # columns 3,4,5 as pre coords
    post_coords = df.iloc[:, [7, 8, 9]].values.astype(np.float32)  # columns 8,9,10 as post coords

    # Compute global max distance on GPU as normalization factor
    print("Computing normalization factor...")
    all_coords_gpu = cp.asarray(np.vstack([pre_coords, post_coords]))
    max_coords = cp.max(all_coords_gpu, axis=0)
    min_coords = cp.min(all_coords_gpu, axis=0)
    max_possible_dist = float(cp.sqrt(cp.sum((max_coords - min_coords)**2)))
    del all_coords_gpu
    clear_gpu_memory()

    # Normalize coordinates by max possible distance (CPU)
    print("Normalizing coordinates...")
    pre_coords_norm = pre_coords / max_possible_dist
    post_coords_norm = post_coords / max_possible_dist

    print("\n=== Starting ID processing ===")
    # Extract pre/post neuron IDs (columns 7, 12)
    print("Extracting ID data...")
    pre_ids = df.iloc[:, 6].values  # column 7
    post_ids = df.iloc[:, 11].values  # column 12

    # Unique IDs and counts n_pre, n_post
    print("Counting unique IDs...")
    unique_pre_ids = np.unique(pre_ids)
    unique_post_ids = np.unique(post_ids)
    n_pre = len(unique_pre_ids)
    n_post = len(unique_post_ids)

    print(f"\nColumn 7 unique ID count: {n_pre}")
    print(f"Column 12 unique ID count: {n_post}")

    # Build ID -> index mapping
    print("Creating ID mapping...")
    pre_id_to_idx = {id_: idx for idx, id_ in enumerate(unique_pre_ids)}
    post_id_to_idx = {id_: idx for idx, id_ in enumerate(unique_post_ids)}

    print("\n=== Starting average coordinate computation ===")
    # Compute mean coordinates per ID in normalized space
    pre_avg_coords = {}
    post_avg_coords = {}

    print("Computing mean coordinates for pre_id...")
    for id_ in tqdm(unique_pre_ids, desc="Processing pre_id"):
        mask = pre_ids == id_
        pre_avg_coords[id_] = np.mean(pre_coords_norm[mask], axis=0)

    print("Computing mean coordinates for post_id...")
    for id_ in tqdm(unique_post_ids, desc="Processing post_id"):
        mask = post_ids == id_
        post_avg_coords[id_] = np.mean(post_coords_norm[mask], axis=0)

    print("\n=== Starting distance matrix and connection count computation ===")
    # Preallocate distance and connection count matrices
    dist_matrix = np.zeros((n_pre, n_post), dtype=np.float32)
    conn_matrix = np.zeros((n_pre, n_post), dtype=np.float32)

    # Compute distance and connection count in batches on GPU (batch_size pre per batch)
    batch_size = 1000  # 1000 pre_id per batch
    print(f"Batch size: {batch_size}")
    print("Starting batched distance and connection count computation...")

    # Upload full pre/post ID arrays to GPU for masking
    pre_ids_gpu = cp.asarray(pre_ids)
    post_ids_gpu = cp.asarray(post_ids)

    for i_start in tqdm(range(0, n_pre, batch_size), desc="Processing pre_id batches"):
        i_end = min(i_start + batch_size, n_pre)
        batch_pre_ids = unique_pre_ids[i_start:i_end]

        # Current batch pre mean coordinates to GPU
        batch_pre_coords = cp.asarray(np.array([pre_avg_coords[id_] for id_ in batch_pre_ids]))

        for j_start in range(0, n_post, batch_size):
            j_end = min(j_start + batch_size, n_post)
            batch_post_ids = unique_post_ids[j_start:j_end]

            # Current batch post mean coordinates to GPU
            batch_post_coords = cp.asarray(np.array([post_avg_coords[id_] for id_ in batch_post_ids]))

            # Pairwise Euclidean distances within batch
            diff = batch_pre_coords[:, None, :] - batch_post_coords[None, :, :]
            batch_dist = cp.sqrt(cp.sum(diff**2, axis=2))

            # Connection count: pre/post ID mask matrices multiply to (n_pre_batch, n_post_batch), >0 means connected
            batch_pre_ids_gpu = cp.asarray(batch_pre_ids)
            batch_post_ids_gpu = cp.asarray(batch_post_ids)

            pre_mask = cp.equal(pre_ids_gpu[:, None], batch_pre_ids_gpu[None, :])  # shape: (n_synapses, n_pre_batch)
            post_mask = cp.equal(post_ids_gpu[:, None], batch_post_ids_gpu[None, :])  # shape: (n_synapses, n_post_batch)

            batch_conn = cp.asarray(pre_mask.T @ post_mask, dtype=cp.float32)

            # Copy batch results back to CPU and write to corresponding block
            dist_matrix[i_start:i_end, j_start:j_end] = cp.asnumpy(batch_dist)
            conn_matrix[i_start:i_end, j_start:j_end] = cp.asnumpy(batch_conn)

            # Release batch GPU intermediates
            del batch_post_coords, diff, batch_dist, batch_conn, pre_mask, post_mask, batch_pre_ids_gpu, batch_post_ids_gpu
            clear_gpu_memory()

        del batch_pre_coords
        clear_gpu_memory()

    del pre_ids_gpu, post_ids_gpu
    clear_gpu_memory()

    print("\n=== Starting connection probability computation ===")
    # Distance binning: 100 bins, [0, 1]
    n_bins = 100
    print("Creating distance bins...")
    distances = np.linspace(0, 1, n_bins+1, dtype=np.float32)

    # b: number of pairs per bin, a: number of connections per bin
    b = np.zeros(n_bins, dtype=np.float32)  # distance distribution
    a = np.zeros(n_bins, dtype=np.float32)  # connection count

    # Accumulate b, a per bin in batches on GPU
    print("Aggregating distance bin data...")
    batch_size = 1000  # 1000 rows per batch
    for i_start in tqdm(range(0, n_pre, batch_size), desc="Processing distance bins"):
        i_end = min(i_start + batch_size, n_pre)

        batch_dist = cp.asarray(dist_matrix[i_start:i_end, :])
        batch_conn = cp.asarray(conn_matrix[i_start:i_end, :])
        distances_gpu = cp.asarray(distances)

        for i in range(n_bins):
            mask = (batch_dist >= distances_gpu[i]) & (batch_dist < distances_gpu[i+1])
            b[i] += float(cp.sum(mask))
            a[i] += float(cp.sum(batch_conn[mask]))

        del batch_dist, batch_conn, distances_gpu
        clear_gpu_memory()

    # Connection probability a/b (set 0 when b=0)
    print("Computing connection probability...")
    connection_prob = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    # Bin centers for plotting
    bin_centers = (distances[:-1] + distances[1:]) / 2

    print("\n=== Starting fitting analysis ===")
    # Log-domain power-law fit: filter valid points (x>0, y>0, finite), ln-ln linear regression
    valid = (bin_centers > 0) & (connection_prob > 0) & np.isfinite(connection_prob)
    log_x = np.log(bin_centers[valid])
    log_y = np.log(connection_prob[valid])

    if len(log_x) > 2:
        slope, intercept = np.polyfit(log_x, log_y, 1)  # ln-ln scale
        alpha = -slope  # ln p = ln k - alpha*ln d -> slope = -alpha
        k = float(np.exp(intercept))  # intercept is ln k
        y_pred = slope * log_x + intercept
        r2 = 1 - np.sum((log_y - y_pred)**2) / np.sum((log_y - np.mean(log_y))**2)
    else:
        alpha = np.nan
        r2 = np.nan
        k = np.nan
        slope = np.nan

    # Generate power-law fit curve sample points for plotting
    x_fit = np.logspace(np.log10(bin_centers[valid][0]), np.log10(bin_centers[valid][-1]), 100) if len(log_x) > 2 else bin_centers
    if not np.isnan(k):
        y_fit = k * x_fit**(-alpha)
    else:
        y_fit = np.full_like(x_fit, np.nan)

    # Detection limit
    detection_limit = 1.5 / b
    detection_limit = np.where(b > 0, detection_limit, np.inf)  # avoid division by zero

    # Write current sample size result
    results_by_sample[sample_size] = {
        'bin_centers': bin_centers,
        'connection_prob': connection_prob,
        'sample_counts': b,
        'detection_limit': detection_limit,
        'alpha': alpha,
        'k': k,
        'r2': r2,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'valid': valid
    }

    print(f"Sample size {sample_size}: alpha = {alpha:.2f}, k = {k:.6e}, R2 = {r2:.4f}")

print("\n=== Starting multi-sample comparison plot ===")

# Dual y-axis: left = connection probability, right = sample count
fig, ax1 = plt.subplots(figsize=(12, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
linestyles = ['-', '-', '-', '-']

# Plot by sample size: scatter, fit line, detection limit
for i, sample_size in enumerate(sample_sizes):
    result = results_by_sample[sample_size]
    color = colors[i]
    marker = markers[i]
    linestyle = linestyles[i]

    valid_bins = result['sample_counts'] > 0
    bin_centers_valid = result['bin_centers'][valid_bins]
    connection_prob_valid = result['connection_prob'][valid_bins]
    sample_counts_valid = result['sample_counts'][valid_bins]
    detection_limit_valid = result['detection_limit'][valid_bins]

    ax1.scatter(bin_centers_valid, connection_prob_valid,
               alpha=0.6, s=30, color=color, marker=marker,
               label=f'{sample_size} sample size data')

    if not np.isnan(result['alpha']):
        ax1.plot(result['x_fit'], result['y_fit'],
                color=color, linestyle=linestyle, linewidth=2,
                label=f'{sample_size} sample size fit (alpha={result["alpha"]:.2f})')

    ax1.plot(bin_centers_valid, detection_limit_valid,
            color=color, linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'{sample_size} sample size detection limit')

# Left axis: log-log, transparent background for overlay with right axis
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_zorder(2)                 # left axis on top
ax1.patch.set_visible(False)      # transparent so right-axis content shows through

# Set y range from all valid data points
all_y_values = []
for sample_size in sample_sizes:
    if sample_size in results_by_sample:
        result = results_by_sample[sample_size]
        valid_bins = result['sample_counts'] > 0
        connection_prob_valid = result['connection_prob'][valid_bins]
        all_y_values.extend(connection_prob_valid[connection_prob_valid > 0])

if all_y_values:
    y_min = np.min(all_y_values) * 0.5  # slightly extend lower bound
    y_max = np.max(all_y_values) * 2.0  # slightly extend upper bound
    ax1.set_ylim(y_min, y_max)
else:
    ax1.set_ylim(1e-8, 1e-1)  # fallback range
ax1.set_xlabel('Distance', fontsize=14)
ax1.set_ylabel('Connection\nprobability', fontsize=14, rotation=0, labelpad=20, color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Right axis: sample count (log), placed behind
ax2 = ax1.twinx()
ax2.set_zorder(1)                 # right axis behind
ax2.patch.set_visible(False)      # transparent to avoid occlusion
ax2.set_xscale('log')             # same log x as left axis

for i, sample_size in enumerate(sample_sizes):
    result = results_by_sample[sample_size]
    color = colors[i]

    valid_bins = result['sample_counts'] > 0
    bin_centers_valid = result['bin_centers'][valid_bins]
    sample_counts_valid = result['sample_counts'][valid_bins]

    ax2.plot(bin_centers_valid, sample_counts_valid,
             color=color, linestyle=':', linewidth=2, alpha=0.8,
             label=f'{sample_size} sample count')

ax2.set_yscale('log')
ax2.set_ylabel('Sample count\n(pairs)', fontsize=14, rotation=0, labelpad=20, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Merge left and right legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
leg.set_zorder(30)  # keep legend on top

plt.title('Multi-sample connection probability comparison', fontsize=16)

img_path = os.path.join(output_dir, 'connection_probability_with_sample_count.png')
plt.savefig(img_path, bbox_inches='tight', dpi=300)
plt.close()

# Write multi-sample numerical results
txt_path = os.path.join(output_dir, 'connection_probability_with_sample_count.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("Multi-sample connection probability analysis data\n")
    f.write("=" * 60 + "\n\n")
    f.write("Data description:\n")
    f.write("- This file contains all data for multi-sample connection probability analysis.\n")
    f.write("- Each sample size includes: bin center (distance), connection probability, sample count, detection limit, fit parameters and fit curve data.\n")
    f.write("- Data grouped by sample size; each group has multiple columns.\n\n")

    for sample_size in sample_sizes:
        if sample_size in results_by_sample:
            result = results_by_sample[sample_size]
            f.write(f"\n{'='*60}\n")
            f.write(f"Sample size: {sample_size}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Fitting parameters:\n")
            f.write(f"  alpha = {result['alpha']:.6f}\n")
            f.write(f"  k = {result['k']:.6e}\n")
            f.write(f"  R2 = {result['r2']:.6f}\n\n")

            valid_bins = result['sample_counts'] > 0
            bin_centers_valid = result['bin_centers'][valid_bins]
            connection_prob_valid = result['connection_prob'][valid_bins]
            sample_counts_valid = result['sample_counts'][valid_bins]
            detection_limit_valid = result['detection_limit'][valid_bins]

            f.write(f"Data point statistics ({len(bin_centers_valid)} valid bins):\n")
            f.write(f"{'Bin center (distance)':<20} {'Connection probability':<20} {'Sample count (pairs)':<20} {'Detection limit':<20}\n")
            f.write("-" * 80 + "\n")
            for i in range(len(bin_centers_valid)):
                f.write(f"{bin_centers_valid[i]:<20.8e} {connection_prob_valid[i]:<20.8e} "
                       f"{sample_counts_valid[i]:<20.0f} {detection_limit_valid[i]:<20.8e}\n")

            if not np.isnan(result['alpha']):
                f.write(f"\nFitted curve data ({len(result['x_fit'])} points):\n")
                f.write(f"{'Distance':<20} {'Fitted connection probability':<20}\n")
                f.write("-" * 40 + "\n")
                for i in range(len(result['x_fit'])):
                    f.write(f"{result['x_fit'][i]:<20.8e} {result['y_fit'][i]:<20.8e}\n")
            f.write("\n")

print("\n=== Starting model comparison (100k sample size only) ===")

result_100k = results_by_sample[100000]

# ====== Four connection probability models: power-law, exponential, Gaussian kernel, logistic ======
def power_law(d, k, alpha):
    """Power-law p(d) = k * d^(-alpha).

    Parameters
    ----------
    d : array-like
        Distance.
    k, alpha : float
        Scale and exponent.

    Returns
    -------
    array-like
        Connection probability.
    """
    return k * d**(-alpha)

def exponential(d, k, lambda_):
    """Exponential decay p(d) = k * exp(-lambda * d).

    Parameters
    ----------
    d : array-like
        Distance.
    k, lambda_ : float
        Scale and decay rate.

    Returns
    -------
    array-like
        Connection probability.
    """
    return k * np.exp(-lambda_ * d)

def gaussian_kernel(d, k, sigma):
    """Gaussian kernel decay p(d) = k * exp(-d^2 / (2*sigma^2)).

    Parameters
    ----------
    d : array-like
        Distance.
    k, sigma : float
        Scale and width.

    Returns
    -------
    array-like
        Connection probability.
    """
    return k * np.exp(- (d**2) / (2.0 * sigma**2))

def logistic_func(d, k, a, d0):
    """Logistic / S-shaped p(d) = k / (1 + exp(a*(d - d0))).

    Parameters
    ----------
    d : array-like
        Distance.
    k, a, d0 : float
        Scale, steepness, inflection point.

    Returns
    -------
    array-like
        Connection probability.
    """
    return k / (1.0 + np.exp(a * (d - d0)))

# ====== Fitting data: valid points, sorted by x ======
x = results_by_sample[100000]['bin_centers']
y = results_by_sample[100000]['connection_prob']

valid = (x > 0) & (y > 0) & np.isfinite(y)
x_fit = x[valid]
y_fit = y[valid]

print(f"[Model comparison] Valid points for fitting: {len(x_fit)}")

order = np.argsort(x_fit)
x_fit = x_fit[order]
y_fit = y_fit[order]

print(f"Data range: x=[{x_fit.min():.4f}, {x_fit.max():.4f}], y=[{y_fit.min():.2e}, {y_fit.max():.2e}]")

eps = 1e-12
sigma_w = np.maximum(y_fit, eps)

print("Fitting four models...")

models = {}
model_names = ['Power-law', 'Exponential', 'Gaussian kernel', 'Logistic']

# Power-law: log-log linear regression
lx, ly = np.log(x_fit), np.log(y_fit)
slope, intercept = np.polyfit(lx, ly, 1)
popt_power = np.array([np.exp(intercept), -slope])
models['power'] = {'params': popt_power, 'cov': None}
print(f"Power-law fit succeeded: k={popt_power[0]:.2e}, alpha={popt_power[1]:.2f}")

# Exponential: log-domain log p = log k - lambda*d
def exponential_log(d, log_k, lambda_):
    """Log-domain exponential: log p = log k - lambda * d. Used for curve_fit."""
    return log_k - lambda_ * d

try:
    popt_exp, pcov_exp = curve_fit(exponential_log, x_fit, ly,
                                   p0=[np.log(np.median(y_fit)), 1.0],
                                   maxfev=10000)
    k_exp = np.exp(popt_exp[0])
    lambda_exp = popt_exp[1]
    models['exp'] = {'params': [k_exp, lambda_exp], 'cov': pcov_exp}
    print(f"Exponential fit succeeded: k={k_exp:.2e}, lambda={lambda_exp:.2f}")
except Exception as e:
    print(f"Exponential fit failed: {e}")
    models['exp'] = {'params': [np.nan, np.nan], 'cov': None}

# Gaussian kernel: log p = log k - d^2/(2*sigma^2), linear regression in d^2
try:
    z = x_fit**2
    slope_g, intercept_g = np.polyfit(z, ly, 1)  # ly approx intercept_g + slope_g * z
    k_g = np.exp(intercept_g)
    if slope_g < 0:
        sigma_g = np.sqrt(-1.0 / (2.0 * slope_g))
        models['gauss'] = {'params': [k_g, sigma_g], 'cov': None}
        print(f"Gaussian kernel fit succeeded: k={k_g:.2e}, sigma={sigma_g:.3f}")
    else:
        print("Gaussian kernel fit failed: non-negative slope, no valid sigma")
        models['gauss'] = {'params': [np.nan, np.nan], 'cov': None}
except Exception as e:
    print(f"Gaussian kernel fit failed: {e}")
    models['gauss'] = {'params': [np.nan, np.nan], 'cov': None}

# Logistic: log-domain fit with constraints a>0, d0 in [x_min, x_max]
def logistic_log(d, log_k, a, d0):
    """log p = log_k - log(1+exp(a*(d-d0))); log1p(exp(...)) for numerical stability."""
    return log_k - np.log1p(np.exp(a * (d - d0)))

try:
    popt_logi, pcov_logi = curve_fit(
        logistic_log, x_fit, ly,
        p0=[np.log(np.median(y_fit)), 5.0, float(np.median(x_fit))],  # initial: log_k, a, d0
        bounds=([-30.0, 1e-6, x_fit.min()], [0.0, np.inf, x_fit.max()]),  # log_k<=0 => k<=1; a>0; d0 in range
        maxfev=20000
    )
    logk_logi, a_logi, d0_logi = popt_logi
    k_logi = float(np.exp(logk_logi))
    models['logistic'] = {'params': [k_logi, a_logi, d0_logi], 'cov': pcov_logi}
    print(f"Logistic (3-parameter) fit succeeded (with bounds): k={k_logi:.3g}, a={a_logi:.3f}, d0={d0_logi:.3f}")
except Exception as e:
    print(f"Logistic fit failed: {e}")
    models['logistic'] = {'params': [np.nan, np.nan, np.nan], 'cov': None}

# ====== Compute AIC/BIC, R2 in log domain ======
print("\nComputing model evaluation metrics...")
def calculate_aic_bic(y_true, y_pred, n_params):
    """AIC/BIC and log-likelihood from log-transformed data (Gaussian log-likelihood approximation).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted connection probabilities (>0).
    n_params : int
        Number of model parameters.

    Returns
    -------
    aic, bic, log_likelihood : float
    """
    n = len(y_true)
    log_y_true = np.log(y_true)
    log_y_pred = np.log(y_pred)
    mse_log = np.mean((log_y_true - log_y_pred)**2)
    log_likelihood = -n/2 * np.log(2*np.pi*mse_log) - n/2
    aic = 2*n_params - 2*log_likelihood
    bic = n_params * np.log(n) - 2*log_likelihood
    return aic, bic, log_likelihood

# Uniform sampling over x_fit range for plotting and evaluation
x_eval = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)

model_results = {}
for model_name, model_data in models.items():
    if not np.any(np.isnan(model_data['params'])):
        if model_name == 'power':
            y_pred = power_law(x_fit, *model_data['params'])
            n_params = 2
        elif model_name == 'exp':
            y_pred = exponential(x_fit, *model_data['params'])
            n_params = 2
        elif model_name == 'gauss':
            y_pred = gaussian_kernel(x_fit, *model_data['params'])
            n_params = 2
        elif model_name == 'logistic':
            y_pred = logistic_func(x_fit, *model_data['params'])
            n_params = 3
        else:
            continue

        # Avoid log(0) breaking log-domain evaluation
        y_pred = np.clip(y_pred, 1e-300, None)

        aic, bic, log_likelihood = calculate_aic_bic(y_fit, y_pred, n_params)
        log_y_fit = np.log(y_fit)
        log_y_pred = np.log(y_pred)
        r2 = 1 - np.sum((log_y_fit - log_y_pred)**2) / np.sum((log_y_fit - np.mean(log_y_fit))**2)

        model_results[model_name] = {
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'r2': r2,
            'n_params': n_params
        }

        print(f"{model_name} model: AIC={aic:.2f}, BIC={bic:.2f}, R2={r2:.4f}")

# ====== Plot four model fit curves and data points ======
print("Plotting model comparison figure...")
plt.figure(figsize=(12, 8))

plt.scatter(x_fit, y_fit, alpha=0.6, color='black', s=30, label='Data points', zorder=5)

colors = ['red', 'blue', 'green', 'purple']
linestyles = ['-', '--', '-.', ':']

for i, (model_name, model_data) in enumerate(models.items()):
    if not np.any(np.isnan(model_data['params'])):
        if model_name == 'power':
            y_curve = power_law(x_eval, *model_data['params'])
        elif model_name == 'exp':
            y_curve = exponential(x_eval, *model_data['params'])
        elif model_name == 'gauss':
            y_curve = gaussian_kernel(x_eval, *model_data['params'])
        elif model_name == 'logistic':
            y_curve = logistic_func(x_eval, *model_data['params'])
        else:
            continue

        y_curve = np.clip(y_curve, 1e-300, None)
        plt.plot(x_eval, y_curve, color=colors[i], linestyle=linestyles[i],
                 linewidth=2, label=f'{model_names[i]} model', alpha=0.8)

plt.xscale('log')
plt.yscale('log')

# Set y range from data and curves
all_y_values = [y_fit]
for model_name, model_data in models.items():
    if not np.any(np.isnan(model_data['params'])):
        if model_name == 'power':
            y_curve = power_law(x_eval, *model_data['params'])
        elif model_name == 'exp':
            y_curve = exponential(x_eval, *model_data['params'])
        elif model_name == 'gauss':
            y_curve = gaussian_kernel(x_eval, *model_data['params'])
        elif model_name == 'logistic':
            y_curve = logistic_func(x_eval, *model_data['params'])
        else:
            continue
        all_y_values.append(np.clip(y_curve, 1e-300, None))

all_y_values = np.concatenate(all_y_values)
valid_y = all_y_values[all_y_values > 0]
if len(valid_y) > 0:
    y_min = np.min(valid_y) * 0.5
    y_max = np.max(valid_y) * 2.0
    plt.ylim(y_min, y_max)
else:
    plt.ylim(1e-8, 1e-1)
plt.xlabel('Distance', fontsize=14)
plt.ylabel('Connection\nprobability', fontsize=14, rotation=0, labelpad=20)
plt.title('Four-model comparison (100k sample size)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

img_path = os.path.join(output_dir, 'model_comparison.png')
plt.savefig(img_path, bbox_inches='tight', dpi=300)
plt.close()

txt_path = os.path.join(output_dir, 'model_comparison.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("Four-model comparison data (100k sample size)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Data description:\n")
    f.write("- This file contains all data for the four-model comparison.\n")
    f.write("- Includes raw data points, fit parameters and fit curve data for each model.\n\n")

    f.write(f"Raw data points ({len(x_fit)} points):\n")
    f.write(f"{'Distance':<20} {'Connection probability':<20}\n")
    f.write("-" * 40 + "\n")
    for i in range(len(x_fit)):
        f.write(f"{x_fit[i]:<20.8e} {y_fit[i]:<20.8e}\n")
    f.write("\n")

    for i, (model_name, model_data) in enumerate(models.items()):
        f.write(f"\n{'='*60}\n")
        f.write(f"Model: {model_names[i]}\n")
        f.write(f"{'='*60}\n\n")

        if not np.any(np.isnan(model_data['params'])):
            f.write("Fitting parameters:\n")
            if model_name == 'power':
                f.write(f"  k = {model_data['params'][0]:.6e}\n")
                f.write(f"  alpha = {model_data['params'][1]:.6f}\n")
            elif model_name == 'exp':
                f.write(f"  k = {model_data['params'][0]:.6e}\n")
                f.write(f"  lambda = {model_data['params'][1]:.6f}\n")
            elif model_name == 'gauss':
                f.write(f"  k = {model_data['params'][0]:.6e}\n")
                f.write(f"  sigma = {model_data['params'][1]:.6f}\n")
            elif model_name == 'logistic':
                f.write(f"  k = {model_data['params'][0]:.6e}\n")
                f.write(f"  a = {model_data['params'][1]:.6f}\n")
                f.write(f"  d0 = {model_data['params'][2]:.6f}\n")

            if model_name in model_results:
                metrics = model_results[model_name]
                f.write(f"\nEvaluation metrics:\n")
                f.write(f"  AIC = {metrics['aic']:.2f}\n")
                f.write(f"  BIC = {metrics['bic']:.2f}\n")
                f.write(f"  R2 = {metrics['r2']:.6f}\n")
                f.write(f"  Log-likelihood = {metrics['log_likelihood']:.2f}\n")

            if model_name == 'power':
                y_curve = power_law(x_eval, *model_data['params'])
            elif model_name == 'exp':
                y_curve = exponential(x_eval, *model_data['params'])
            elif model_name == 'gauss':
                y_curve = gaussian_kernel(x_eval, *model_data['params'])
            elif model_name == 'logistic':
                y_curve = logistic_func(x_eval, *model_data['params'])
            else:
                continue

            y_curve = np.clip(y_curve, 1e-300, None)
            f.write(f"\nFitted curve data ({len(x_eval)} points):\n")
            f.write(f"{'Distance':<20} {'Fitted connection probability':<20}\n")
            f.write("-" * 40 + "\n")
            for j in range(len(x_eval)):
                f.write(f"{x_eval[j]:<20.8e} {y_curve[j]:<20.8e}\n")
        else:
            f.write("Fit failed\n")
        f.write("\n")

# ====== Write model evaluation metrics summary ======
print("Writing model evaluation results to file...")
with open(os.path.join(output_dir, 'model_evaluation_results.txt'), 'w', encoding='utf-8') as f:
    f.write("Model evaluation metrics comparison\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"{'Model':<15} {'N params':<8} {'AIC':<12} {'BIC':<12} {'R2':<10} {'Log-likelihood':<12}\n")
    f.write("-" * 80 + "\n")

    for i, (model_name, model_data) in enumerate(models.items()):
        if model_name in model_results:
            metrics = model_results[model_name]
            f.write(f"{model_names[i]:<15} {metrics['n_params']:<8} {metrics['aic']:<12.2f} {metrics['bic']:<12.2f} {metrics['r2']:<10.4f} {metrics['log_likelihood']:<12.2f}\n")
        else:
            f.write(f"{model_names[i]:<15} {'Failed':<8} {'-':<12} {'-':<12} {'-':<10} {'-':<12}\n")

    f.write("\n\nNotes:\n")
    f.write("- AIC (Akaike Information Criterion): lower is better\n")
    f.write("- BIC (Bayesian Information Criterion): lower is better\n")
    f.write("- R2 (coefficient of determination): closer to 1 is better\n")
    f.write("- Log-likelihood: higher is better\n")

print("\nAnalysis complete. Results saved to the following files (in visual_cortex_result):")
print("1. connection_probability_with_sample_count.png - Multi-sample comparison figure")
print("2. connection_probability_with_sample_count.txt - Multi-sample comparison data")
print("3. model_comparison.png - Four-model comparison figure (100k sample size)")
print("4. model_comparison.txt - Four-model comparison data")
print("5. model_evaluation_results.txt - Model evaluation metrics")

print("\nFitting results by sample size:")
for sample_size in sample_sizes:
    r = results_by_sample[sample_size]
    print(f"Sample size {sample_size}: alpha = {r['alpha']:.2f}, k = {r['k']:.6e}, R2 = {r['r2']:.4f}")

print("\nNote:")
print("- Detection limit uses 1.5/N formula for statistical significance.")
print("- Multi-sample figure uses different colors and markers.")
print("- Model comparison uses 100k sample size only.")
