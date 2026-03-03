import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import scipy.spatial
import gc  # Import garbage collection module
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LogNorm

RANDOM_SEED = 1757414760
print(f"Random seed: {RANDOM_SEED}")
np.random.seed(RANDOM_SEED)
cp.random.seed(RANDOM_SEED)

# Get absolute path of current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
coordinates_file = os.path.join(current_dir, "sampled_normalized_coordinates.npy")  # Use sampled normalized data
region_path = os.path.join(os.path.dirname(__file__), 'sampled_region_labels.npy')
region_labels = np.load(region_path)

# Brain region name order
regions = [
    'ACx',
    'CN',
    'SP',
    'FP',
    'HPC',
    'IC',
    'IL',
    'MGB',
    'OFC',
    'PL',
    'Pons',
    'LP',
    'TRN'
]

# Count neurons per region
dict_region_count = {region: np.sum(region_labels == region) for region in regions}

# Compute ave_deg per region; largest region 50, others scaled linearly
max_deg = 20
min_deg = 5
counts = np.array([dict_region_count[region] for region in regions])
n_max = counts.max()
n_min = counts.min()
if n_max == n_min:
    dict_region_deg = {region: max_deg for region in regions}
else:
    dict_region_deg = {
        region: int(round(min_deg + (dict_region_count[region] - n_min) / (n_max - n_min) * (max_deg - min_deg)))
        for region in regions
    }
ave_deg_vec = np.array([dict_region_deg[label] for label in region_labels], dtype=np.int32)
ave_deg_vec = cp.array(ave_deg_vec)

# Fixed parameters
ave_deg = 50

def clear_gpu_memory():
    """Release GPU memory and pinned memory pool used by CuPy for large matrices or to avoid OOM."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def clear_cpu_memory():
    """Trigger Python garbage collection to free CPU memory of unreferenced objects."""
    gc.collect()

def calculate_distances(coordinates):
    """Compute pairwise Euclidean distances between neurons from 3D coordinates on GPU; diagonal may be 0.

    Parameters
    ----------
    coordinates : cp.ndarray or np.ndarray, shape (n, 3), dtype float32
        Neuron normalized coordinates (see upstream sampled normalized data).

    Returns
    -------
    distances : cp.ndarray, shape (n, n), dtype cp.float32
        Pairwise Euclidean distance matrix; units match coordinates.
    """
    print("Computing pairwise neuron distances...")
    n = coordinates.shape[0]
    distances = cp.zeros((n, n), dtype=cp.float32)
    
    # Use GPU to compute distances
    for i in tqdm(range(n), desc="Computing distance matrix"):
        diff = coordinates - coordinates[i]
        distances[i] = cp.sqrt(cp.sum(diff**2, axis=1))
    
    print("Distance computation completed.")
    return distances

def calculate_connection_probabilities(distances, alpha):
    """Compute connection probability matrix.

    Parameters
    ----------
    distances : cp.ndarray, shape (n, n), dtype float32
        Pairwise Euclidean distance matrix; units match coordinates.
    alpha : float
        Distance decay exponent: prob ∝ distance**(-alpha); typically in (0, 2].

    Returns
    -------
    prob_matrix : cp.ndarray, shape (n, n)
        Pairwise connection probability matrix; clipped to [0, 1], diagonal 0.
    """
    print(f"Computing connection probability for alpha={alpha:.2f}...")
    n = distances.shape[0]
    
    # Base distance probability
    prob_matrix = distances**(-alpha)
    cp.fill_diagonal(prob_matrix, 0)
    prob_matrix = prob_matrix/cp.max(prob_matrix)
    
    target_degree_weights = cp.zeros(n, dtype=cp.float16) 
    node_indices = cp.arange(n, dtype=cp.float16)
    power_law_weights = 1.0 / (node_indices + 1)**0.8  # Power-law distribution
    power_law_weights = power_law_weights / cp.sum(power_law_weights) * n  # Normalize

    # Apply degree weights to probability matrix
    prob_matrix = prob_matrix * power_law_weights[:, cp.newaxis]
    prob_matrix = prob_matrix * power_law_weights[cp.newaxis, :]
    
    # Clear temporary variables
    del power_law_weights, node_indices
    cp.get_default_memory_pool().free_all_blocks()

    neighbor_strength = 1.0 / (distances + 1e-6)
    cp.fill_diagonal(neighbor_strength, 0)
    row_sums = cp.sum(neighbor_strength, axis=1, keepdims=True)
    row_sums = cp.where(row_sums == 0, 1, row_sums)
    normalized_neighbor_strength = neighbor_strength / row_sums
    neighbor_density_features = cp.sum(normalized_neighbor_strength, axis=1)
    neighbor_density_similarity = cp.zeros_like(distances, dtype=cp.float16)
    
    block_size = 1000
    neighbor_density_similarity = cp.zeros((n, n), dtype=cp.float16)
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            density_i = neighbor_density_features[i:end_i, cp.newaxis]
            density_j = neighbor_density_features[j:end_j, cp.newaxis]
            density_diff_block = cp.abs(density_i - density_j.T)
            max_diff = cp.max(density_diff_block)
            similarity_block = 1.0 - (density_diff_block / (max_diff + 1e-6))
            neighbor_density_similarity[i:end_i, j:end_j] = similarity_block.astype(cp.float16)
            
            del density_diff_block, similarity_block
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear temporary variables
    del neighbor_density_features
    cp.get_default_memory_pool().free_all_blocks()
    
    local_threshold = cp.percentile(distances, 10)
    global_threshold = cp.percentile(distances, 50)
    block_size = 1000
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            dist_block = distances[i:end_i, j:end_j]
            sim_block = neighbor_density_similarity[i:end_i, j:end_j]
            prob_block = prob_matrix[i:end_i, j:end_j]
            local_mask = dist_block < local_threshold
            global_mask = (dist_block >= local_threshold) & (dist_block < global_threshold)
            distant_mask = dist_block >= global_threshold
            enhancement_block = cp.ones_like(prob_block, dtype=cp.float16)
            enhancement_block[local_mask] = 1.0 + sim_block[local_mask] * 0.8
            enhancement_block[global_mask] = 1.0 + sim_block[global_mask] * 0.3
            enhancement_block[distant_mask] = 1.0 + sim_block[distant_mask] * 0.1
            prob_matrix[i:end_i, j:end_j] = prob_block * enhancement_block
            del dist_block, sim_block, prob_block, enhancement_block
            del local_mask, global_mask, distant_mask
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear temporary variables
    del local_threshold, global_threshold, neighbor_density_similarity
    cp.get_default_memory_pool().free_all_blocks()
    
    connection_tendency = cp.sum(normalized_neighbor_strength, axis=1)
    mean_tendency = cp.mean(connection_tendency)
    std_tendency = cp.std(connection_tendency)
    offset = -0.8 * std_tendency
    gaussian_weights = cp.exp(-0.5 * ((connection_tendency - (mean_tendency + offset)) / (std_tendency * 0.8 + 1e-6))**2)
    gaussian_weights = gaussian_weights.astype(cp.float16)
    del connection_tendency, mean_tendency, std_tendency
    cp.get_default_memory_pool().free_all_blocks()
    
    block_size = 1000
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            prob_block = prob_matrix[i:end_i, j:end_j]
            weights_i = gaussian_weights[i:end_i, cp.newaxis]
            weights_j = gaussian_weights[j:end_j, cp.newaxis]
            prob_matrix[i:end_i, j:end_j] = prob_block * weights_i * weights_j.T
            del prob_block, weights_i, weights_j
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear weight variables
    del gaussian_weights
    cp.get_default_memory_pool().free_all_blocks()
    
    neighbor_density_variance = cp.var(normalized_neighbor_strength, axis=1)
    mean_variance = cp.mean(neighbor_density_variance)
    std_variance = cp.std(neighbor_density_variance)
    clustering_weights = cp.exp(-2 * ((neighbor_density_variance - mean_variance) / (std_variance + 1e-6))**2)
    clustering_weights = clustering_weights.astype(cp.float16)
    del neighbor_density_variance, mean_variance, std_variance
    cp.get_default_memory_pool().free_all_blocks()
    
    block_size = 1000
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            prob_block = prob_matrix[i:end_i, j:end_j]
            weights_i = clustering_weights[i:end_i, cp.newaxis]
            weights_j = clustering_weights[j:end_j, cp.newaxis]
            prob_matrix[i:end_i, j:end_j] = prob_block * weights_i * weights_j.T
            del prob_block, weights_i, weights_j
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear weight variables
    del clustering_weights
    cp.get_default_memory_pool().free_all_blocks()
    
    block_size = 1000
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            dist_block = distances[i:end_i, j:end_j]
            prob_block = prob_matrix[i:end_i, j:end_j]
            mean_dist = cp.mean(distances)
            distance_penalty_block = cp.exp(-dist_block / (mean_dist * 0.8))
            prob_matrix[i:end_i, j:end_j] = prob_block * (1.0 - distance_penalty_block * 0.3)
            del dist_block, prob_block, distance_penalty_block
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear all temporary variables
    del normalized_neighbor_strength, neighbor_strength
    cp.get_default_memory_pool().free_all_blocks()
    print("All temporary variables cleared.")
    
    node_connection_density = cp.sum(prob_matrix, axis=1)
    mean_density = cp.mean(node_connection_density)
    std_density = cp.std(node_connection_density)
    skewness = cp.mean(((node_connection_density - mean_density) / (std_density + 1e-6))**3)
    
    if skewness > 2.0:
        balance_factor = 0.5
        print(f"Detected heavy right skew (skewness={skewness:.2f}); applying strong balance.")
    elif skewness < 1.0: 
        balance_factor = 0.8
        print(f"Detected overly uniform distribution (skewness={skewness:.2f}); applying weak balance.")
    else:
        balance_factor = 0.65 
        print(f"Detected reasonable distribution (skewness={skewness:.2f}); applying moderate balance.")
    
    density_threshold = cp.percentile(node_connection_density, 75)
    high_density_mask = node_connection_density > density_threshold
    penalty_factor = cp.ones_like(node_connection_density, dtype=cp.float16)
    penalty_factor[high_density_mask] = balance_factor
    
    block_size = 1000
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            prob_block = prob_matrix[i:end_i, j:end_j]
            penalty_i = penalty_factor[i:end_i, cp.newaxis]
            penalty_j = penalty_factor[j:end_j, cp.newaxis]
            prob_matrix[i:end_i, j:end_j] = prob_block * penalty_i * penalty_j.T
            del prob_block, penalty_i, penalty_j
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear temporary variables
    del node_connection_density, density_threshold, high_density_mask, penalty_factor
    del mean_density, std_density, skewness
    cp.get_default_memory_pool().free_all_blocks()
    
    # Ensure probabilities in valid range
    prob_matrix = cp.clip(prob_matrix, 0, 1)

    print("Connection probability computation completed.")
    return prob_matrix

def generate_connection_matrix(prob_matrix, ave_deg, multiplier=1):
    """Generate binary connection matrix from probability matrix; fixed seed, GPU sampling: random < prob*multiplier.

    Parameters
    ----------
    prob_matrix : cp.ndarray, shape (n, n), values in [0, 1]
        Pairwise connection probability matrix.
    ave_deg : int or float
        Target mean degree (see caller for usage).
    multiplier : number, default 1
        Probability scale: compare random < prob_matrix * multiplier.

    Returns
    -------
    cp.ndarray, same shape as prob_matrix, dtype cp.int8
        Binary connection matrix (0 or 1).
    """
    print("Generating connection matrix...")
    cp.random.seed(RANDOM_SEED)
    
    # Generate connection matrix: [0,1) uniform random on GPU, same shape as prob_matrix
    random_matrix = cp.random.rand(*prob_matrix.shape)

    # Compare random < probability at each position
    result_matrix = random_matrix < prob_matrix * multiplier
    print("Connection matrix generation completed.")

    # Convert to integer (0 or 1)
    return result_matrix.astype(cp.int8)

def calculate_entropy(connection_matrix):
    """Compute information entropy of node connection distribution (1st- and 2nd-order neighbors); batched, returns graph mean.

    Parameters
    ----------
    connection_matrix : cp.ndarray, shape (n, n), int8-compatible
        Binary adjacency matrix; 0/1 for no connection / connection.

    Returns
    -------
    avg_entropy : float
        Graph mean information entropy (nats), ≥ 0.
    """
    print("Computing information entropy...")
    n = connection_matrix.shape[0]
    batch_size = 500  # 500 nodes per batch
    total_entropy = 0
    total_nodes = 0
    
    # Compute second-order neighbor matrix (matrix squared)
    print("Computing second-order neighbor matrix...")
    second_order_matrix = cp.dot(connection_matrix, connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Computing information entropy"):
        end_idx = min(i + batch_size, n)
        # Get current batch connection matrix
        batch_matrix = connection_matrix[i:end_idx]
        batch_second_order = second_order_matrix[i:end_idx]

        # Merge 1st- and 2nd-order neighbors
        combined_matrix = batch_matrix + batch_second_order

        # Set non-zero to 1
        combined_matrix = (combined_matrix > 0).astype(cp.int8)
        cp.fill_diagonal(combined_matrix, 0)

        # Compute current batch connection counts
        row_sums = cp.sum(combined_matrix, axis=1)
        row_sums = cp.where(row_sums == 0, 1, row_sums)  # Avoid division by zero

        # Compute current batch information entropy
        q_j = combined_matrix / row_sums[:, cp.newaxis]
        q_j = cp.where(q_j == 0, 1, q_j)  # Avoid log(0)
        batch_entropy = -cp.sum(q_j * cp.log(q_j), axis=1)

        # Accumulate batch entropy
        total_entropy += float(cp.sum(batch_entropy))
        total_nodes += end_idx - i
    
    # Compute average entropy
    avg_entropy = total_entropy / total_nodes if total_nodes > 0 else 0
    print("Information entropy computation completed.")
    return avg_entropy

def get_energy_bound(connection_matrix, distances, w):
    """Prune connection matrix under energy upper bound: keep edges whose cumulative distance ≤ w per node; batched GPU.

    Parameters
    ----------
    connection_matrix : cp.ndarray, shape (n, n), dtype int8
        Current binary connection matrix.
    distances : cp.ndarray, shape (n, n)
        Pairwise distance matrix; same scale as connection_matrix.
    w : cp.ndarray, shape (n,)
        Energy budget per node; units match distances (e.g. mean_distance * ave_deg_vec).

    Returns
    -------
    new_connection_matrix : cp.ndarray, shape (n, n), dtype cp.int8
        Binary connection matrix satisfying energy constraint (0/1).
    """
    print("Applying energy constraint...")
    n = connection_matrix.shape[0]
    batch_size = 500  # 500 nodes per batch
    new_connection_matrix = cp.zeros_like(connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Applying energy constraint"):
        end_idx = min(i + batch_size, n)
        # Get current batch connections and distances
        batch_adj = connection_matrix[i:end_idx]
        batch_dis = distances[i:end_idx]
        batch_w = w[i:end_idx]
        batch_w = batch_w[:, cp.newaxis]

        # Process current batch in parallel
        sorted_indices = cp.argsort(batch_dis, axis=1)
        sorted_dis = cp.take_along_axis(batch_dis, sorted_indices, axis=1)
        sorted_adj = cp.take_along_axis(batch_adj, sorted_indices, axis=1)
        
        valid_connections = (sorted_adj != 0)
        energy_cumsum = cp.cumsum(sorted_dis * valid_connections, axis=1)
        threshold_mask = (energy_cumsum <= batch_w) & valid_connections
        reverse_indices = cp.argsort(sorted_indices, axis=1)
        original_order_mask = cp.take_along_axis(threshold_mask, reverse_indices, axis=1)
        
        # Set new connections
        new_connection_matrix[i:end_idx] = original_order_mask.astype(cp.int8)

    print("Energy constraint applied.")
    return new_connection_matrix

def plot_alpha_curve(alpha_values, entropy_values, save_path):
    """Plot single Alpha vs Information Entropy curve and save figure; x-axis alpha, y-axis Information Entropy.

    Parameters
    ----------
    alpha_values : array-like, 1d
        Alpha values (see experiment design).
    entropy_values : array-like, 1d
        Information entropy values (nats) corresponding to alpha_values.
    save_path : str
        Full path (including filename) to save the figure.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, entropy_values, 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('Information Entropy')
    plt.title(f'Information Entropy vs Alpha (ave_deg={ave_deg})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_weighted_connection_matrix(conn_bin, distances, mu=0, sigma=1, reciprocal_boost=2, single_ratio=0.5, seed=None):
    """Generate weighted (float) connection matrix from binary matrix: lognormal weights, reciprocal boost, unilateral scale, random inhibitory, zero-mean.

    Parameters
    ----------
    conn_bin : cp.ndarray, shape (n, n), 0/1
        Binary connection matrix.
    distances : cp.ndarray, shape (n, n)
        Distance matrix (not used in this function; see caller).
    mu, sigma : number
        Lognormal parameters; default 0, 1.
    reciprocal_boost : number, default 2
        Multiplier for reciprocal pair weights.
    single_ratio : number, default 0.5
        Scale for unilateral connections relative to reciprocal.
    seed : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    W_raw : cp.ndarray, shape (n, n), dtype cp.float32
        Weighted connection matrix; zero where no connection; non-zero weights zero-mean.
    """
    print("Generating weighted connection matrix...")
    if seed is not None:
        cp.random.seed(seed)
    n = conn_bin.shape[0]
    W_raw = cp.zeros_like(conn_bin, dtype=cp.float32)
    mask = conn_bin == 1
    W_raw[mask] = cp.random.lognormal(mu, sigma, size=int(cp.sum(mask).item()))
    print(cp.max(W_raw[mask]), cp.min(W_raw[mask]), cp.mean(W_raw[mask]))

    # Batch process reciprocal pairs
    reciprocal_mask = (conn_bin == 1) & (conn_bin.T == 1)
    upper_tri_mask = cp.triu(cp.ones_like(conn_bin, dtype=bool), k=1)
    reciprocal_pairs = cp.where(reciprocal_mask & upper_tri_mask)
    if reciprocal_pairs[0].size > 0:
        w1 = W_raw[reciprocal_pairs]
        w2 = W_raw[(reciprocal_pairs[1], reciprocal_pairs[0])]
        w_mean = (w1 + w2) / 2
        noise = cp.random.normal(0, 0.1 * w_mean)
        w1_new = (w_mean + noise) * reciprocal_boost
        w2_new = (w_mean - noise) * reciprocal_boost
        W_raw[reciprocal_pairs] = w1_new
        W_raw[(reciprocal_pairs[1], reciprocal_pairs[0])] = w2_new
    print("Reciprocal pairs done; scaling unilateral connections...")

    # Batch process unilateral connections
    single_mask = (conn_bin == 1) & (conn_bin.T == 0)
    single_idx = cp.where(single_mask)
    if single_idx[0].size > 0:
        W_raw[single_idx] *= single_ratio
    print("Weighted connection matrix generation completed.")

    # 1. Reciprocal pair mask (upper triangle only to avoid duplicate)
    reciprocal_pairs = cp.array(reciprocal_pairs).T  # shape (n_reci, 2)

    # 2. Unilateral connection mask
    single_pairs = cp.array(single_idx).T  # shape (n_single, 2)

    # 3. Randomly assign inhibitory to reciprocal and unilateral
    inhib_ratio = 0.2
    n_reci = reciprocal_pairs.shape[0]
    n_single = single_pairs.shape[0]
    n_reci_inhib = int(n_reci * inhib_ratio)
    n_single_inhib = int(n_single * inhib_ratio)

    # Reciprocal pair inhibitory
    if n_reci_inhib > 0:
        reci_inhib_idx = cp.random.choice(n_reci, n_reci_inhib, replace=False)
        reci_inhib_pairs = reciprocal_pairs[reci_inhib_idx]
        # Assign negative to both (i,j) and (j,i)
        W_raw[reci_inhib_pairs[:, 0], reci_inhib_pairs[:, 1]] *= -1
        W_raw[reci_inhib_pairs[:, 1], reci_inhib_pairs[:, 0]] *= -1

    # Unilateral inhibitory
    if n_single_inhib > 0:
        single_inhib_idx = cp.random.choice(n_single, n_single_inhib, replace=False)
        single_inhib_pairs = single_pairs[single_inhib_idx]
        W_raw[single_inhib_pairs[:, 0], single_inhib_pairs[:, 1]] *= -1

    # Zero-mean
    mean_val = cp.mean(W_raw[mask])
    W_raw[mask] -= mean_val

    return W_raw

def main():
    """Main workflow: load coordinates and region labels, compute distances and connection probability;
    generate binary connection matrix with best alpha, apply energy bound and weighting, plot 3D partial connections and save final weight matrix."""
    # Set up GPU
    print("Initializing GPU...")
    cp.cuda.Device(0).use()
    
    # Load coordinate data
    print("Loading coordinate data...")
    print(f"Loading data from {coordinates_file}...")
    coordinates = np.load(coordinates_file)
    coordinates = cp.array(coordinates, dtype=cp.float32)
    
    # Compute distance matrix
    distances = calculate_distances(coordinates)

    # Experiment parameters
    alpha_values = np.round(np.arange(0, 1.05, 0.05), 2)  # 0 to 2, step 0.05
    mean_distance = cp.mean(distances)  # Mean distance

    # Main loop: sweep alpha
    entropy_results = []
    for alpha in tqdm(alpha_values, desc="Sweeping alpha"):
        print(f"\nProcessing alpha={alpha:.2f}...")

        # Compute connection probability
        prob_matrix = calculate_connection_probabilities(distances, alpha)

        # Generate connection matrix
        connection_matrix = generate_connection_matrix(prob_matrix, ave_deg, multiplier=10)

        # Apply energy constraint
        connection_matrix = get_energy_bound(connection_matrix, distances, mean_distance * ave_deg_vec)

        # Compute information entropy
        entropy = calculate_entropy(connection_matrix)

        # Save result
        entropy_results.append(entropy)
        print(f"alpha={alpha:.2f} entropy={entropy}")

        # Clear connection matrix
        del connection_matrix, prob_matrix
        clear_gpu_memory()
    
    # Create result directory if not exists
    result_dir = os.path.join(current_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    # Plot
    image_path = os.path.join(result_dir, 'alpha_entropy_curve.png')
    plot_alpha_curve(alpha_values, entropy_results, image_path)

    # Save data to txt file
    txt_path = os.path.join(result_dir, 'alpha_entropy_curve.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Alpha-Entropy curve data\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment parameters:\n")
        f.write(f"  - Mean degree (ave_deg): {ave_deg}\n")
        f.write(f"  - Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f}\n")
        f.write(f"  - Alpha step: {alpha_values[1] - alpha_values[0]:.2f}\n")
        f.write(f"  - Total test points: {len(alpha_values)}\n")
        f.write(f"  - Random seed: {RANDOM_SEED}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Data list (Alpha, Information Entropy)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Alpha':<15} {'Information Entropy':<20}\n")
        f.write("-" * 60 + "\n")
        for alpha, entropy in zip(alpha_values, entropy_results):
            f.write(f"{alpha:<15.2f} {entropy:<20.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Max information entropy: {max(entropy_results):.6f}\n")
        f.write(f"  Min information entropy: {min(entropy_results):.6f}\n")
        f.write(f"  Mean information entropy: {np.mean(entropy_results):.6f}\n")
        f.write(f"  Std information entropy: {np.std(entropy_results):.6f}\n")
        best_idx = int(np.argmax(entropy_results))
        best_alpha = float(alpha_values[best_idx])
        f.write(f"\n  Best Alpha: {best_alpha:.2f}\n")
        f.write(f"  Corresponding max information entropy: {entropy_results[best_idx]:.6f}\n")
    print(f"Data saved to: {txt_path}")
    
    # Select best alpha
    best_idx = int(np.argmax(entropy_results))
    best_alpha = float(alpha_values[best_idx])
    print(f"\nBest alpha: {best_alpha:.2f}, max information entropy: {entropy_results[best_idx]}")

    # Generate final connection matrix with best alpha
    print("\nGenerating final connection matrix with best alpha...")
    final_prob_matrix = calculate_connection_probabilities(distances, best_alpha)
    final_connection_matrix = generate_connection_matrix(final_prob_matrix, ave_deg)
    final_connection_matrix = get_energy_bound(final_connection_matrix, distances, mean_distance * ave_deg_vec)
    
    # Obtain weighted connection matrix
    final_connection_matrix = generate_weighted_connection_matrix(final_connection_matrix, distances, mu=0, sigma=0.2,
                                                                  reciprocal_boost=1.3, single_ratio=0.7)

    # Plot interactive 3D neuron distribution and partial connections
    print("Plotting 3D neuron distribution and partial connections...")

    # Get 3D coordinates
    if isinstance(coordinates, cp.ndarray):
        coords_np = cp.asnumpy(coordinates)
    else:
        coords_np = coordinates

    region_labels_np = np.array(region_labels)

    # Color mapping
    region_list = list(dict_region_count.keys())
    cmap = cm.get_cmap('tab20', len(region_list))
    region_color_dict = {region: cmap(i) for i, region in enumerate(region_list)}

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    for region in region_list:
        idx = np.where(region_labels_np == region)[0]
        ax.scatter(coords_np[idx, 0], coords_np[idx, 1], coords_np[idx, 2],
                   s=8, color=region_color_dict[region], label=region, alpha=0.7)

    # Draw partial connection lines for target region only
    target_region = "PrelimbicArea"
    line_ratio = 0.5

    if isinstance(final_connection_matrix, cp.ndarray):
        conn_mat = cp.asnumpy(final_connection_matrix)
    else:
        conn_mat = final_connection_matrix

    auditory_idx = np.where(region_labels_np == target_region)[0]

    # Consider non-zero absolute values only
    abs_weights = np.abs(conn_mat[conn_mat != 0])
    vmin = max(abs_weights.min(), 5e-2)
    vmax = np.percentile(abs_weights, 80)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    for i in auditory_idx:
        connected = np.where(conn_mat[i] != 0)[0]
        if len(connected) > 0:
            num_to_draw = max(1, int(len(connected) * line_ratio))
            draw_idx = np.random.choice(connected, size=num_to_draw, replace=False)
            for j in draw_idx:
                weight = conn_mat[i, j]
                color = cmap(norm(abs(weight)))
                ax.plot([coords_np[i, 0], coords_np[j, 0]],
                        [coords_np[i, 1], coords_np[j, 1]],
                        [coords_np[i, 2], coords_np[j, 2]],
                        color=color, linewidth=1.5, alpha=0.8)

    # Add log-scale colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label('Connection Weight (log scale)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Neuron Distribution and {target_region} Partial Connections')
    ax.legend(loc='upper right', fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.show()
    print(f"Partial neuron connections for {target_region} displayed ({int(line_ratio*100)}%).")

    # Save final connection matrix
    print("Saving final connection matrix...")
    matrix_file = os.path.join(current_dir, 'final_connection_matrix.npy')
    np.save(matrix_file, cp.asnumpy(final_connection_matrix))
    
    print("All processing completed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} s") 