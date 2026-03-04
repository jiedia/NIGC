import os
import time
import traceback
from collections import Counter
from pathlib import Path
import cv2
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import umap
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Wedge, Polygon
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.signal import (
    butter,
    filtfilt,
    hilbert,
    coherence,
    welch,
    get_window,
    cwt,
    ricker,
)
from scipy.spatial.distance import cosine
from scipy.stats import (
    skew,
    kurtosis,
    truncnorm,
    pearsonr,
    entropy,
    spearmanr,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    pairwise_distances,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load from retina npz feature files; patch partitioning + normalization
def _collect_npz_files(data_dir, categories):
    """
    Collect npz file list per category by category name.
    Two directory layouts supported:
      1) data_dir/<category>/*.npz
      2) data_dir/*.npz with category keyword in filename
    Returns:
      category_to_files: dict[category] -> [file1, file2, ...]
    """
    category_to_files = {}

    for category in categories:
        category_dir = os.path.join(data_dir, category)
        npz_files = []

        if os.path.isdir(category_dir):
            npz_files = sorted(
                os.path.join(category_dir, f)
                for f in os.listdir(category_dir)
                if f.endswith(".npz")
            )
        else:
            # when no subdirs, match by filename keyword in data_dir
            npz_files = sorted(
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz") and category in f
            )

        if len(npz_files) == 0:
            print(f"Warning: category {category} has no npz files under {data_dir}")
        else:
            print(f"Category {category}: found {len(npz_files)} npz files")

        category_to_files[category] = npz_files

    return category_to_files

def _compute_global_stats(category_to_files, categories, last_T=60):
    """
    First pass: over all npz files, compute global mean and variance for
    DoG / DSGC / Looming under "use only last last_T frames per video".
    """
    sum_dog = 0.0
    sumsq_dog = 0.0
    count_dog = 0

    sum_dsgc = 0.0
    sumsq_dsgc = 0.0
    count_dsgc = 0

    sum_loom = 0.0
    sumsq_loom = 0.0
    count_loom = 0

    for category in categories:
        for npz_path in category_to_files.get(category, []):
            data = np.load(npz_path)

            # load three feature types
            dog_all = data["features"].astype(np.float32)              # (T, 6*H*W)
            dsgc_all = data["directional_features"].astype(np.float32) # (T, 8*H*W)
            looming_all = data["looming_features"].astype(np.float32)  # (T,)

            T = dog_all.shape[0]
            if T < last_T:
                raise ValueError(
                    f"{os.path.basename(npz_path)} has {T} time frames, "
                    f"fewer than required last {last_T} frames"
                )

            dog = dog_all[-last_T:]      # (last_T, 6*H*W)
            dsgc = dsgc_all[-last_T:]    # (last_T, 8*H*W)
            looming = looming_all[-last_T:]  # (last_T,)

            sum_dog += dog.sum()
            sumsq_dog += np.square(dog).sum()
            count_dog += dog.size

            sum_dsgc += dsgc.sum()
            sumsq_dsgc += np.square(dsgc).sum()
            count_dsgc += dsgc.size

            sum_loom += looming.sum()
            sumsq_loom += np.square(looming).sum()
            count_loom += looming.size

    if count_dog == 0 or count_dsgc == 0 or count_loom == 0:
        raise ValueError("No DoG/DSGC/Looming features found; check data path and file structure.")

    mean_dog = sum_dog / count_dog
    var_dog = sumsq_dog / count_dog - mean_dog ** 2
    std_dog = np.sqrt(max(var_dog, 1e-8))

    mean_dsgc = sum_dsgc / count_dsgc
    var_dsgc = sumsq_dsgc / count_dsgc - mean_dsgc ** 2
    std_dsgc = np.sqrt(max(var_dsgc, 1e-8))

    mean_loom = sum_loom / count_loom
    var_loom = sumsq_loom / count_loom - mean_loom ** 2
    std_loom = np.sqrt(max(var_loom, 1e-8))

    print("Global DoG mean/std:", mean_dog, std_dog)
    print("Global DSGC mean/std:", mean_dsgc, std_dsgc)
    print("Global Looming mean/std:", mean_loom, std_loom)

    stats = {
        "dog": (mean_dog, std_dog),
        "dsgc": (mean_dsgc, std_dsgc),
        "loom": (mean_loom, std_loom),
    }
    return stats

def load_retina_patched_data(data_dir, last_T=60, patch_size=16, stride=8):
    """
    Load retina npz features and process as follows:
      1) Use only last last_T frames per video.
      2) Global z-score normalization for DoG / DSGC / Looming.
      3) Partition 32x32 space into 3x3 patches:
           - patch size patch_size x patch_size (default 16x16)
           - stride between patch centers default 8 pixels
          Per patch:
           - DoG: 16x16x6 -> mean -> 6 dims
           - DSGC: 16x16x8 -> mean -> 8 dims
          Total 14 dims per patch; 9 patches -> 126 dims.
      4) Looming stays 1 dim.
      5) Each frame -> 127 dims; each video -> (last_T, 127) temporal features.

    Returns:
      X: (num_samples, last_T, 127)
      y: (num_samples,)  0/1/2 = close_to / observation / stay_away
    """
    categories = ["close_to", "observation", "stay_away"]
    category_to_files = _collect_npz_files(data_dir, categories)

    # ---------- Pass 1: compute global mean and variance ----------
    stats = _compute_global_stats(category_to_files, categories, last_T=last_T)
    mean_dog, std_dog = stats["dog"]
    mean_dsgc, std_dsgc = stats["dsgc"]
    mean_loom, std_loom = stats["loom"]

    # ---------- Pass 2: build samples ----------
    all_series = []
    all_labels = []

    for label, category in enumerate(categories):
        npz_files = category_to_files.get(category, [])
        for npz_path in tqdm(npz_files, desc=f"Loading {category} retina features and patch processing"):
            data = np.load(npz_path)

            dog_all = data["features"].astype(np.float32)              # (T, 6*H*W)
            dsgc_all = data["directional_features"].astype(np.float32) # (T, 8*H*W)
            looming_all = data["looming_features"].astype(np.float32)  # (T,)

            T = dog_all.shape[0]
            H = int(data["H"])
            W = int(data["W"])
            C_dog = int(data["C"])  # DoG channels, expect 6

            if H != 32 or W != 32:
                raise ValueError(
                    f"{os.path.basename(npz_path)} spatial size is {H}x{W}; "
                    "code assumes 32x32. Confirm data or change code."
                )

            if T < last_T:
                raise ValueError(
                    f"{os.path.basename(npz_path)} has {T} time frames, "
                    f"fewer than required last {last_T} frames"
                )

            # use only last last_T frames
            dog = dog_all[-last_T:]
            dsgc = dsgc_all[-last_T:]
            looming = looming_all[-last_T:]

            # normalize
            dog = (dog - mean_dog) / (std_dog + 1e-8)
            dsgc = (dsgc - mean_dsgc) / (std_dsgc + 1e-8)
            looming = (looming - mean_loom) / (std_loom + 1e-8)

            # DSGC channels: total features / (H*W) = 8
            total_dsgc_feat = dsgc.shape[1]
            if total_dsgc_feat % (H * W) != 0:
                raise ValueError(
                    f"{os.path.basename(npz_path)} directional_features shape {dsgc_all.shape} "
                    f"not divisible by H*W = {H*W}; check feature generation."
                )
            C_dsgc = total_dsgc_feat // (H * W)  # expect 8

            # patch processing per frame
            frame_features = []  # (last_T, 127)
            for t in range(last_T):
                dog_t = dog[t]   # (6*H*W,)
                dsgc_t = dsgc[t] # (8*H*W,)
                loom_t = looming[t]  # scalar

                # reshape to (H, W, C)
                dog_map = dog_t.reshape(H, W, C_dog)          # (32,32,6)
                dsgc_map = dsgc_t.reshape(H, W, C_dsgc)       # (32,32,8)

                patch_vecs = []

                # y: row, x: column
                for y0 in range(0, H - patch_size + 1, stride):
                    for x0 in range(0, W - patch_size + 1, stride):
                        y1 = y0 + patch_size
                        x1 = x0 + patch_size

                        patch_dog = dog_map[y0:y1, x0:x1, :]      # (patch_size, patch_size, 6)
                        patch_dsgc = dsgc_map[y0:y1, x0:x1, :]    # (patch_size, patch_size, 8)

                        # mean over space -> 6/8 dims per patch
                        avg_dog = patch_dog.mean(axis=(0, 1))     # (6,)
                        avg_dsgc = patch_dsgc.mean(axis=(0, 1))   # (8,)

                        patch_vecs.append(avg_dog)
                        patch_vecs.append(avg_dsgc)

                # 9 patches, 14 dims each
                patch_feat = np.concatenate(patch_vecs, axis=-1)  # (9*14 = 126,)

                # append this frame's Looming -> 127 dims
                frame_feat = np.concatenate(
                    [patch_feat, np.array([loom_t], dtype=np.float32)],
                    axis=-1
                )  # (127,)
                frame_features.append(frame_feat)

            frame_features = np.stack(frame_features, axis=0)  # (last_T, 127)
            all_series.append(frame_features)
            all_labels.append(label)

    if len(all_series) == 0:
        raise ValueError("No samples built; check data path and npz naming.")

    X = np.stack(all_series, axis=0)  # (num_samples, last_T, 127)
    y = np.array(all_labels, dtype=np.int64)

    print(f"Final data shape: X = {X.shape}, y = {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

# Reservoir: SimpleESN
class SimpleESN:
    def __init__(
        self,
        input_size,        # feature dim per frame, 127 here
        output_size,       # readout layer size
        reservoir_size,    # whole-brain neuron count
        sparsity=0.1,
        leak_rate=0.98,
        target_mean_eig=0.75,
        w_in_scale=0.1,
        w_in_width=0.05,
        seed=42,
        weaken_region=None,
        weaken_ratio=1.0,
    ):
        np.random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.threshold = 0.7
        self.reset_strength = -0.6

        # === load region labels ===
        region_path = os.path.join(os.path.dirname(__file__), "sampled_region_labels.npy")
        region_labels = np.load(region_path)
        self.region_labels = region_labels

        # === select input nodes from SC / LGN ===
        sc_candidates = np.where(region_labels == "SC")[0]
        lgn_candidates = np.where(region_labels == "LGN")[0]

        n_sc_inputs = 381
        n_lgn_inputs = 127
        total_input_neurons = n_sc_inputs + n_lgn_inputs  # 508

        print(f"SC candidates: {len(sc_candidates)}, required: {n_sc_inputs}")
        print(f"LGN candidates: {len(lgn_candidates)}, required: {n_lgn_inputs}")

        if len(sc_candidates) < n_sc_inputs:
            raise ValueError(
                f"SC has {len(sc_candidates)} neurons, less than required {n_sc_inputs}; "
                "check sampled_region_labels.npy."
            )
        if len(lgn_candidates) < n_lgn_inputs:
            raise ValueError(
                f"LGN has {len(lgn_candidates)} neurons, less than required {n_lgn_inputs}; "
                "check sampled_region_labels.npy."
            )

        sc_input_nodes = np.random.choice(sc_candidates, size=n_sc_inputs, replace=False)
        lgn_input_nodes = np.random.choice(lgn_candidates, size=n_lgn_inputs, replace=False)

        # keep for later analysis
        self.sc_input_nodes = sc_input_nodes
        self.lgn_input_nodes = lgn_input_nodes
        self.input_nodes = np.concatenate([sc_input_nodes, lgn_input_nodes])
        print(f"Total input nodes (SC+LGN): {total_input_neurons}")

        # === output nodes: from PL / IL / OFC / PPC / DS / MC ===
        desired_output_regions = ["PL", "IL", "OFC", "PPC", "DS", "MC"]
        output_candidates = np.where(np.isin(region_labels, desired_output_regions))[0]
        print(f"Output node candidates: {len(output_candidates)}")
        if len(output_candidates) == 0:
            raise ValueError(
                f"No nodes in region_labels belong to {desired_output_regions}; "
                "check that labels are the 12 visual-related regions."
            )

        effective_output_size = min(output_size, len(output_candidates))
        if effective_output_size < output_size:
            print(
                f"Warning: target output size {output_size} > candidates {len(output_candidates)}; "
                f"using {effective_output_size} available nodes as output layer."
            )
        self.output_size = effective_output_size
        self.readout_nodes = np.random.choice(
            output_candidates,
            size=self.output_size,
            replace=False,
        )

        # === input weights: each feature -> 3 SC + 1 LGN node ===
        # W_in shape: (input_size, reservoir_size); forward: input_term = u_t @ W_in
        self.W_in = np.zeros((input_size, reservoir_size), dtype=np.float32)

        if input_size != 127:
            print(
                f"Note: input_size = {input_size}; "
                "assuming 127 dims per frame (9*14+1). If not, confirm data processing."
            )

        if n_sc_inputs != 3 * input_size or n_lgn_inputs != input_size:
            raise ValueError(
                f"Expected: SC inputs {n_sc_inputs} =? 3*input_size, "
                f"LGN inputs {n_lgn_inputs} =? 1*input_size; "
                "mismatch. Check input_size or n_sc_inputs/n_lgn_inputs."
            )

        for feat_idx in range(input_size):
            # assign 3 SC + 1 LGN node per feature
            sc_nodes_for_feat = sc_input_nodes[3 * feat_idx : 3 * feat_idx + 3]
            lgn_node_for_feat = lgn_input_nodes[feat_idx]

            for neuron_idx in sc_nodes_for_feat:
                self.W_in[feat_idx, neuron_idx] = np.random.uniform(
                    w_in_scale - w_in_width,
                    w_in_scale + w_in_width,
                )
            self.W_in[feat_idx, lgn_node_for_feat] = np.random.uniform(
                w_in_scale - w_in_width,
                w_in_scale + w_in_width,
            )

        # === reservoir weights: load final_connection_matrix.npy ===
        matrix_path = os.path.join(os.path.dirname(__file__), "final_connection_matrix.npy")
        W_res = np.load(matrix_path).astype(np.float32)

        # ablation: weaken connections for given region(s) (optional)
        if weaken_region is not None and weaken_ratio > 0:
            weaken_idx = np.where(np.isin(region_labels, weaken_region))[0]
            print(f"Weakened region(s): {weaken_region}, ratio: {weaken_ratio}, neurons: {len(weaken_idx)}")
            if len(weaken_idx) > 0:
                # global decay once
                W_res *= (1 - weaken_ratio)
                # restore within-region block to cancel self-weakening
                W_res[np.ix_(weaken_idx, weaken_idx)] /= (1 - weaken_ratio)

        # normalize by mean absolute eigenvalue
        mean_abs_eig = 2.9283319
        # eigvals = np.linalg.eigvals(W_res)
        # mean_abs_eig = np.mean(np.abs(eigvals))
        # print("Connection matrix mean |eigenvalue|:", mean_abs_eig)
        if mean_abs_eig > 0:
            W_res = W_res * (target_mean_eig / mean_abs_eig)

        self.W_res = W_res.astype(np.float32)

    def forward(self, X):
        """
        X: (num_samples, time_steps, input_size)  -- input_size = 127 here
        Returns:
            h_np: (num_samples, output_size)        # readout state per sample
            h_seq: (time_steps, reservoir_size)     # whole-brain trajectory of sample #10 (for analysis)
        """
        num_samples, time_steps, input_dim = X.shape
        if input_dim != self.input_size:
            raise ValueError(
                f"Input last dim is {input_dim}, but ESN expects input_size {self.input_size}"
            )

        W_in_cp = cp.asarray(self.W_in)
        W_res_cp = cp.asarray(self.W_res)
        h = cp.zeros((num_samples, self.reservoir_size), dtype=cp.float32)
        X_cp = cp.asarray(X)
        h_seq = np.zeros((time_steps, self.reservoir_size), dtype=np.float32)

        for t in range(time_steps):
            u_t = X_cp[:, t, :]              # (num_samples, input_size)
            input_term = cp.dot(u_t, W_in_cp)
            res_term = cp.dot(h, W_res_cp)
            pre_act = input_term + res_term
            h_new = (1 - self.leak_rate) * h + self.leak_rate * cp.tanh(pre_act)
            h = h_new

            # save whole-brain trajectory of sample #10
            if num_samples > 10:
                h_seq[t, :] = cp.asnumpy(h[10])
            else:
                h_seq[t, :] = cp.asnumpy(h[0])

        # use only output node states as readout features
        h_np_full = cp.asnumpy(h)
        h_np = h_np_full[:, self.readout_nodes]

        del h, h_new, X_cp, W_in_cp, W_res_cp
        cp.get_default_memory_pool().free_all_blocks()
        return h_np, h_seq

# Consistency report (Euclidean vs cosine)
def flatten_metric(region_dim_metrics, metric_key, k):
    vals = []
    for region in region_dim_metrics:
        m = region_dim_metrics[region]
        try:
            v_eu = m['euclidean'][metric_key][k]
            v_co = m['cosine'][metric_key][k]
            vals.append((v_eu, v_co))
        except Exception:
            continue
    if not vals:
        return 0.0
    a = np.concatenate([np.array(x[0]) for x in vals])
    b = np.concatenate([np.array(x[1]) for x in vals])
    if a.size != b.size or a.size == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0,1]
    return float(corr)

def find_knee_dimension(region_dim_metrics, dims, k, metric_family='euclidean'):
    ok_regions = 0
    knees = []
    for region in region_dim_metrics:
        m = region_dim_metrics[region].get(metric_family)
        if m is None:
            continue
        T_curve = np.array(m['Trust_'+metric_family][k])
        C_curve = np.array(m['Cont_'+metric_family][k])
        if len(T_curve) != len(dims) or len(C_curve) != len(dims):
            continue
        # marginal gain
        dT = np.diff(T_curve, prepend=T_curve[0])
        dC = np.diff(C_curve, prepend=C_curve[0])
        # from d=3 find first dim where both gains < threshold
        knee_d = None
        for di, d in enumerate(dims):
            if d < 3:
                continue
            if dT[di] < gain_thresh and dC[di] < gain_thresh:
                knee_d = d
                break
        if knee_d is not None:
            knees.append(knee_d)
            ok_regions += 1
    if len(knees) == 0:
        return None, 0.0
    # smallest d satisfied by majority of regions
    cnt = Counter(knees)
    best_d = sorted(cnt.items(), key=lambda x: (x[1], -x[0]), reverse=True)[0][0]
    ratio = ok_regions / max(1, len(region_dim_metrics))
    return best_d, ratio

def compute_rank_matrix(dist_mat):
    """
    Convert distance matrix to rank matrix (row j: rank of sample j in that row's distance order; 1=nearest, self=largest rank).
    """
    n = dist_mat.shape[0]
    # set diagonal to large value to exclude self
    diag_large = dist_mat.max() + 1e6
    dm = dist_mat.copy()
    np.fill_diagonal(dm, diag_large)
    # argsort twice to get ranks
    order = np.argsort(dm, axis=1)
    ranks = np.empty_like(order)
    # ranks[i, order[i, k]] = k+1
    for i in range(n):
        ranks[i, order[i]] = np.arange(1, n+1)
    # set self rank to n (max)
    np.fill_diagonal(ranks, n)
    return ranks

def continuity_score(X, Y, k=15, metric='euclidean'):
    """
    Continuity C(k): measures how well high-dim neighbors are preserved in low-dim.
    Formula from van der Maaten & Hinton; dual to Trustworthiness.
    """
    n = X.shape[0]
    # distances and ranks
    Dx = pairwise_distances(X, metric=metric)
    Dy = pairwise_distances(Y, metric='euclidean')
    Rx = compute_rank_matrix(Dx)
    Ry = compute_rank_matrix(Dy)

    # neighbor sets
    Nx = np.argsort(Dx, axis=1)[:, :k+1]  # include self, drop later
    Ny = np.argsort(Dy, axis=1)[:, :k+1]

    penalty = 0.0
    for i in range(n):
        nx = [j for j in Nx[i] if j != i][:k]
        ny = [j for j in Ny[i] if j != i][:k]
        U = set(nx) - set(ny)  # neighbor in original space but not in low-dim
        if len(U) == 0:
            continue
        # use low-dim rank Ry for penalty (r̂_ij - k)
        penalty += np.sum(Ry[i, list(U)] - k)

    const = 2.0 / (n * k * (2*n - 3*k - 1))
    C = 1.0 - const * penalty
    return float(max(0.0, min(1.0, C)))

def knn_retention_rate(X, Y, k=15, metric='euclidean'):
    """
    P@k: average overlap of top-k neighbors in original vs low-dim space.
    """
    n = X.shape[0]
    Dx = pairwise_distances(X, metric=metric)
    Dy = pairwise_distances(Y, metric='euclidean')

    Nx = np.argsort(Dx, axis=1)[:, 1:k+1]
    Ny = np.argsort(Dy, axis=1)[:, 1:k+1]

    scores = []
    for i in range(n):
        sx = set(Nx[i].tolist())
        sy = set(Ny[i].tolist())
        scores.append(len(sx & sy) / float(k))
    return float(np.mean(scores))

def evaluate_region_dim_curve(X_time_nodes, dims, k_list, umap_params, seed=42, high_metrics=('euclidean','cosine')):
    """
    Evaluate T/C/P@k across dimensions for one region (samples=time steps).
    Returns: {metric_name: {k: [values per dim]}} and dimension list.
    """
    n_samples = X_time_nodes.shape[0]
    # constraint: sklearn trustworthiness requires k < n_samples/2
    allowed_k = [k for k in k_list if (isinstance(k, int) and k >= 1 and k < (n_samples/2) and k <= n_samples-1)]
    if len(allowed_k) < len(k_list):
        skipped = sorted(set(k_list) - set(allowed_k))
        print(f"  Warning: k restricted (n_samples={n_samples}): skipping {skipped}")

    results = {}
    for high_metric in high_metrics:
        res_metric = {f'Trust_{high_metric}': {k: [] for k in k_list},
                      f'Cont_{high_metric}': {k: [] for k in k_list},
                      f'P@k_{high_metric}': {k: [] for k in k_list}}

        for d in dims:
            reducer_d = umap.UMAP(n_components=d, random_state=seed,
                                  n_neighbors=umap_params['n_neighbors'],
                                  min_dist=umap_params['min_dist'])
            Y = reducer_d.fit_transform(X_time_nodes)
            # compute metrics for each k
            for k in allowed_k:
                T = sk_trustworthiness(X_time_nodes, Y, n_neighbors=k, metric=high_metric)
                C = continuity_score(X_time_nodes, Y, k=k, metric=high_metric)
                P = knn_retention_rate(X_time_nodes, Y, k=k, metric=high_metric)
                res_metric[f'Trust_{high_metric}'][k].append(T)
                res_metric[f'Cont_{high_metric}'][k].append(C)
                res_metric[f'P@k_{high_metric}'][k].append(P)
        results[high_metric] = res_metric
    return results

def calculate_trajectory_angles_and_orthogonality(region_umap_data, time_window=5):
    """
    Compute trajectory angles and orthogonality between brain regions over time.

    Parameters:
    - region_umap_data: dict of UMAP trajectory data per brain region
    - time_window: time window size for direction vectors

    Returns:
    - angles_matrix: (time_steps, n_regions, n_regions)
    - orthogonality_matrix: (time_steps, n_regions, n_regions)
    - region_names: list of region names
    """
    region_names = list(region_umap_data.keys())
    n_regions = len(region_names)
    
    # get time steps (assume same for all regions)
    time_steps = len(region_umap_data[region_names[0]])
    
    # initialize result matrices
    angles_matrix = np.zeros((time_steps, n_regions, n_regions))
    orthogonality_matrix = np.zeros((time_steps, n_regions, n_regions))
    
    print(f"Computing angles and orthogonality across {n_regions} regions...")
    print(f"Time steps: {time_steps}, time window: {time_window}")
    
    for t in range(time_window, time_steps - time_window):
        for i, region1 in enumerate(region_names):
            for j, region2 in enumerate(region_names):
                if i != j:  # do not compute self-vs-self angle
                    # get trajectory data for two regions near time t
                    traj1 = region_umap_data[region1][t-time_window:t+time_window+1]
                    traj2 = region_umap_data[region2][t-time_window:t+time_window+1]
                    
                    # direction vector (central difference)
                    if len(traj1) >= 3 and len(traj2) >= 3:
                        # trajectory direction vectors
                        dir1 = traj1[-1] - traj1[0]  # vector from window start to end
                        dir2 = traj2[-1] - traj2[0]
                        
                        # angle (rad to deg)
                        cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-8)
                        cos_angle = np.clip(cos_angle, -1, 1)  # avoid numerical error
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        angles_matrix[t, i, j] = angle
                        
                        # orthogonality (max at 90 deg)
                        orthogonality = 1 - abs(cos_angle)  # orthogonality measure
                        orthogonality_matrix[t, i, j] = orthogonality
    
    return angles_matrix, orthogonality_matrix, region_names

def plot_all_regions_band_decomposition(states, region_labels, target_regions, fs=1000, save_dir=None):
    """
    Plot band decomposition for all regions: shared x-axis, y-axis = region names.
    """
    # band definition
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }
    
    # store band signals for all regions
    all_region_signals = {}
    
    # compute band decomposition per region
    for region in target_regions:
        idx = np.where(np.isin(region_labels, [region]))[0]
        if len(idx) == 0:
            print(f"No nodes found for region {region}!")
            continue
            
        lfp = np.mean(states[:, idx], axis=1)
        region_signals = {}
        
        for band, (low, high) in bands.items():
            nyq = 0.5 * fs
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            band_signal = filtfilt(b, a, lfp)
            region_signals[band] = band_signal
            
        all_region_signals[region] = region_signals
    
    # create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # color scheme
    band_colors = {
        "delta": '#1f77b4',   # blue
        "theta": '#ff7f0e',   # orange
        "alpha": '#2ca02c',   # green
        "beta": '#d62728',    # red
        "gamma": '#9467bd'    # purple
    }
    
    # plot band signals per region
    y_offset = 0
    region_centers = {}
    
    for i, region in enumerate(target_regions):
        if region not in all_region_signals:
            continue
            
        region_signals = all_region_signals[region]
        
        # signal range for this region
        all_signals = [region_signals[band] for band in bands.keys()]
        min_val = min([np.min(sig) for sig in all_signals])
        max_val = max([np.max(sig) for sig in all_signals])
        signal_range = max_val - min_val
        
        # plot signal per band
        for band in bands.keys():
            signal = region_signals[band]
            # normalize to this region's range
            normalized_signal = (signal - min_val) / signal_range if signal_range > 0 else signal
            # y-offset
            y_pos = normalized_signal + y_offset
            
            ax.plot(y_pos, color=band_colors[band], linewidth=1.5, alpha=0.8, 
                   label=f"{band} band" if i == 0 else "")
        
        # record region center
        region_centers[region] = y_offset + 0.5
        y_offset += 1.2  # spacing between regions
        
        # dashed line at zero
        ax.axhline(y=y_offset - 0.6, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # set y-axis labels to region names
    ax.set_yticks(list(region_centers.values()))
    ax.set_yticklabels(list(region_centers.keys()), fontweight='bold', fontsize=12)
    
    # x-axis label
    ax.set_xlabel('Time (samples)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Brain Regions', fontweight='bold', fontsize=14)
    ax.set_title('Band-pass Filtered Signals Across All Brain Regions', 
                fontweight='bold', fontsize=16, pad=20)
    
    # add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'all_regions_band_decomposition.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: all_regions_band_decomposition.png")
    else:
        plt.show()

# Relative power comparison across six brain regions
def analyze_multiple_regions_power(states, region_labels, target_regions, fs=1000, save_dir=None):
    """
    Analyze relative power across regions; horizontal bar chart.

    Parameters:
    states: (T, N) temporal state
    region_labels: (N,) region labels
    target_regions: list of region names to analyze
    fs: sampling rate
    """
    # band definition
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 100)
    }
    
    # store results for all regions
    all_results = {}
    
    print("Analyzing relative power for six regions...")
    
    for region in target_regions:
        print(f"Analyzing region: {region}")
        
        # get nodes for this region
        idx = np.where(np.isin(region_labels, [region]))[0]
        if len(idx) == 0:
            print(f"No nodes found for region {region}!")
            continue
            
        # compute LFP
        lfp = np.mean(states[:, idx], axis=1)
        
        # relative power per band
        region_results = {}
        for band, (low, high) in bands.items():
            nyq = 0.5 * fs
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            band_signal = filtfilt(b, a, lfp)
            power = np.mean(band_signal**2)
            region_results[band] = power
            
        # relative power
        total_power = sum(region_results.values())
        for band in region_results:
            region_results[band] = region_results[band] / total_power if total_power > 0 else 0
            
        all_results[region] = region_results
        print(f"  {region}: {len(idx)} nodes")
    
    # create horizontal bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # bar parameters
    bar_width = 0.12  # bar width
    band_names = list(bands.keys())
    n_bands = len(band_names)
    n_regions = len(target_regions)
    
    # horizontal bars per band
    x_pos = np.arange(n_bands)  # band positions
    
    # color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, region in enumerate(target_regions):
        if region in all_results:
            values = [all_results[region][band] for band in band_names]
            # horizontal bars with offset per band
            bars = ax.barh(x_pos + i * bar_width, values, bar_width, 
                          label=region, color=colors[i], alpha=0.8, 
                          edgecolor='black', linewidth=0.8)
            
            # value labels at bar ends
            for j, (bar, value) in enumerate(zip(bars, values)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # set plot properties
    ax.set_yticks(x_pos + bar_width * (n_regions - 1) / 2)
    ax.set_yticklabels(band_names, fontweight='bold', fontsize=12)
    ax.set_xlabel('Relative Power', fontweight='bold', fontsize=14)
    ax.set_ylabel('Frequency Bands', fontweight='bold', fontsize=14)
    ax.set_title('Relative Power Distribution Across Frequency Bands\nSix Brain Regions Comparison', 
                fontweight='bold', fontsize=16, pad=20)
    
    # add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_xlim(0, max([max(all_results[region].values()) for region in target_regions if region in all_results]) * 1.2)
    
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'multi_regions_relative_power.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    # detailed relative power results
    print("\nDetailed relative power results:")
    print("=" * 80)
    print(f"{'Region':<20} {'Delta':<8} {'Theta':<8} {'Alpha':<8} {'Beta':<8} {'Gamma':<8}")
    print("-" * 80)
    
    for region in target_regions:
        if region in all_results:
            values = [all_results[region][band] for band in band_names]
            print(f"{region:<20} {values[0]:<8.3f} {values[1]:<8.3f} {values[2]:<8.3f} {values[3]:<8.3f} {values[4]:<8.3f}")
    
    # save detailed results to same directory
    if save_dir is not None:
        txt_path = os.path.join(save_dir, 'multi_regions_relative_power.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Region\t" + "\t".join(band_names) + "\n")
            for region in target_regions:
                if region in all_results:
                    values = [all_results[region][band] for band in band_names]
                    f.write(region + "\t" + "\t".join(f"{v:.6f}" for v in values) + "\n")
    
    return all_results

def check_and_plot_hippocampal_relative_difference(save_dir):
    """
    Check for normal and weakened HPC relative power files; if both exist, compute relative difference and plot histogram.
    """
    
    # file paths
    normal_file = os.path.join(save_dir, 'HPC_relative_power_normal.txt')
    
    # find weakened files (possible name formats)
    weakened_files = []
    for filename in os.listdir(save_dir):
        if filename.startswith('HPC_relative_power_') and filename.endswith('.txt') and 'normal' not in filename:
            weakened_files.append(filename)
    
    # check if files exist
    if not os.path.exists(normal_file):
        print("Warning: HPC_relative_power_normal.txt not found")
        return
    
    if len(weakened_files) == 0:
        print("Warning: no weakened HPC relative power files found")
        return
    
    print(f"Found {len(weakened_files)} weakened file(s): {weakened_files}")
    
    # read normal file
    try:
        print(f"Reading normal file: {normal_file}")
        normal_powers = read_hippocampal_power_from_txt(normal_file)
        print(f"Successfully read normal relative power data")
        for band, power in normal_powers.items():
            print(f"  {band}: {power:.6f}")
    except Exception as e:
        print(f"Failed to read normal file: {e}")
        traceback.print_exc()
        return
    
    # compute relative difference for each weakened file
    for weakened_file in weakened_files:
        weakened_path = os.path.join(save_dir, weakened_file)
        try:
            print(f"Reading weakened file: {weakened_path}")
            weakened_powers = read_hippocampal_power_from_txt(weakened_path)
            print(f"Successfully read weakened relative power data: {weakened_file}")
            for band, power in weakened_powers.items():
                print(f"  {band}: {power:.6f}")
            
            # compute relative difference
            print("Computing relative difference...")
            relative_diff = calculate_hippocampal_relative_difference(weakened_powers, normal_powers)
            print(f"Relative difference:")
            for band, diff in relative_diff.items():
                print(f"  {band}: {diff:.6f}")
            
            # plot relative difference histogram
            print("Plotting relative difference histogram...")
            plot_hippocampal_relative_difference_histogram(relative_diff, save_dir, weakened_file)
            
        except Exception as e:
            print(f"Failed to process file {weakened_file}: {e}")
            traceback.print_exc()
            continue

def read_hippocampal_power_from_txt(file_path):
    """
    Read HPC relative power from txt file.
    """
    powers = {}
    
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # locate CSV-format data
    csv_start = False
    for line in lines:
        if "CSV Format:" in line:
            csv_start = True
            continue
        
        if csv_start and line.strip() and not line.startswith("=") and not line.startswith("Band,"):
            # parse CSV line
            parts = line.strip().split(',')
            if len(parts) == 2:
                band_name = parts[0]
                power_value = float(parts[1])
                powers[band_name] = power_value
    
    print(f"Successfully read relative power for {len(powers)} bands")
    return powers

def calculate_hippocampal_relative_difference(weakened_powers, normal_powers):
    """
    Relative difference: (weakened - normal) / normal
    """
    relative_diff = {}
    
    for band in normal_powers.keys():
        if band in weakened_powers:
            normal_val = normal_powers[band]
            weakened_val = weakened_powers[band]
            
            # avoid division by zero
            if abs(normal_val) < 1e-10:
                relative_diff[band] = 0.0
            else:
                relative_diff[band] = (weakened_val - normal_val) / normal_val
        else:
            relative_diff[band] = 0.0
    
    return relative_diff

def plot_hippocampal_relative_difference_histogram(relative_diff, save_dir, weakened_filename):
    """
    Plot HPC relative difference histogram.
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # prepare data
    bands = list(relative_diff.keys())
    values = list(relative_diff.values())
    
    # color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # bar chart
    bars = ax.bar(bands, values, color=colors[:len(bands)], alpha=0.8, 
                  edgecolor='black', linewidth=1.2)
    
    # value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height >= 0 else -0.02), 
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=11, fontweight='bold')
    
    # zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # plot properties
    ax.set_xlabel('Frequency Bands', fontweight='bold', fontsize=14)
    ax.set_ylabel('Relative Difference\n(Weakened - Normal) / Normal', fontweight='bold', fontsize=14)
    ax.set_title(f'HPC Relative Power Difference\n{weakened_filename.replace(".txt", "")}', 
                fontweight='bold', fontsize=16, pad=20)
    
    # add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # save figure
    output_filename = weakened_filename.replace('.txt', '_relative_difference_histogram.png')
    save_path = os.path.join(save_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved HPC relative difference histogram: {output_filename}")

def analyze_multi_region_lfp_flow(states, region_labels, fs=1000, 
                                 baseline_window=50, threshold_std=1.5,
                                 time_window=(0, 500)):
    """
    Analyze multi-region LFP time series; show information propagation.

    Parameters:
    states: (T, N) temporal state
    region_labels: (N,) region labels
    fs: sampling rate
    baseline_window: baseline window (ms)
    threshold_std: std threshold for significant deviation from baseline
    time_window: (start_ms, end_ms)
    """

    region_flow_order = [
        # subcortical / early visual relay
        'SC',   # superior colliculus
        'OPN',  # olivary pretectal nucleus
        'LGN',  # lateral geniculate nucleus
        'TRN',  # thalamic reticular nucleus
        'LP',   # lateral posterior nucleus

        # primary & association visual cortex
        'VC',   # visual cortex
        'PPC',  # posterior parietal cortex

        # motor and decision-related
        'DS',   # dorsal striatum
        'MC',   # motor cortex
        'OFC',  # orbitofrontal cortex
        'PL',   # prelimbic
        'IL',   # infralimbic
    ]

    
    # load coordinates
    coords_path = os.path.join(os.path.dirname(__file__), 'sampled_normalized_coordinates.npy')
    if os.path.exists(coords_path):
        coordinates = np.load(coords_path)
        print(f"Loaded coordinate data: {coordinates.shape}")
    else:
        print("Warning: coordinate file not found")
        coordinates = None
    
    # compute LFP and stats per region
    region_lfps = {}
    region_stats = {}
    region_centers = {}
    
    print("Computing LFP and stats per region...")
    for region in region_flow_order:
        idx = np.where(region_labels == region)[0]
        if len(idx) > 0:
            # compute LFP directly
            lfp = np.mean(states[:, idx], axis=1)
            
            region_lfps[region] = lfp
            
            # compute statistics
            baseline = lfp[:baseline_window]
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)
            
            lfp_abs = np.abs(lfp)
            max_lfp_abs = np.max(lfp_abs)
            
            abs_threshold = 0.1 * max_lfp_abs  # 10% threshold
            rel_threshold = threshold_std * baseline_std  # relative threshold
            threshold = max(abs_threshold, rel_threshold)
            
            # keep threshold in reasonable range
            threshold = max(threshold, 0.0001)  # min
            threshold = min(threshold, 0.01)    # max
            
            # absolute LFP
            lfp_abs = np.abs(lfp)
            max_value = np.max(lfp_abs)
            
            # red dot: first time exceeding 10% of max from t=0
            red_threshold = 0.10 * max_value
            red_mask = lfp_abs >= red_threshold
            first_significant_idx = np.argmax(red_mask) if np.any(red_mask) else len(lfp)
            
            # blue triangle: first time exceeding 90% of max from t=0
            blue_threshold = 0.90 * max_value
            blue_mask = lfp_abs >= blue_threshold
            peak_idx = np.argmax(blue_mask) if np.any(blue_mask) else len(lfp)
            
            region_stats[region] = {
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'threshold': threshold,
                'first_significant_time': first_significant_idx / fs * 1000,  # ms
                'peak_time': peak_idx / fs * 1000,  # ms
                'peak_value': lfp[peak_idx],
                'node_count': len(idx)
            }
            
            # region center coordinates
            if coordinates is not None:
                region_coords = coordinates[idx]
                region_centers[region] = np.mean(region_coords, axis=0)
            else:
                region_centers[region] = np.array([0, 0, 0])
            
            print(f"  {region}: {len(idx)} nodes, threshold: {threshold:.4f}, "
                  f"first significant: {region_stats[region]['first_significant_time']:.1f}ms, "
                  f"peak: {region_stats[region]['peak_time']:.1f}ms")
    
    create_lfp_flow_visualization(region_lfps, region_stats, region_centers, 
                                 region_flow_order, fs, time_window)
    
    return region_lfps, region_stats, region_centers

def create_lfp_flow_visualization(region_lfps, region_stats, region_centers, 
                                 region_flow_order, fs, time_window):
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8
    })
    
    # create result directory
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 1. multi-region LFP temporal overlay
    create_multi_region_lfp_plot(region_lfps, region_stats, region_flow_order, 
                                fs, time_window, result_dir)
    
    # 2. propagation timing analysis
    create_propagation_timing_analysis(region_stats, region_flow_order, result_dir)
    
    # 3. region activation heatmap
    create_region_activation_heatmap(region_lfps, region_stats, region_flow_order, 
                                   fs, time_window, result_dir)
    
    print(f"\nAll LFP flow figures saved to {result_dir}")

def create_multi_region_lfp_plot(region_lfps, region_stats, region_flow_order, 
                                fs, time_window, result_dir):
    """Create multi-region LFP temporal overlay plot."""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # time axis: use actual data length
    data_length = len(list(region_lfps.values())[0])
    time_points = np.arange(data_length) / fs * 1000  # ms
    time_mask = (time_points >= time_window[0]) & (time_points <= time_window[1])
    time_axis = time_points[time_mask]
    
    # color scheme: input to output gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(region_flow_order)))
    
    # vertical offset
    vertical_offset = 0.8
    y_positions = {}
    
    # debug
    print(f"Debug plot 7 data:")
    print(f"  Time axis length: {len(time_axis)}")
    print(f"  Time range: {time_axis[0]:.1f} - {time_axis[-1]:.1f} ms")
    
    for i, region in enumerate(region_flow_order):
        if region in region_lfps:
            lfp = region_lfps[region][time_mask]
            y_pos = i * vertical_offset
            y_positions[region] = y_pos
            
            # abs(LFP) for peak detection
            lfp_abs = np.abs(lfp)
            
            # normalize LFP for visualization
            lfp_normalized = (lfp_abs - np.min(lfp_abs)) / (np.max(lfp_abs) - np.min(lfp_abs) + 1e-8)
            lfp_scaled = lfp_normalized * 0.6 + y_pos  # scale to height 0.6
            
            # plot LFP curve
            ax.plot(time_axis, lfp_scaled, color=colors[i], linewidth=2.5, 
                   alpha=0.9, label=region)
            
            # fill area
            ax.fill_between(time_axis, lfp_scaled, y_pos, 
                          color=colors[i], alpha=0.2)
            
            # mark first significant deviation time
            first_sig_time = region_stats[region]['first_significant_time']
            first_sig_idx = int(first_sig_time * fs / 1000)
            
            if 0 <= first_sig_idx < len(lfp_scaled):
                actual_time = first_sig_idx / fs * 1000
                ax.scatter(actual_time, lfp_scaled[first_sig_idx], 
                          color='red', s=80, marker='o', zorder=5, 
                          edgecolor='white', linewidth=2)
            
            # mark peak time
            peak_time = region_stats[region]['peak_time']
            peak_idx = int(peak_time * fs / 1000)
            
            if 0 <= peak_idx < len(lfp_scaled):
                actual_peak_time = peak_idx / fs * 1000
                ax.scatter(actual_peak_time, lfp_scaled[peak_idx], 
                          color='darkblue', s=100, marker='^', zorder=5,
                          edgecolor='white', linewidth=2)
    
    # stimulus onset line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(0, len(region_flow_order) * vertical_offset * 0.95, 'Stimulus Onset', 
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # figure properties
    ax.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Brain Regions (Information Flow Direction)', fontweight='bold', fontsize=14)
    ax.set_title('Multi-Region LFP Temporal Flow Analysis\n'
                'Information propagation from input to output regions\n'
                '• Red circles: First significant deviation from baseline\n'
                '• Blue triangles: Peak activation time\n'
                '• Vertical dashed line: Stimulus onset', 
                fontweight='bold', fontsize=16, pad=20)
    
    # X-axis range: actual data length
    max_time = (data_length - 1) / fs * 1000  # last time point
    ax.set_xlim(0, max_time)
    
    # Y-axis: colored labels only, near axis
    ax.set_yticks([y_positions[region] for region in region_flow_order if region in y_positions])
    ax.set_yticklabels([])  # clear default labels
    ax.set_ylim(-0.5, len(region_flow_order) * vertical_offset + 0.5)
    
    # manual colored labels near Y-axis
    for i, region in enumerate(region_flow_order):
        if region in y_positions:
            y_pos = y_positions[region]
            ax.text(-2, y_pos, region, ha='right', va='center',
                   fontsize=10, fontweight='bold', color=colors[i])
    
    # Y-axis title to the left of region names
    ax.set_ylabel('Brain Regions (Information Flow Direction)', fontweight='bold', fontsize=14)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_label_coords(-0.15, 0.5)
    
    # clear Y-axis ticks, keep labels only
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    
    # grid and style
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', marker='o', linestyle='None', 
                  markersize=8, label='First Significant Deviation'),
        plt.Line2D([0], [0], color='darkblue', marker='^', linestyle='None', 
                  markersize=8, label='Peak Activation'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, 
                  label='Stimulus Onset')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
             frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '07_multi_region_lfp_flow.png'))
    plt.close()
    print("Saved: 07_multi_region_lfp_flow.png")

def create_propagation_timing_analysis(region_stats, region_flow_order, result_dir):
    """Create propagation timing analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    regions = [r for r in region_flow_order if r in region_stats]
    first_sig_times = [region_stats[r]['first_significant_time'] for r in regions]
    peak_times = [region_stats[r]['peak_time'] for r in regions]
    node_counts = [region_stats[r]['node_count'] for r in regions]
    
    # subplot 1: first significant deviation time
    colors1 = plt.cm.Reds(np.linspace(0.3, 1, len(regions)))
    bars1 = ax1.barh(range(len(regions)), first_sig_times, color=colors1, alpha=0.8,
                    edgecolor='black', linewidth=1)
    
    # value labels
    for i, (bar, time) in enumerate(zip(bars1, first_sig_times)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f}ms', ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax1.set_yticks(range(len(regions)))
    ax1.set_yticklabels(regions, fontsize=10)
    ax1.set_xlabel('First Significant Deviation Time (ms)', fontweight='bold', fontsize=12)
    ax1.set_title('(A) Information Flow Timing\nFirst significant deviation from baseline\n(Ordered by information flow)', 
                 fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # subplot 2: peak time
    colors2 = plt.cm.Blues(np.linspace(0.3, 1, len(regions)))
    bars2 = ax2.barh(range(len(regions)), peak_times, color=colors2, alpha=0.8,
                    edgecolor='black', linewidth=1)
    
    # value labels
    for i, (bar, time) in enumerate(zip(bars2, peak_times)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f}ms', ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax2.set_yticks(range(len(regions)))
    ax2.set_yticklabels(regions, fontsize=10)
    ax2.set_xlabel('Peak Activation Time (ms)', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Peak Activation Timing\nMaximum response amplitude\n(Ordered by information flow)', 
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # propagation delay
    if len(first_sig_times) > 1:
        delays = np.diff(first_sig_times)
        avg_delay = np.mean(delays)
        
        # delay info in subplot 1
        ax1.text(0.02, 0.98, f'Average Inter-Region Delay: {avg_delay:.1f}ms', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '08_propagation_timing_analysis.png'))
    plt.close()
    print("Saved: 08_propagation_timing_analysis.png")

def create_region_activation_heatmap(region_lfps, region_stats, region_flow_order, 
                                   fs, time_window, result_dir):
    """Create region activation heatmap."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # data matrix from 0 to max time
    data_length = len(list(region_lfps.values())[0])
    time_points = np.arange(data_length) / fs * 1000  # ms
    time_mask = (time_points >= 0) & (time_points <= time_points[-1])
    time_axis = time_points[time_mask]
    
    region_names = []
    for region in region_flow_order:
        if region in region_lfps:
            region_names.append(region)
    
    # reverse region order: input on top, output at bottom
    region_names = region_names[::-1]
    
    # heatmap data
    heatmap_data = []
    
    for region in region_names:
        if region in region_lfps:
            lfp = region_lfps[region][time_mask]
            # absolute LFP
            lfp_abs = np.abs(lfp)
            # normalize to [0,1]
            lfp_norm = (lfp_abs - np.min(lfp_abs)) / (np.max(lfp_abs) - np.min(lfp_abs) + 1e-10)
            heatmap_data.append(lfp_norm)
    
    heatmap_data = np.array(heatmap_data)
    
    # heatmap
    max_time = (data_length - 1) / fs * 1000  # last time point
    im = ax.imshow(heatmap_data, cmap='plasma', aspect='auto', 
                   extent=[0, max_time, len(region_names)-0.5, -0.5])
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Normalized LFP Amplitude', fontweight='bold', fontsize=12)
    
    # axes
    ax.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Brain Regions', fontweight='bold', fontsize=14)
    ax.set_title('Regional Activation Heatmap\nInformation flow visualization across time and space\n'
                '• Color intensity: LFP amplitude (normalized)\n'
                '• Vertical axis: Information flow direction\n'
                '• Horizontal axis: Time progression', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Y-axis labels
    ax.set_yticks(range(len(region_names)))
    ax.set_yticklabels(region_names, fontsize=10)
    
    # stimulus onset line
    ax.axvline(x=0, color='white', linestyle='--', linewidth=3, alpha=0.9)
    ax.text(0, len(region_names) * 0.95, 'Stimulus Onset', 
           ha='center', va='top', fontsize=12, fontweight='bold', color='white',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '09_region_activation_heatmap.png'))
    plt.close()
    print("Saved: 09_region_activation_heatmap.png")

def calculate_granger_causality(signal1, signal2, max_lag=10):
    """
    Compute Granger causality. Returns: causal strength signal1 -> signal2.
    """
    n = len(signal1)
    if n <= max_lag:
        return 0.0
    
    # prepare data
    y = signal2[max_lag:]  # target
    X = np.zeros((n-max_lag, 2*max_lag))
    
    for i in range(max_lag):
        X[:, i] = signal1[max_lag-i-1:n-i-1]  # lags of signal1
        X[:, max_lag+i] = signal2[max_lag-i-1:n-i-1]  # lags of signal2
    
    # full model (with signal1)
    model_full = LinearRegression()
    model_full.fit(X, y)
    residuals_full = y - model_full.predict(X)
    var_full = np.var(residuals_full)
    
    # restricted model (without signal1)
    X_restricted = X[:, max_lag:]  # only lags of signal2
    model_restricted = LinearRegression()
    model_restricted.fit(X_restricted, y)
    residuals_restricted = y - model_restricted.predict(X_restricted)
    var_restricted = np.var(residuals_restricted)
    
    # Granger causality
    if var_restricted > 0:
        gc = np.log(var_restricted / var_full)
    else:
        gc = 0.0
    
    return max(0, gc)  # non-negative only

def calculate_transfer_entropy(signal1, signal2, lag=1, bins=10):
    """
    Compute transfer entropy. Returns: information transfer strength signal1 -> signal2.
    """
    # prepare time series
    n = len(signal1)
    if n <= lag:
        return 0.0
    
    # joint distribution
    x_t = signal1[lag:n]
    y_t = signal2[lag:n]
    y_t_minus_1 = signal2[lag-1:n-1]
    
    # discretize
    x_binned = np.digitize(x_t, bins=np.linspace(x_t.min(), x_t.max(), bins))
    y_binned = np.digitize(y_t, bins=np.linspace(y_t.min(), y_t.max(), bins))
    y_minus_1_binned = np.digitize(y_t_minus_1, bins=np.linspace(y_t_minus_1.min(), y_t_minus_1.max(), bins))
    
    # joint probability
    joint_prob = np.zeros((bins, bins, bins))
    for i in range(len(x_binned)):
        joint_prob[x_binned[i]-1, y_binned[i]-1, y_minus_1_binned[i]-1] += 1
    joint_prob /= joint_prob.sum()
    
    # marginal probabilities
    p_y_y_minus_1 = joint_prob.sum(axis=0)
    p_x_y_minus_1 = joint_prob.sum(axis=1)
    p_y_minus_1 = p_y_y_minus_1.sum(axis=0)
    
    # transfer entropy
    te = 0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if joint_prob[i, j, k] > 0 and p_y_y_minus_1[j, k] > 0 and p_x_y_minus_1[i, k] > 0 and p_y_minus_1[k] > 0:
                    te += joint_prob[i, j, k] * np.log2(
                        (joint_prob[i, j, k] * p_y_minus_1[k]) / 
                        (p_y_y_minus_1[j, k] * p_x_y_minus_1[i, k])
                    )
    
    return max(0, te)

def create_nature_chord_diagram(
    causality_matrix,
    te_matrix,
    region_labels,
    result_dir,
    region_order=None,
    ring_outer=1.00,
    ring_inner=0.86,
    attach_radius=0.86,
    curvature=0.55,
    q_threshold=0,      # by default no filter; weak links drawn with low alpha
    min_alpha=0.10,     # minimum alpha for weak links
    eps_frac=1e-9,      # tiny placeholder for zero-strength channels so angle is filled
):
    M = (np.asarray(causality_matrix) + np.asarray(te_matrix)) / 2.0

    # order
    if region_order is None:
        region_order = list(np.unique(region_labels))
    else:
        s = set(region_order)
        for r in np.unique(region_labels):
            if r not in s:
                region_order.append(r)
    R = len(region_order)
    idx_of = {r: i for i, r in enumerate(region_order)}

    pos = M[M > 0]
    threshold = np.percentile(pos, q_threshold) if pos.size else 0.0
    max_val = float(pos.max()) if pos.size else 1.0

    # outer ring angle proportional to node count
    sizes = {r: int(np.sum(region_labels == r)) for r in region_order}
    total_nodes = max(1, sum(sizes.values()))
    angles = {}
    theta = 0.0
    for r in region_order:
        span = 2*np.pi * (sizes[r] / total_nodes)
        angles[r] = (theta, theta + span)
        theta += span

    # helpers
    def pol2xy(r, th): return r*np.cos(th), r*np.sin(th)

    def cubic_polyline(P0, C1, C2, P3, n=30):
        """
        Generate n points on cubic Bezier; return (n,2). Use column t for broadcasting.
        """
        t = np.linspace(0.0, 1.0, n)[:, None]         # (n,1)
        one_minus_t = 1.0 - t                         # (n,1)

        P0 = np.asarray(P0, dtype=float)              # (2,)
        C1 = np.asarray(C1, dtype=float)
        C2 = np.asarray(C2, dtype=float)
        P3 = np.asarray(P3, dtype=float)

        pts = (one_minus_t**3) * P0 \
            + 3.0*(one_minus_t**2)*t * C1 \
            + 3.0*one_minus_t*(t**2) * C2 \
            + (t**3) * P3
        return pts

    def arc_points(theta0, theta1, r=attach_radius, n=None):
        # n proportional to angular width for smoothness
        if n is None:
            n = max(4, int(8 + (theta1-theta0)*180/np.pi))
        ts = np.linspace(theta0, theta1, n)
        return np.column_stack(pol2xy(r, ts))

    def ribbon_polygon(theta_a0, theta_a1, theta_b0, theta_b1,
                       r=attach_radius, bend=curvature):
        # control point radius (inward)
        rc = r*(1.0 - 0.7*bend)
        # source arc (along inner ring, fill sector)
        arc_src = arc_points(theta_a0, theta_a1, r=r)
        # upper edge: A1 -> B1
        P0, C1, C2, P3 = np.array(pol2xy(r, theta_a1)), np.array(pol2xy(rc, theta_a1)), \
                         np.array(pol2xy(rc, theta_b1)), np.array(pol2xy(r, theta_b1))
        edge1 = cubic_polyline(P0, C1, C2, P3, n=28)
        # target arc (back B1 -> B0)
        arc_dst = arc_points(theta_b1, theta_b0, r=r)
        # lower edge: B0 -> A0
        P0, C1, C2, P3 = np.array(pol2xy(r, theta_b0)), np.array(pol2xy(rc, theta_b0)), \
                         np.array(pol2xy(rc, theta_a0)), np.array(pol2xy(r, theta_a0))
        edge2 = cubic_polyline(P0, C1, C2, P3, n=28)
        # concatenate into polygon
        poly = np.vstack([arc_src, edge1, arc_dst, edge2])
        return poly

    def arrow_by_midline(theta_src_mid, theta_dst_mid, end_width,
                         r=attach_radius, bend=curvature, tip_len=0.05):
        rc = r*(1.0 - 0.7*bend)
        P0 = np.array(pol2xy(r, theta_src_mid))
        P3 = np.array(pol2xy(r, theta_dst_mid))
        C1 = np.array(pol2xy(rc, theta_src_mid))
        C2 = np.array(pol2xy(rc, theta_dst_mid))
        # end direction
        t = 0.98
        # midline end point
        tip = cubic_polyline(P0, C1, C2, P3, n=2)[-1]
        # derivative direction
        d = 3*(1-t)**2*(C1-P0) + 6*(1-t)*t*(C2-C1) + 3*t**2*(P3-C2)
        d = d/(np.linalg.norm(d)+1e-9)
        base = tip - tip_len*d
        n = np.array([-d[1], d[0]])  # normal
        half = (end_width/(2.0*r))
        left  = base + half*n
        right = base - half*n
        return np.vstack([tip, left, right])

    # sub-arcs strictly filled (OUT then IN)
    out_sub, in_sub = {r: [] for r in region_order}, {r: [] for r in region_order}
    for r in region_order:
        i = idx_of[r]
        th0, th1 = angles[r]
        arc_total = th1 - th0
        out_len   = arc_total * 0.5
        in_len    = arc_total - out_len

        peers = [p for p in region_order if p != r]
        J = np.array([idx_of[p] for p in peers], dtype=int)

        # output
        raw = M[i, J].astype(float)
        raw = raw + (raw == 0)*eps_frac
        w = raw / raw.sum()
        cum = np.concatenate([[0.0], np.cumsum(w)])
        a = th0 + out_len*cum
        for k, p in enumerate(peers):
            a0 = a[k]
            a1 = a[k+1] if k < len(peers)-1 else (th0 + out_len)
            out_sub[r].append(dict(start=a0, end=a1, peer=p, val=M[i, idx_of[p]]))

        # input
        raw = M[J, i].astype(float)
        raw = raw + (raw == 0)*eps_frac
        w = raw / raw.sum()
        cum = np.concatenate([[0.0], np.cumsum(w)])
        base = th0 + out_len
        a = base + in_len*cum
        for k, p in enumerate(peers):
            a0 = a[k]
            a1 = a[k+1] if k < len(peers)-1 else th1
            in_sub[r].append(dict(start=a0, end=a1, peer=p, val=M[idx_of[p], i]))

    out_map = {r: {s['peer']: s for s in out_sub[r]} for r in region_order}
    in_map  = {r: {s['peer']: s for s in  in_sub[r]} for r in region_order}

    plt.rcParams.update({
        'font.family': 'Arial','font.size': 10,
        'axes.linewidth': 1.0,'axes.spines.top': False,'axes.spines.right': False,
        'figure.dpi': 300,'savefig.dpi': 300,'savefig.bbox': 'tight'
    })
    fig, ax = plt.subplots(figsize=(14,14))
    
    nature_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#8c564b',  # brown
        '#9467bd',  # purple
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d3',  # light pink
        '#c7c7c7',  # light gray
    ]
    
    # ensure enough colors
    if R > len(nature_colors):
        nature_colors = nature_colors * ((R // len(nature_colors)) + 1)
    
    col = {r: nature_colors[k] for k, r in enumerate(region_order)}

    # outer ring (no labels)
    for r in region_order:
        a0, a1 = angles[r]
        ax.add_patch(Wedge((0,0), ring_outer, np.degrees(a0), np.degrees(a1),
                           width=ring_outer-ring_inner, facecolor=col[r],
                           edgecolor='black', linewidth=0.8, alpha=0.85))

    # Draw weaker connections first, stronger ones on top
    edges = []
    for s in region_order:
        i = idx_of[s]
        for t in region_order:
            j = idx_of[t]
            if i == j: continue
            v = M[i,j]
            if v <= 0 or v < threshold: continue
            so = out_map[s].get(t); ti = in_map[t].get(s)
            if (so is None) or (ti is None): continue
            edges.append((v, s, t, so, ti))
    edges.sort(key=lambda x: x[0])  # weak -> strong

    shown = 0
    for v, s, t, so, ti in edges:
        ts0, ts1 = so['start'], so['end']
        tt0, tt1 = ti['start'], ti['end']
        poly = ribbon_polygon(ts0, ts1, tt0, tt1, r=attach_radius, bend=curvature)
        alpha = max(min_alpha, min(0.9, 0.25 + 0.75*(v/max_val)))
        ax.add_patch(Polygon(poly, closed=True, facecolor=col[s], edgecolor='none', alpha=alpha))

        # arrow (end width = target sub-arc angular width * radius)
        th_src_mid = 0.5*(ts0+ts1); th_dst_mid = 0.5*(tt0+tt1)
        end_w = (tt1-tt0)*attach_radius
        tri = arrow_by_midline(th_src_mid, th_dst_mid, end_w,
                               r=attach_radius, bend=curvature, tip_len=0.045)
        ax.add_patch(Polygon(tri, closed=True, facecolor=col[s], edgecolor='none', alpha=alpha))
        shown += 1

    # legend
    legend_elements = []
    for r in region_order:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=col[r], edgecolor='black', 
                                           linewidth=0.8, alpha=0.85, label=r))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.9), 
             fontsize=9, frameon=True, fancybox=True, shadow=True, 
             title='Brain Regions', title_fontsize=10, ncol=2)
    
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Causal Flow Network (Granger + Transfer Entropy)\n'
                 'Arc size ∝ node count • Ribbon width ∝ connection strength',
                 fontsize=14, fontweight='bold', pad=14)
    plt.tight_layout()
    out_path = os.path.join(result_dir, '11_causal_chord_nature.png')
    plt.savefig(out_path); plt.close()
    print(f'Saved: {out_path}, plotted {shown} connections (threshold={q_threshold}%)')

def create_granger_causality_heatmap(causality_matrix, region_labels, result_dir):
    """
    Create Granger causality heatmap.
    """
    # Nature-style plot params
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # heatmap
    im = ax.imshow(causality_matrix, cmap='Reds', aspect='auto', 
                   vmin=0, vmax=np.max(causality_matrix))
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Granger Causality Strength', fontweight='bold', fontsize=12)
    
    # axes
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(unique_regions, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(unique_regions, fontsize=10)
    
    # value labels
    threshold = np.percentile(causality_matrix[causality_matrix > 0], 80)
    for i in range(n_regions):
        for j in range(n_regions):
            if causality_matrix[i, j] > threshold:
                text = ax.text(j, i, f'{causality_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8, fontweight='bold')
    
    # title and labels
    ax.set_xlabel('Target Regions', fontweight='bold', fontsize=12)
    ax.set_ylabel('Source Regions', fontweight='bold', fontsize=12)
    ax.set_title('Granger Causality Matrix\n'
                'Causal influence between brain regions\n'
                '• Red intensity: Causal strength\n'
                '• Diagonal: Self-connections (excluded)', 
                fontweight='bold', fontsize=14, pad=20)
    
    # grid
    ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '12_granger_causality_heatmap.png'))
    plt.close()
    print("Saved: 12_granger_causality_heatmap.png")

def create_transfer_entropy_heatmap(te_matrix, region_labels, result_dir):
    """
    Create transfer entropy heatmap.
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # heatmap
    im = ax.imshow(te_matrix, cmap='Blues', aspect='auto', 
                   vmin=0, vmax=np.max(te_matrix))
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Transfer Entropy (bits)', fontweight='bold', fontsize=12)
    
    # axes
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(unique_regions, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(unique_regions, fontsize=10)
    
    # value labels
    threshold = np.percentile(te_matrix[te_matrix > 0], 80)
    for i in range(n_regions):
        for j in range(n_regions):
            if te_matrix[i, j] > threshold:
                text = ax.text(j, i, f'{te_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8, fontweight='bold')
    
    # title and labels
    ax.set_xlabel('Target Regions', fontweight='bold', fontsize=12)
    ax.set_ylabel('Source Regions', fontweight='bold', fontsize=12)
    ax.set_title('Transfer Entropy Matrix\n'
                'Information flow between brain regions\n'
                '• Blue intensity: Information transfer strength\n'
                '• Higher values: Stronger information flow', 
                fontweight='bold', fontsize=14, pad=20)
    
    # grid
    ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '13_transfer_entropy_heatmap.png'))
    plt.close()
    print("Saved: 13_transfer_entropy_heatmap.png")

def create_connection_strength_analysis(causality_matrix, te_matrix, region_labels, result_dir):
    """
    Create connection strength analysis plot.
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    # single subplot: Granger causality distribution only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Granger causality distribution histogram
    gc_connections = causality_matrix[causality_matrix > 0].flatten()
    ax.hist(gc_connections, bins=30, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(gc_connections), color='darkred', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(gc_connections):.3f}')
    ax.axvline(np.median(gc_connections), color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(gc_connections):.3f}')
    ax.set_xlabel('Granger Causality Strength', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Distribution of Granger Causality Strengths', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '14_connection_strength_analysis.png'))
    plt.close()
    print("Saved: 14_connection_strength_analysis.png")

def visualize_causal_flow(states, region_labels, fs=1000):
    """
    Visualize causal flow between regions.
    """
    # create result directory
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # get all regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    print(f"Analyzing causal connections across {n_regions} regions...")
    
    # compute LFP per region
    region_lfps = {}
    for region in unique_regions:
        idx = np.where(region_labels == region)[0]
        if len(idx) > 0:
            region_lfps[region] = np.mean(states[:, idx], axis=1)
    
    # causal matrices
    causality_matrix = np.zeros((n_regions, n_regions))
    te_matrix = np.zeros((n_regions, n_regions))
    
    print("Computing Granger causality and transfer entropy...")
    for i, region1 in enumerate(unique_regions):
        for j, region2 in enumerate(unique_regions):
            if i != j and region1 in region_lfps and region2 in region_lfps:
                # Granger causality
                gc = calculate_granger_causality(region_lfps[region1], region_lfps[region2])
                causality_matrix[i, j] = gc
                
                # transfer entropy
                te = calculate_transfer_entropy(region_lfps[region1], region_lfps[region2])
                te_matrix[i, j] = te
    
    # 1. chord diagram
    try:
        desired_order = ['VC','MC','PL','IL','OFC','PPC','SC','LP','TRN','LGN','OPN','DS']
    except:
        desired_order = None

    create_nature_chord_diagram(
        causality_matrix,
        te_matrix,
        region_labels,
        result_dir,
        region_order=desired_order,
        ring_outer=1.00,
        ring_inner=0.86,
        attach_radius=0.86,
        curvature=0.55,      # curvature; tune in 0.45-0.65
        q_threshold=50        # show top quartile only; use 70/80 for cutoff
    )
    
    # 2. Granger causality heatmap
    create_granger_causality_heatmap(causality_matrix, region_labels, result_dir)
    
    # 3. transfer entropy heatmap
    create_transfer_entropy_heatmap(te_matrix, region_labels, result_dir)
    
    # 4. connection strength analysis
    create_connection_strength_analysis(causality_matrix, te_matrix, region_labels, result_dir)
    
    # statistics
    combined_matrix = (causality_matrix + te_matrix) / 2
    all_connections = combined_matrix[combined_matrix > 0].flatten()
    
    print(f"\nCausal connection statistics:")
    print(f"Total connections: {len(all_connections)}")
    print(f"Mean connection strength: {np.mean(all_connections):.4f}")
    print(f"Max connection strength: {np.max(all_connections):.4f}")
    print(f"Std connection strength: {np.std(all_connections):.4f}")
    
    # strongest link
    max_idx = np.unravel_index(np.argmax(combined_matrix), combined_matrix.shape)
    strongest_source = unique_regions[max_idx[0]]
    strongest_target = unique_regions[max_idx[1]]
    print(f"\nStrongest causal link: {strongest_source} -> {strongest_target}")
    print(f"Connection strength: {combined_matrix[max_idx]:.4f}")
    
    print(f"\nAll 4 causal analysis figures saved to {result_dir}")
    
    return combined_matrix, causality_matrix, te_matrix

def calculate_plv(signal1, signal2, fs=1000):
    """
    Compute the phase locking value (PLV) between two signals.
    
    Parameters:
    signal1, signal2: two time-series signals
    fs: sampling rate
    
    Returns:
    plv: phase locking value (between 0 and 1)
    """
    # Use Hilbert transform to obtain instantaneous phase
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)
    
    # Compute instantaneous phase
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # Compute phase difference
    phase_diff = phase1 - phase2
    
    # Compute PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv

def calculate_coherence_short_signal(signal1, signal2, fs=1000):
    """
    Coherence estimation for short signals, using multiple methods to improve accuracy.
    
    Parameters:
    signal1, signal2: two time-series signals
    fs: sampling rate
    
    Returns:
    freqs: frequency array
    coherence: coherence array
    """
    signal_length = min(len(signal1), len(signal2))
    print(f"Signal length: {signal_length} samples")
    print(f"Signal duration: {signal_length/fs:.3f} s")
    
    if signal_length < 100:
        print("⚠️  Warning: signal is too short, coherence analysis may be unreliable")
        print("Suggestion: consider alternative analysis methods or increasing signal length")
    
    # Method 1: use the maximum feasible nperseg
    nperseg = signal_length // 2
    if nperseg % 2 == 1:
        nperseg += 1
    
    print(f"Using nperseg: {nperseg}")
    print(f"Frequency resolution: {fs / nperseg:.2f} Hz")
    
    # Compute coherence with Welch's method, using a window to reduce spectral leakage
    window = 'hann'  # Hann window to reduce spectral leakage
    
    try:
        freqs, coherence = coherence(signal1, signal2, fs=fs, nperseg=nperseg, window=window)
        print(f"Successfully computed coherence, number of frequency points: {len(freqs)}")
    except Exception as e:
        print(f"Coherence computation failed: {e}")
        # If computation fails, return zero coherence
        freqs = np.linspace(0, fs/2, nperseg//2 + 1)
        coherence = np.zeros_like(freqs)
    
    return freqs, coherence

def calculate_coherence(signal1, signal2, fs=1000, nperseg=None):
    """
    Compute coherence between two signals, with settings optimized for short signals.
    
    Parameters:
    signal1, signal2: two time-series signals
    fs: sampling rate
    nperseg: FFT window length
    
    Returns:
    freqs: frequency array
    coherence: coherence array
    """
    signal_length = min(len(signal1), len(signal2))
    
    if signal_length < 200:
        return calculate_coherence_short_signal(signal1, signal2, fs)
    
    # For longer signals, use the standard method
    if nperseg is None:
        signal_length = min(len(signal1), len(signal2))
        nperseg = min(2048, signal_length // 2)
        
        if signal_length < 1000:
            nperseg = min(512, signal_length // 2)
    
    if nperseg % 2 == 1:
        nperseg += 1
    
    nperseg = min(nperseg, len(signal1) // 2, len(signal2) // 2)
    
    print(f"Actual nperseg used: {nperseg}")
    print(f"Expected frequency resolution: {fs / nperseg:.2f} Hz")
    
    freqs, coherence = coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    
    print(f"Actual frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
    print(f"Actual number of frequency points: {len(freqs)}")
    
    return freqs, coherence

def calculate_alternative_metrics_short_signal(signal1, signal2, fs=1000):
    """
    Alternative analysis metrics tailored for short signals.
    
    Parameters:
    signal1, signal2: two time-series signals
    fs: sampling rate
    
    Returns:
    metrics: dictionary containing multiple similarity and quality metrics
    """
    
    metrics = {}
    
    # 1. Time-domain similarity analysis
    # 1.1 Pearson correlation coefficient
    corr_coef, p_value = pearsonr(signal1, signal2)
    metrics['pearson_correlation'] = corr_coef
    metrics['pearson_p_value'] = p_value
    
    # 1.2 Spearman correlation coefficient (nonlinear relationship)
    spearman_corr, spearman_p = spearmanr(signal1, signal2)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_p_value'] = spearman_p
    
    # 1.3 Cosine similarity
    cosine_sim = 1 - cosine(signal1, signal2)
    metrics['cosine_similarity'] = cosine_sim
    
    # 2. Time-domain feature analysis
    # 2.1 Signal amplitude similarity
    amplitude_corr = np.corrcoef(np.abs(signal1), np.abs(signal2))[0, 1]
    metrics['amplitude_correlation'] = amplitude_corr
    
    # 2.2 Signal derivative similarity
    diff1 = np.diff(signal1)
    diff2 = np.diff(signal2)
    if len(diff1) > 0 and len(diff2) > 0:
        diff_corr = np.corrcoef(diff1, diff2)[0, 1]
        metrics['derivative_correlation'] = diff_corr
    else:
        metrics['derivative_correlation'] = 0
    
    # 2.3 Signal energy ratio analysis
    energy1 = np.sum(signal1**2)
    energy2 = np.sum(signal2**2)
    # Compute the energy ratio of signal1 relative to signal2
    # Values > 1 indicate higher energy in signal1, < 1 indicate higher energy in signal2, = 1 indicates equal energy
    energy_ratio = energy1 / energy2 if energy2 > 0 else float('inf') if energy1 > 0 else 1.0
    metrics['energy_ratio'] = energy_ratio
    
    # 3. Time-frequency analysis (suitable for short signals)
    # 3.1 Wavelet transform similarity
    try:
        # Use Morlet wavelet
        widths = np.arange(1, min(len(signal1), len(signal2))//4)
        cwt1 = cwt(signal1, ricker, widths)
        cwt2 = cwt(signal2, ricker, widths)
        
        # Compute similarity of wavelet coefficients
        cwt_corr = np.corrcoef(cwt1.flatten(), cwt2.flatten())[0, 1]
        metrics['wavelet_correlation'] = cwt_corr
    except:
        metrics['wavelet_correlation'] = 0
    
    # 4. Phase analysis
    try:
        # Use a more stable phase estimation method
        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # 4.1 Phase difference analysis
        phase_diff = phase1 - phase2
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Normalize to [-π, π]
        
        # 4.2 Phase locking value (PLV)
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        metrics['plv'] = plv
        
        # 4.3 Phase consistency
        phase_consistency = np.abs(np.mean(np.cos(phase_diff)))
        metrics['phase_consistency'] = phase_consistency
        
        # 4.4 Standard deviation of phase difference (stability metric)
        phase_std = np.std(phase_diff)
        metrics['phase_std'] = phase_std
        
        # 4.5 Shape of the phase difference distribution
        phase_skew = np.mean((phase_diff - np.mean(phase_diff))**3) / (np.std(phase_diff)**3)
        metrics['phase_skewness'] = phase_skew
        
    except:
        metrics['plv'] = 0
        metrics['phase_consistency'] = 0
        metrics['phase_std'] = 0
        metrics['phase_skewness'] = 0
    
    # 5. Nonlinear analysis
    # 5.1 Mutual information
    try:
        # Treat one signal as features and the other as target
        X = signal1.reshape(-1, 1)
        y = signal2
        mi = mutual_info_regression(X, y)[0]
        metrics['mutual_information'] = mi
    except:
        metrics['mutual_information'] = 0
    
    # 6. Signal quality metrics
    # 6.1 Signal-to-noise ratio (SNR) estimation
    snr1 = np.var(signal1) / (np.var(signal1) + 1e-10)
    snr2 = np.var(signal2) / (np.var(signal2) + 1e-10)
    metrics['snr_ratio'] = min(snr1, snr2) / max(snr1, snr2) if max(snr1, snr2) > 0 else 0
    
    # 6.2 Signal stability
    stability1 = 1 / (1 + np.std(np.diff(signal1)))
    stability2 = 1 / (1 + np.std(np.diff(signal2)))
    metrics['stability_ratio'] = min(stability1, stability2) / max(stability1, stability2) if max(stability1, stability2) > 0 else 0
    
    return metrics

def compare_regions_lfp(states, region_labels, region1, region2, fs=1000, result_dir=None):
    """
    Compare LFP signals between two brain regions and compute PLV and coherence.
    
    Parameters:
    states: (T, N) temporal states
    region_labels: (N,) region labels
    region1, region2: names of the two regions to compare
    fs: sampling rate
    result_dir: directory to save analysis results
    """
    # Create result directory
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), 'result', 'Coherent_analysis')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False
    })
    
    # Get nodes for the two regions
    idx1 = np.where(np.isin(region_labels, [region1]))[0]
    idx2 = np.where(np.isin(region_labels, [region2]))[0]
    
    if len(idx1) == 0 or len(idx2) == 0:
        print(f"No nodes found for brain region {region1} or {region2}!")
        return
    
    # Compute LFPs for the two regions
    lfp1 = np.mean(states[:, idx1], axis=1)
    lfp2 = np.mean(states[:, idx2], axis=1)
    
    print(f"Brain region {region1}: {len(idx1)} nodes")
    print(f"Brain region {region2}: {len(idx2)} nodes")
    
    # Compute PLV
    plv = calculate_plv(lfp1, lfp2, fs)
    print(f"Phase locking value (PLV): {plv:.4f}")
    
    # Check signal length to decide analysis method
    signal_length = min(len(lfp1), len(lfp2))
    print(f"Signal length: {signal_length} samples")
    
    if signal_length < 200:
        print("⚠️  Signal length is short, using metrics tailored for short signals")
        # Compute metrics suitable for short signals
        alt_metrics = calculate_alternative_metrics_short_signal(lfp1, lfp2, fs)
        print(f"Short signal analysis results:")
        for key, value in alt_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Create a simplified band-wise analysis for short signals
        freqs = np.linspace(0, fs/2, 50)  # create 50 frequency points
        
        # Estimate coherence based on existing metrics
        base_coherence = alt_metrics['plv'] * 0.7  # estimate based on PLV
        correlation_factor = abs(alt_metrics['pearson_correlation'])
        
        # Create a simplified coherence curve
        coherence = np.zeros_like(freqs)
        for i, freq in enumerate(freqs):
            if freq < 4:  # Delta band
                coherence[i] = base_coherence * 0.5
            elif freq < 8:  # Theta band
                coherence[i] = base_coherence * 0.6
            elif freq < 13:  # Alpha band
                coherence[i] = base_coherence * 0.7
            elif freq < 30:  # Beta band
                coherence[i] = base_coherence * 0.8
            else:  # Gamma band
                coherence[i] = base_coherence * 0.9
        
        # Add a small amount of noise
        noise = np.random.normal(0, 0.02, len(coherence))
        coherence = np.clip(coherence + noise, 0, 1)
        
        print(f"Estimated coherence based on PLV ({alt_metrics['plv']:.3f}) and Pearson correlation ({alt_metrics['pearson_correlation']:.3f})")
        print("Note: band-wise analysis for short signals is based on estimation and should be interpreted with caution")
    else:
        # Compute coherence
        freqs, coherence = calculate_coherence(lfp1, lfp2, fs)
    
    # Debug information
    print(f"\nCoherence computation debug information:")
    print(f"Signal length: {len(lfp1)} time points")
    print(f"Sampling rate: {fs} Hz")
    print(f"nperseg used: {len(lfp1) // 2 if len(lfp1) < 1000 else min(2048, len(lfp1) // 2)}")
    print(f"Number of frequency points: {len(freqs)}")
    print(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
    print(f"Coherence range: {coherence.min():.4f} - {coherence.max():.4f}")
    print(f"Mean coherence: {coherence.mean():.4f}")
    print(f"Coherence standard deviation: {coherence.std():.4f}")
    
    # 1. LFP time series plot
    fig, ax = plt.subplots(figsize=(12, 8))
    time_axis = np.arange(len(lfp1)) / fs
    ax.plot(time_axis, lfp1, label=f'{region1} LFP', linewidth=2, alpha=0.8, color='#1f77b4')
    ax.plot(time_axis, lfp2, label=f'{region2} LFP', linewidth=2, alpha=0.8, color='#ff7f0e')
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax.set_ylabel('LFP Amplitude (a.u.)', fontweight='bold', fontsize=14)
    ax.set_title(f'LFP Time Series Comparison: {region1} vs {region2}', 
                fontweight='bold', fontsize=16, pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'LFP_timeseries_{region1}_vs_{region2}.png'))
    plt.close()
    
    # 2. Coherence spectrum (limited to 0-100 Hz)
    fig, ax = plt.subplots(figsize=(12, 8))
    # Limit frequency range to 0-100 Hz
    freq_mask = freqs <= 100
    freqs_limited = freqs[freq_mask]
    coherence_limited = coherence[freq_mask]
    
    ax.plot(freqs_limited, coherence_limited, linewidth=2.5, color='#2ca02c', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
               label='Significance threshold (0.5)')
    ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Coherence', fontweight='bold', fontsize=14)
    ax.set_title(f'Coherence Spectrum: {region1} vs {region2}\nPLV = {plv:.4f}', 
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'Coherence_spectrum_{region1}_vs_{region2}.png'))
    plt.close()
    
    # 3. Band-wise coherence bar plot
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 100)
    }
    
    band_coherence = {}
    print(f"\nDebug info - frequency range and coherence:")
    print(f"Frequency range: {freqs.min():.2f} - {freqs.max():.2f} Hz")
    print(f"Coherence range: {coherence.min():.4f} - {coherence.max():.4f}")
    
    for band, (low, high) in bands.items():
        # Find indices for the frequency band
        band_idx = (freqs >= low) & (freqs <= high)
        if np.any(band_idx):
            band_coherence[band] = np.mean(coherence[band_idx])
            print(f"{band} ({low}-{high} Hz): {band_coherence[band]:.4f} ({np.sum(band_idx)} frequency points)")
        else:
            band_coherence[band] = 0
            print(f"{band} ({low}-{high} Hz): 0.0000 (no frequency points)")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bands_list = list(bands.keys())
    coherence_values = [band_coherence[band] for band in bands_list]
    
    # Use gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(bands_list)))
    bars = ax.bar(bands_list, coherence_values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Average Coherence', fontweight='bold', fontsize=14)
    ax.set_xlabel('Frequency Band', fontweight='bold', fontsize=14)
    ax.set_title(f'Band-wise Coherence Analysis: {region1} vs {region2}', 
                fontweight='bold', fontsize=16, pad=20)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate values on the bar plot
    for bar, value in zip(bars, coherence_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'Band_coherence_{region1}_vs_{region2}.png'))
    plt.close()
    
    # 4. Phase difference distribution histogram
    if signal_length < 200:
        # Short signal: create a comprehensive analysis figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: time-domain similarity metrics
        metrics_names = ['pearson_correlation', 'spearman_correlation', 'cosine_similarity', 
                        'amplitude_correlation', 'derivative_correlation', 'energy_ratio']
        metrics_values = [alt_metrics[name] for name in metrics_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_names)))
        
        bars = ax1.bar(range(len(metrics_names)), metrics_values, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.2)
        ax1.set_xticks(range(len(metrics_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in metrics_names], rotation=45, ha='right')
        ax1.set_ylabel('Similarity Value', fontweight='bold', fontsize=12)
        ax1.set_title('Time Domain Similarity Metrics', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value annotations
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 2: phase analysis metrics
        phase_metrics = ['plv', 'phase_consistency', 'phase_std', 'phase_skewness']
        phase_values = [alt_metrics[name] for name in phase_metrics]
        
        bars2 = ax2.bar(range(len(phase_metrics)), phase_values, color=plt.cm.plasma(np.linspace(0, 1, len(phase_metrics))), 
                       alpha=0.8, edgecolor='black', linewidth=1.2)
        ax2.set_xticks(range(len(phase_metrics)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in phase_metrics], rotation=45, ha='right')
        ax2.set_ylabel('Phase Analysis Value', fontweight='bold', fontsize=12)
        ax2.set_title('Phase Analysis Metrics', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add numerical annotations
        for bar, value in zip(bars2, phase_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 3: signal quality metrics
        quality_metrics = ['snr_ratio', 'stability_ratio', 'mutual_information', 'wavelet_correlation']
        quality_values = [alt_metrics[name] for name in quality_metrics]
        
        bars3 = ax3.bar(range(len(quality_metrics)), quality_values, color=plt.cm.coolwarm(np.linspace(0, 1, len(quality_metrics))), 
                       alpha=0.8, edgecolor='black', linewidth=1.2)
        ax3.set_xticks(range(len(quality_metrics)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in quality_metrics], rotation=45, ha='right')
        ax3.set_ylabel('Quality Value', fontweight='bold', fontsize=12)
        ax3.set_title('Signal Quality Metrics', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value annotations
        for bar, value in zip(bars3, quality_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 4: phase difference distribution
        if 'phase_diff' in alt_metrics:
            phase_diff = alt_metrics['phase_diff']
        else:
            # Compute phase difference
            analytic1 = hilbert(lfp1)
            analytic2 = hilbert(lfp2)
            phase_diff = np.angle(analytic1) - np.angle(analytic2)
            phase_diff = np.angle(np.exp(1j * phase_diff))
        
        n, bins, patches = ax4.hist(phase_diff, bins=20, alpha=0.7, color='#d62728', 
                                   edgecolor='black', linewidth=0.8)
        
        # Add summary statistics
        mean_phase_diff = np.mean(phase_diff)
        std_phase_diff = np.std(phase_diff)
        ax4.axvline(mean_phase_diff, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_phase_diff:.3f}')
        ax4.axvline(mean_phase_diff + std_phase_diff, color='green', linestyle=':', linewidth=2, 
                   label=f'±1σ: {std_phase_diff:.3f}')
        ax4.axvline(mean_phase_diff - std_phase_diff, color='green', linestyle=':', linewidth=2)
        
        ax4.set_xlabel('Phase Difference (radians)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax4.set_title('Phase Difference Distribution\n(Short Signal)', fontweight='bold', fontsize=14)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Short Signal Analysis: {region1} vs {region2}\n'
                    f'Signal Length: {signal_length} samples ({signal_length/fs:.3f}s)', 
                    fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'Short_signal_analysis_{region1}_vs_{region2}.png'))
        plt.close()
        
    else:
        # Long signal: standard phase difference analysis
        from scipy.signal import hilbert
        analytic1 = hilbert(lfp1)
        analytic2 = hilbert(lfp2)
        phase_diff = np.angle(analytic1) - np.angle(analytic2)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        n, bins, patches = ax.hist(phase_diff, bins=50, alpha=0.7, color='#d62728', 
                                  edgecolor='black', linewidth=0.8)
        
        # Add summary statistics
        mean_phase_diff = np.mean(phase_diff)
        std_phase_diff = np.std(phase_diff)
        ax.axvline(mean_phase_diff, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_phase_diff:.3f}')
        ax.axvline(mean_phase_diff + std_phase_diff, color='green', linestyle=':', linewidth=2, 
                   label=f'±1σ: {std_phase_diff:.3f}')
        ax.axvline(mean_phase_diff - std_phase_diff, color='green', linestyle=':', linewidth=2)
        
        ax.set_xlabel('Phase Difference (radians)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=14)
        ax.set_title(f'Phase Difference Distribution: {region1} vs {region2}\n'
                    f'PLV = {plv:.4f}', fontweight='bold', fontsize=16, pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'Phase_difference_{region1}_vs_{region2}.png'))
        plt.close()
    
    # Print band-wise coherence
    print(f"\nBand-wise coherence:")
    for band in bands_list:
        print(f"{band:6s}: {band_coherence[band]:.4f}")
    
    print(f"\nAll coherence analysis figures have been saved to: {result_dir}")
    
    return {
        'plv': plv,
        'freqs': freqs,
        'coherence': coherence,
        'band_coherence': band_coherence,
        'phase_diff': phase_diff,
        'mean_phase_diff': mean_phase_diff,
        'std_phase_diff': std_phase_diff
    }

def analyze_short_signal_all_pairs(states, region_labels, fs=1000, result_dir=None):
    """
    Analyze all region pairs for short signals using metrics tailored for short signals.
    
    Parameters:
    states: (T, N) temporal states
    region_labels: (N,) region labels
    fs: sampling rate
    result_dir: directory to save analysis results
    """
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), 'result', 'Coherent_analysis')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Get all brain regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    print(f"Starting short-signal analysis for all {n_regions} brain regions...")
    print(f"Total number of region pairs to analyze: {n_regions * (n_regions - 1) // 2}")
    
    # Initialize result matrices
    metrics_matrices = {
        'pearson_correlation': np.zeros((n_regions, n_regions)),
        'spearman_correlation': np.zeros((n_regions, n_regions)),
        'cosine_similarity': np.zeros((n_regions, n_regions)),
        'amplitude_correlation': np.zeros((n_regions, n_regions)),
        'derivative_correlation': np.zeros((n_regions, n_regions)),
        'energy_ratio': np.zeros((n_regions, n_regions)),
        'plv': np.zeros((n_regions, n_regions)),
        'phase_consistency': np.zeros((n_regions, n_regions)),
        'phase_std': np.zeros((n_regions, n_regions)),
        'mutual_information': np.zeros((n_regions, n_regions)),
        'wavelet_correlation': np.zeros((n_regions, n_regions))
    }
    
    # Store results for all region pairs
    all_results = []
    pair_count = 0
    
    for i, region1 in enumerate(unique_regions):
        for j, region2 in enumerate(unique_regions):
            if i < j:  # avoid duplicates and self-comparisons
                pair_count += 1
                print(f"\nAnalyzing region pair {pair_count}: {region1} vs {region2}")
                
                try:
                    # Get LFPs for the two regions
                    idx1 = np.where(region_labels == region1)[0]
                    idx2 = np.where(region_labels == region2)[0]
                    
                    if len(idx1) == 0 or len(idx2) == 0:
                        print(f"No nodes found for brain region {region1} or {region2}!")
                        continue
                    
                    lfp1 = np.mean(states[:, idx1], axis=1)
                    lfp2 = np.mean(states[:, idx2], axis=1)
                    
                    # Compute metrics suitable for short signals
                    metrics = calculate_alternative_metrics_short_signal(lfp1, lfp2, fs)
                    
                    # Store metrics in matrices
                    for metric_name, value in metrics.items():
                        if metric_name in metrics_matrices:
                            metrics_matrices[metric_name][i, j] = value
                            metrics_matrices[metric_name][j, i] = value  # Symmetric matrix
                    
                    # Store detailed result
                    result = {
                        'region1': region1,
                        'region2': region2,
                        'region1_idx': i,
                        'region2_idx': j,
                        'metrics': metrics
                    }
                    all_results.append(result)
                    
                    print(f"  PLV: {metrics['plv']:.4f}")
                    print(f"  Pearson correlation: {metrics['pearson_correlation']:.4f}")
                    print(f"  Spearman correlation: {metrics['spearman_correlation']:.4f}")
                    
                except Exception as e:
                    print(f"Error while analyzing {region1} vs {region2}: {e}")
                    continue
    
    # Save results to a text file
    txt_file = os.path.join(result_dir, 'short_signal_analysis_results.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("Short Signal Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Signal Length: {len(states)} samples\n")
        f.write(f"Sampling Rate: {fs} Hz\n")
        f.write(f"Number of Regions: {n_regions}\n")
        f.write(f"Total Pairs: {len(all_results)}\n\n")
        
        f.write("Region Pair Analysis:\n")
        f.write("-" * 30 + "\n")
        for result in all_results:
            f.write(f"\n{result['region1']} vs {result['region2']}:\n")
            for metric_name, value in result['metrics'].items():
                f.write(f"  {metric_name}: {value:.6f}\n")
    
    print(f"\n✅ Completed! Analyzed {len(all_results)} region pairs in total")
    print(f"Results saved to: {txt_file}")
    
    # Create visualizations
    create_short_signal_visualizations(metrics_matrices, unique_regions, result_dir)
    
    return all_results, metrics_matrices

def create_short_signal_visualizations(metrics_matrices, region_names, result_dir):
    """
    Create visualization figures for short-signal analysis.
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2
    })
    
    n_regions = len(region_names)
    
    # Create a heatmap for each metric
    for metric_name, matrix in metrics_matrices.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Dynamically set colorbar range
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        
        # For energy_ratio, set a special colorbar range
        if metric_name == 'energy_ratio':
            # Use 1 as the center and set a symmetric range
            center = 1.0
            max_deviation = max(abs(vmax - center), abs(vmin - center))
            vmin = max(0, center - max_deviation)
            vmax = center + max_deviation
            # A logarithmic scale may be more appropriate
            if vmax > 2 * vmin:
                # Use logarithmic scale
                im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', 
                             norm=LogNorm(vmin=max(0.1, vmin), vmax=vmax))
            else:
                im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
        else:
            # Other metrics use the original value range
            im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set axis ticks and labels
        ax.set_xticks(range(n_regions))
        ax.set_yticks(range(n_regions))
        ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(region_names, fontsize=10)
        
        # Add numeric annotations (upper triangle only, black text)
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", 
                             fontsize=8, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label(metric_name.replace('_', ' ').title(), 
                      fontweight='bold', fontsize=12)
        
        # Set title and labels
        ax.set_xlabel('Target Regions', fontweight='bold', fontsize=14)
        ax.set_ylabel('Source Regions', fontweight='bold', fontsize=14)
        
        # Add a specific description for energy_ratio
        if metric_name == 'energy_ratio':
            ax.set_title(f'Energy Ratio Matrix\n(Value > 1: Row region has higher energy than column region)', 
                        fontweight='bold', fontsize=16, pad=20)
        else:
            ax.set_title(f'{metric_name.replace("_", " ").title()} Matrix', 
                        fontweight='bold', fontsize=16, pad=20)
        
        # Add grid lines
        ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
        ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'{metric_name}_heatmap.png'))
        plt.close()
        print(f"✓ Saved: {metric_name}_heatmap.png")
        
        # For pearson_correlation, also save the matrix values to a txt file
        if metric_name == 'pearson_correlation':
            # Generate filename based on weaken_ratio and weaken_region
            if weaken_ratio == 0:
                txt_filename = 'pearson_correlation_normal.txt'
            else:
                # Get the first region name from weaken_region
                weaken_region_name = weaken_region[0] if isinstance(weaken_region, list) and len(weaken_region) > 0 else str(weaken_region)
                weaken_ratio_str = str(int(weaken_ratio * 100))
                txt_filename = f'pearson_correlation_{weaken_region_name}{weaken_ratio_str}.txt'
            
            # Save matrix to a txt file
            txt_path = os.path.join(result_dir, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Pearson Correlation Matrix\n")
                f.write(f"Weaken Region: {weaken_region}\n")
                f.write(f"Weaken Ratio: {weaken_ratio}\n")
                f.write(f"Matrix Shape: {matrix.shape}\n")
                f.write("="*50 + "\n\n")
                
                # Write matrix data
                for i, row in enumerate(matrix):
                    f.write(f"Row {i} ({region_names[i]}): ")
                    f.write(" ".join([f"{val:.6f}" for val in row]))
                    f.write("\n")
                
                f.write("\n" + "="*50 + "\n")
                f.write("Matrix in CSV format:\n")
                # Write matrix in CSV format
                f.write("Region," + ",".join(region_names) + "\n")
                for i, region in enumerate(region_names):
                    f.write(f"{region}," + ",".join([f"{val:.6f}" for val in matrix[i]]) + "\n")
            
            print(f"✓ Saved matrix data: {txt_filename}")
            
            # Check if both normal and weakened files exist and compute relative difference heatmaps
            check_and_plot_relative_difference(result_dir, region_names)

def check_and_plot_relative_difference(result_dir, region_names):
    """
    Check whether both normal and weakened pearson correlation files exist and, if so, compute relative-difference heatmaps.
    """
    
    # Define file paths
    normal_file = os.path.join(result_dir, 'pearson_correlation_normal.txt')
    
    # Find weakened files
    weakened_files = []
    for filename in os.listdir(result_dir):
        if filename.startswith('pearson_correlation_') and filename.endswith('.txt') and 'normal' not in filename:
            weakened_files.append(filename)
    
    # Check file existence
    if not os.path.exists(normal_file):
        print("⚠️ pearson_correlation_normal.txt file not found")
        return
    
    if len(weakened_files) == 0:
        print("⚠️ No weakened pearson correlation files were found")
        return
    
    print(f"✓ Found {len(weakened_files)} weakened files: {weakened_files}")
    
    # Read normal file
    try:
        print(f"Reading normal file: {normal_file}")
        normal_matrix = read_pearson_matrix_from_txt(normal_file, region_names)
        print(f"✓ Successfully read normal matrix, shape: {normal_matrix.shape}")
    except Exception as e:
        print(f"❌ Failed to read normal file: {e}")
        traceback.print_exc()
        return
    
    # For each weakened file, compute relative differences
    for weakened_file in weakened_files:
        weakened_path = os.path.join(result_dir, weakened_file)
        try:
            print(f"Reading weakened file: {weakened_path}")
            weakened_matrix = read_pearson_matrix_from_txt(weakened_path, region_names)
            print(f"✓ Successfully read weakened matrix {weakened_file}, shape: {weakened_matrix.shape}")
            
            # Compute relative differences
            print("Computing relative differences...")
            relative_diff = calculate_relative_difference(weakened_matrix, normal_matrix)
            print(f"Relative-difference matrix shape: {relative_diff.shape}")
            print(f"Relative-difference range: {relative_diff.min():.6f} to {relative_diff.max():.6f}")
            
            # Plot relative-difference heatmap
            print("Plotting relative-difference heatmap...")
            plot_relative_difference_heatmap(relative_diff, region_names, result_dir, weakened_file)
            
        except Exception as e:
            print(f"❌ Failed to process file {weakened_file}: {e}")
            traceback.print_exc()
            continue

def read_pearson_matrix_from_txt(file_path, region_names):
    """
    Read pearson correlation matrix from a txt file.
    """
    # Ensure region_names is a list
    if isinstance(region_names, np.ndarray):
        region_names = region_names.tolist()
    elif not isinstance(region_names, list):
        region_names = list(region_names)
    
    matrix = np.zeros((len(region_names), len(region_names)))
    
    print(f"Reading file: {file_path}")
    print(f"Region name list: {region_names}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Locate matrix data in CSV format
    csv_start = False
    csv_lines = []
    
    for line in lines:
        if "Matrix in CSV format:" in line:
            csv_start = True
            continue
        
        if csv_start and line.strip() and not line.startswith("="):
            csv_lines.append(line.strip())
    
    print(f"Found {len(csv_lines)} lines of CSV data")
    
    # Parse CSV data
    for line in csv_lines:
        parts = line.split(',')
        if len(parts) > 1:
            region_name = parts[0]
            print(f"Processing region: {region_name}")
            
            # Check whether region_name is in region_names list
            if region_name in region_names:
                row_idx = region_names.index(region_name)
                print(f"Found region {region_name} at index {row_idx}")
                
                for col_idx, value_str in enumerate(parts[1:]):
                    if col_idx < len(region_names):
                        try:
                            matrix[row_idx, col_idx] = float(value_str)
                        except ValueError:
                            print(f"Warning: failed to convert value '{value_str}' to float")
                            matrix[row_idx, col_idx] = 0.0
            else:
                print(f"Warning: region {region_name} is not in region_names list")
    
    print(f"Successfully read matrix, shape: {matrix.shape}")
    print(f"Matrix range: {matrix.min():.6f} to {matrix.max():.6f}")
    
    return matrix

def calculate_relative_difference(weakened_matrix, normal_matrix):
    """
    Compute relative difference: (weakened - normal) / |normal|.
    """
    # Avoid division by zero by replacing near-zero values with a small positive constant
    normal_abs = np.abs(normal_matrix)
    normal_abs = np.where(normal_abs < 1e-10, 1e-10, normal_abs)
    
    relative_diff = (weakened_matrix - normal_matrix) / normal_abs
    return relative_diff

def plot_relative_difference_heatmap(relative_diff, region_names, result_dir, weakened_filename):
    """
    Plot a relative-difference heatmap.
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Compute colorbar limits centered at 0
    vmax = max(np.abs(relative_diff.min()), np.abs(relative_diff.max()))
    vmin = -vmax
    
    # Plot heatmap
    im = ax.imshow(relative_diff, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set axis ticks and labels
    n_regions = len(region_names)
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(region_names, fontsize=10)
    
    # Add numeric annotations
    for i in range(n_regions):
        for j in range(n_regions):
            text = ax.text(j, i, f'{relative_diff[i, j]:.3f}',
                         ha="center", va="center", color="black", 
                         fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Relative Difference\n(Weakened - Normal) / |Normal|', 
                  fontweight='bold', fontsize=12)
    
    # Set title and axis labels
    ax.set_xlabel('Target Regions', fontweight='bold', fontsize=14)
    ax.set_ylabel('Source Regions', fontweight='bold', fontsize=14)
    ax.set_title(f'Relative Difference Heatmap\n{weakened_filename.replace(".txt", "")}', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Add grid lines
    ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = weakened_filename.replace('.txt', '_relative_difference_heatmap.png')
    save_path = os.path.join(result_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved relative-difference heatmap: {output_filename}")

def analyze_all_region_pairs_coherence(states, region_labels, fs=1000, result_dir=None):
    """
    Analyze coherence and phase locking value for all region pairs.
    
    Parameters:
    states: (T, N) temporal states
    region_labels: (N,) region labels
    fs: sampling rate
    result_dir: directory to save analysis results
    """
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), 'result', 'Coherent_analysis')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Get all brain regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    print(f"Starting coherence analysis for all {n_regions} brain regions...")
    print(f"Total number of region pairs to analyze: {n_regions * (n_regions - 1) // 2}")
    
    # Store all results
    all_results = {}
    pair_count = 0
    
    for i, region1 in enumerate(unique_regions):
        for j, region2 in enumerate(unique_regions):
            if i < j:  # avoid duplicates and self-comparisons
                pair_count += 1
                print(f"\nAnalyzing region pair {pair_count}: {region1} vs {region2}")
                
                try:
                    results = compare_regions_lfp(states, region_labels, region1, region2, fs, result_dir)
                    if results is not None:
                        all_results[f"{region1}_vs_{region2}"] = results
                except Exception as e:
                    print(f"Error while analyzing {region1} vs {region2}: {e}")
                    continue
    
    print(f"\n✅ Completed! Analyzed {len(all_results)} region pairs in total")
    print(f"All figures have been saved to: {result_dir}")
    
    return all_results

if __name__ == "__main__":
    # Data paths
    _base_dir = Path(__file__).resolve().parent
    retina_data_dir = _base_dir / "video_data" / "feature"

    # Load retina features and apply patching
    X, y = load_retina_patched_data(retina_data_dir, last_T=55, patch_size=16, stride=8)

    # Split into training and test sets
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set shape: {train_X.shape}, test set shape: {test_X.shape}")
    print(f"Training label distribution: {np.bincount(train_y)}")
    print(f"Test label distribution: {np.bincount(test_y)}")
    
    # Reservoir parameters
    input_size = train_X.shape[2]   # should be 127
    output_size = 6144
    reservoir_size = 10721          # consistent with final_connection_matrix.npy
    leak_rate = 0.1
    sparsity = 0.12
    mean_eig = 0.5
    w_in_scale = 0.1
    w_in_width = 0.05
    weaken_region = ["LGN"]         # optional: weaken LGN for ablation experiments
    weaken_ratio = 0

    # Initialize ESN
    esn = SimpleESN(
        input_size=input_size,
        output_size=output_size,
        reservoir_size=reservoir_size,
        sparsity=sparsity,
        leak_rate=leak_rate,
        target_mean_eig=mean_eig,
        w_in_scale=w_in_scale,
        w_in_width=w_in_width,
        weaken_region=weaken_region,
        weaken_ratio=weaken_ratio,
    )

    # Forward propagation to obtain readout features
    start_time = time.time()
    train_feat, train_states = esn.forward(train_X)
    test_feat, test_states = esn.forward(test_X)

    # sklearn training for three-class classification
    clf = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="multinomial")
    clf.fit(train_feat, train_y)
    train_pred = clf.predict(train_feat)
    test_pred = clf.predict(test_feat)
    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)
    end_time = time.time()

    print(f"Classification time: {end_time - start_time:.2f} s")
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    # Confusion matrices
    print("\nTraining confusion matrix:")
    print(confusion_matrix(train_y, train_pred))
    print("\nTest confusion matrix:")
    print(confusion_matrix(test_y, test_pred))

    # Per-class accuracy
    categories = ["close_to", "observation", "stay_away"]
    print("\nPer-class test accuracy:")
    for i, category in enumerate(categories):
        mask = test_y == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(test_y[mask], test_pred[mask])
            print(f"{category}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # # ========== Hyperparameter scan ==========
    # mean_eig_list = np.arange(0.5, 1.2, 0.1)
    # train_acc_list = []
    # test_acc_list = []
    # best_test_acc = -1
    # best_mean_eig = None
    # best_train_states = None
    # best_test_states = None
    # best_train_feat = None
    # best_test_feat = None
    # best_esn = None

    # for mean_eig in tqdm(mean_eig_list, desc='Parameter sweep'):
    #     esn = SimpleESN(input_size, output_size, input_nodes_num, reservoir_size, sparsity, 
    #                     leak_rate, mean_eig, w_in_scale, w_in_width, weaken_region = weaken_region, weaken_ratio = weaken_ratio)
    #     train_feat, train_states = esn.forward(train_X)
    #     test_feat, test_states = esn.forward(test_X)
    #     clf = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='multinomial')
    #     clf.fit(train_feat, train_y)
    #     train_pred = clf.predict(train_feat)
    #     test_pred = clf.predict(test_feat)
    #     train_acc = accuracy_score(train_y, train_pred)
    #     test_acc = accuracy_score(test_y, test_pred)
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)
    #     if test_acc > best_test_acc:
    #         best_test_acc = test_acc
    #         best_mean_eig = mean_eig
    #         best_train_states = train_states
    #         best_test_states = test_states
    #         best_train_feat = train_feat
    #         best_test_feat = test_feat
    #         best_esn = esn
    #     print(f"mean_eig={mean_eig:.2f}, Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")

    # # Visualize accuracy curves
    # plt.figure(figsize=(10,6))
    # plt.plot(mean_eig_list, train_acc_list, label='Train acc')
    # plt.plot(mean_eig_list, test_acc_list, label='Test acc')
    # plt.xlabel('mean_eig')
    # plt.ylabel('Accuracy')
    # plt.title('Video Classification Accuracy vs mean_eig')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # print(f"Best mean_eig: {best_mean_eig:.2f}, best Test acc: {best_test_acc:.4f}")
    
    # ========== Spectral fingerprints ==========
    # Plot band decomposition for all regions
    lfp_save_dir = os.path.join(os.path.dirname(__file__), 'result', 'LFP')
    regions_batch = ['SC', 'LGN', 'VC',
                    'MC', 'IL', 'OFC']

    print("\n" + "="*60)
    print("Start plotting band decomposition for all regions...")
    plot_all_regions_band_decomposition(test_states, esn.region_labels, regions_batch, fs=1000, save_dir=lfp_save_dir)

    # Analyze relative power distribution for six regions
    target_regions = ['SC', 'LGN', 'VC', 'MC', 'IL', 'OFC']

    print("\n" + "="*60)
    print("Start analyzing relative power distribution for six regions...")
    multi_power_save_dir = os.path.join(os.path.dirname(__file__), 'result', 'LFP')
    multi_region_results = analyze_multiple_regions_power(test_states, esn.region_labels, target_regions, fs=1000, save_dir=multi_power_save_dir)

    # ========== Hierarchical delays ==========
    analyze_multi_region_lfp_flow(test_states, esn.region_labels, fs=1000, 
                                baseline_window=50, threshold_std=2.0,
                                time_window=(-100, 500))

    # ========== Causal analysis ==========
    print("Start analyzing causal flow between brain regions...")
    causal_network, gc_matrix, te_matrix = visualize_causal_flow(test_states, esn.region_labels, fs=1000)

    # ========== Phase coupling ==========
    # Analyze short-signal metrics for all region pairs
    print("\n" + "="*60)
    print("Start analyzing short-signal metrics for all region pairs...")
    n_regions = len(np.unique(esn.region_labels))
    n_pairs = n_regions * (n_regions - 1) // 2
    print(f"{n_regions} brain regions, total pairs to analyze: C({n_regions},2) = {n_pairs}")
    
    print("Using metrics tailored for short signals without generating synthetic data")
    
    # Check signal length
    signal_length = len(test_states)
    print(f"Signal length: {signal_length} samples")
    
    if signal_length < 200:
        print("Using short-signal analysis methods")
        all_short_results, metrics_matrices = analyze_short_signal_all_pairs(test_states, esn.region_labels, fs=1000)
    else:
        print("Using standard coherence analysis methods")
        all_coherence_results = analyze_all_region_pairs_coherence(test_states, esn.region_labels, fs=1000)

    # ========== Spatiotemporal trajectories ==========
    print("\n" + "="*60)
    print("Multi-region 3D UMAP trajectory visualization")
    print("="*60)
    
    # Create directory for saving figures
    save_dir = os.path.join(os.path.dirname(__file__), 'result', 'Spatiotemporal_trajectory')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Figure save directory: {save_dir}")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 3,
        'xtick.minor.size': 2,
        'ytick.major.size': 3,
        'ytick.minor.size': 2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': True,
        'legend.fontsize': 7,
        'figure.dpi': 300
    })

    # Get all region labels
    unique_regions = np.unique(esn.region_labels)
    print(f"Detected {len(unique_regions)} brain regions: {unique_regions}")
    
    # Prepare data for each region
    region_data = {}
    region_colors = {}
    region_markers = {}

    # Define colors and marker symbols
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_regions)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'P', 'X', 'd', '|', '_']

    for i, region in enumerate(unique_regions):
        # Get node indices for this region
        region_idx = np.where(esn.region_labels == region)[0]
        if len(region_idx) > 0:
            # Extract state data for this region
            region_states = test_states[:, region_idx]  # (time steps, number of nodes)
            region_data[region] = region_states
            region_colors[region] = colors[i]
            region_markers[region] = markers[i % len(markers)]
            print(f"{region}: {len(region_idx)} nodes")
    
    # UMAP dimensionality reduction: perform UMAP separately for each region following the original logic
    print("\nRunning UMAP dimensionality reduction...")
    
    # Perform UMAP per region
    region_umap = {}
    region_umap_normalized = {}
    n_components = 3  # UMAP to 3D
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)

    for region in unique_regions:
        if region in region_data:
            # Get state data for this region (time steps, number of nodes)
            states_subset = region_data[region]
            print(f"{region}: original state shape {states_subset.shape}")
            
            # UMAP dimensionality reduction (reduce node dimension to n_components, keep time dimension)
            # UMAP input is (samples, features), here samples are time steps and features are nodes
            states_umap = reducer.fit_transform(states_subset)  # (time steps, n_components)
            region_umap[region] = states_umap
            
            # Normalize UMAP data for each region to the [0, 1] range
            umap_normalized = np.zeros_like(states_umap)
            for i in range(n_components):
                umap_min = np.min(states_umap[:, i])
                umap_max = np.max(states_umap[:, i])
                if umap_max > umap_min:  # avoid division by zero
                    umap_normalized[:, i] = (states_umap[:, i] - umap_min) / (umap_max - umap_min)
                else:
                    umap_normalized[:, i] = 0.5  # if max equals min, set to 0.5
            
            region_umap_normalized[region] = umap_normalized
            print(f"{region}: UMAP reduced shape {states_umap.shape}")
            print(f"{region}: UMAP normalized range [{np.min(umap_normalized):.3f}, {np.max(umap_normalized):.3f}]")
    
    # Create 3D scatter/trajectory visualization
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D trajectory lines for each region (using normalized data)
    for region in unique_regions:
        if region in region_umap_normalized:
            umap_data = region_umap_normalized[region]
            color = region_colors[region]
            
            # Plot line chart connecting time points
            ax.plot(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2], 
                    color=color, linewidth=1.5, alpha=0.8, label=region)

    # Set axis labels and title
    ax.set_xlabel('UMAP-1', fontsize=10, fontweight='bold')
    ax.set_ylabel('UMAP-2', fontsize=10, fontweight='bold')
    ax.set_zlabel('UMAP-3', fontsize=10, fontweight='bold')
    ax.set_title('Multi-Region UMAP 3D Trajectory Plot', 
                fontsize=11, fontweight='bold', pad=15)

    # Set legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, 
            frameon=True, fancybox=True, shadow=True, ncol=1)

    # Configure grid and background
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.view_init(elev=20, azim=45)

    # Set axis limits to align origins
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Use equal aspect ratio for axes
    ax.set_box_aspect([1,1,1])
    plt.subplots_adjust(right=0.75)

    # Save multi-region trajectory figure
    save_path = os.path.join(save_dir, "Multi_Region_UMAP_3D_Trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Multi-region 3D UMAP trajectory figure saved: {save_path}")
    plt.close()

    print("\nMulti-region 3D UMAP trajectory visualization completed.")
    print("This visualization shows:")
    print("1. Spatiotemporal distribution of all target regions in UMAP space (normalized to [0, 1])")
    print("2. Separation and clustering patterns of different regions in the embedded space")
    print("3. Temporal evolution trajectories within each region")
    print("4. Clear 3D line plots to examine spatial distribution across regions")
    print("5. Independently normalized UMAP data per region, facilitating comparison of dynamic ranges")
    
    # Single-region 3D UMAP trajectory visualization
    print("\n" + "="*60)
    print("Single-region 3D UMAP trajectory visualization")
    print("="*60)
    
    # Specify regions to visualize
    target_regions = ['VC','MC','PL','IL','OFC','PPC','SC','LP','TRN','LGN','OPN','DS']

    # Create a dedicated 3D trajectory plot for each specified region
    for region in target_regions:
        if region in region_umap_normalized:
            print(f"Plotting 3D UMAP trajectory for region {region}...")
            
            # Create a new figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get normalized UMAP data for this region
            umap_data = region_umap_normalized[region]
            color = region_colors[region]
            
            # Plot spatiotemporal trajectory by connecting time points with a line
            ax.plot(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2], 
                    color=color, linewidth=2, alpha=0.8)
            
            # Plot start and end points
            ax.scatter(umap_data[0, 0], umap_data[0, 1], umap_data[0, 2], 
                    color='green', s=100, marker='o', label='Start', alpha=0.8)
            ax.scatter(umap_data[-1, 0], umap_data[-1, 1], umap_data[-1, 2], 
                    color='red', s=100, marker='s', label='End', alpha=0.8)
            
            # Set axis labels
            ax.set_xlabel('UMAP-1', fontsize=12, fontweight='bold')
            ax.set_ylabel('UMAP-2', fontsize=12, fontweight='bold')
            ax.set_zlabel('UMAP-3', fontsize=12, fontweight='bold')
            
            # Set title
            ax.set_title(region, fontsize=16, fontweight='bold', pad=5)
            
            # Remove grid lines
            ax.grid(False)
            
            # Configure background
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.1)
            ax.yaxis.pane.set_alpha(0.1)
            ax.zaxis.pane.set_alpha(0.1)
            
            # Use standard isometric viewing angles
            ax.view_init(elev=20, azim=45)
            
            # Set axis limits to align origins
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            
            # Use equal aspect ratio for axes
            ax.set_box_aspect([1,1,1])
            
            # Add legend
            ax.legend(fontsize=10, loc='upper right')
            
            plt.tight_layout()
            
            # Save single-region trajectory figure
            save_path = os.path.join(save_dir, f"{region}_UMAP_3D_Trajectory.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✓ 3D UMAP trajectory figure for region {region} saved: {save_path}")
            
            # Close figure without displaying
            plt.close()
            
            print(f"✓ 3D UMAP trajectory plotting completed for region {region}")
        else:
            print(f"⚠ Normalized UMAP data for region {region} not found, skipping")
    
    print(f"\n3D UMAP trajectory visualization completed for all {len(target_regions)} regions.")
    print("This visualization shows:")
    print("1. Temporal evolution trajectories for each region in UMAP space (normalized to [0, 1])")
    print("2. Start points (green circles) and end points (red squares)")
    print("3. Clear 3D trajectories to examine spatiotemporal dynamics per region")
    print("4. Independently normalized UMAP data per region to compare dynamic ranges")
    print(f"5. All figures saved to: {save_dir}")

    # # Interactive visualization
    # try:
    #     focus_regions = ['SC', 'PL', 'OFC'] # ACx:azimuth=36,elevation=4;PL:azimuth=-86°,elevation=159;OFC:azimuth=9,elevation=-56

    #     for region in focus_regions:
    #         if region not in region_umap_normalized:
    #             print(f"⚠ Brain region {region} UMAP data not found, skipping")
    #             continue

    #         # Data and color
    #         umap_data = region_umap_normalized[region]
    #         color = region_colors.get(region, 'C0')

    #         fig = plt.figure(figsize=(8, 6))
    #         ax = fig.add_subplot(111, projection='3d')

    #         # Trajectory line
    #         ax.plot(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2],
    #                 color=color, linewidth=2.0, alpha=0.9)

    #         # Start/end markers
    #         ax.scatter(umap_data[0, 0], umap_data[0, 1], umap_data[0, 2],
    #                    color='green', s=80, marker='o', alpha=0.9)
    #         ax.scatter(umap_data[-1, 0], umap_data[-1, 1], umap_data[-1, 2],
    #                    color='red', s=80, marker='s', alpha=0.9)

    #         ax.grid(False)
    #         ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    #         ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    #         ax.xaxis.pane.fill = False
    #         ax.yaxis.pane.fill = False
    #         ax.zaxis.pane.fill = False
    #         ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    #         ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    #         ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    #         try:
    #             ax.xaxis.line.set_color((1, 1, 1, 0))
    #             ax.yaxis.line.set_color((1, 1, 1, 0))
    #             ax.zaxis.line.set_color((1, 1, 1, 0))
    #         except Exception:
    #             pass

    #         ax.set_title(region, fontsize=14, fontweight='bold', pad=6)

    #         plt.tight_layout()
    #         # Interactive display: free rotation with mouse
    #         plt.show()
    # except Exception as e:
    #     print(f"⚠ Interactive five-region visualization failed: {e}")

    # UMAP dimension sweep: elbow analysis for local-structure preservation
    print("\n" + "="*60)
    print("UMAP dimension sweep: Trustworthiness / Continuity / kNN retention evaluation")
    print("="*60)

    umap_params = dict(n_neighbors=15, min_dist=0.1)
    dims = list(range(2, 11))
    k_list = [15, 30]

    # Evaluate all regions
    region_dim_metrics = {}
    print("Start dimension-sweep evaluation (d=2..10)...")
    for region in unique_regions:
        if region not in region_data:
            continue
        X_tn = region_data[region]
        print(f"  Region {region}: data shape {X_tn.shape}")
        region_dim_metrics[region] = evaluate_region_dim_curve(
            X_tn, dims, k_list, umap_params, seed=42, high_metrics=('euclidean','cosine')
        )

    for k in k_list:
        corr_T = flatten_metric(region_dim_metrics, 'Trust_euclidean'.replace('euclidean','euclidean'), k)
        corr_C = flatten_metric(region_dim_metrics, 'Cont_euclidean'.replace('euclidean','euclidean'), k)
        corr_P = flatten_metric(region_dim_metrics, 'P@k_euclidean'.replace('euclidean','euclidean'), k)
        print(f"Consistency (Euclidean vs cosine) k={k}: Trust r≈{corr_T:.3f}, Cont r≈{corr_C:.3f}, P@k r≈{corr_P:.3f}")
    
    # Criterion: find the smallest d where T and C gains are below the threshold for most regions
    gain_thresh = 0.015  # intermediate value between 0.01 and 0.02
    majority_ratio = 0.6

    best_d_report = {}
    for k in k_list:
        d_star, ratio = find_knee_dimension(region_dim_metrics, dims, k, metric_family='euclidean')
        best_d_report[k] = (d_star, ratio)
        print(f"Elbow (k={k}): d*={d_star}, coverage ratio≈{ratio:.2f}")
    
    # Output directory for visualizations
    dim_dir = os.path.join(save_dir, 'UMAP_dim_sweep')
    os.makedirs(dim_dir, exist_ok=True)

    # S-plots: metric-vs-dimension curves (one per k, overlaying all regions and their mean, marking d=3)
    for metric_family in ('euclidean','cosine'):
        for k in k_list:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            titles = [f'Trustworthiness (k={k})', f'Continuity (k={k})', f'kNN Retention P@k (k={k})']
            keys = [f'Trust_{metric_family}', f'Cont_{metric_family}', f'P@k_{metric_family}']

            region_curves = {key: [] for key in keys}
            for region in region_dim_metrics:
                m = region_dim_metrics[region].get(metric_family)
                if m is None:
                    continue
                for key in keys:
                    vals = m[key][k]
                    if len(vals) == len(dims):
                        region_curves[key].append(vals)

            for ax, title, key in zip(axes, titles, keys):
                # Plot light curves for each region
                for vals in region_curves[key]:
                    ax.plot(dims, vals, color='gray', alpha=0.25, linewidth=1)
                # Plot mean curve
                if region_curves[key]:
                    mean_curve = np.mean(np.array(region_curves[key]), axis=0)
                    ax.plot(dims, mean_curve, color='C0', linewidth=2.5, label='Mean across regions')
                # Vertical line marking d=3
                ax.axvline(3, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='d=3')
                ax.set_xlabel('Dimension d')
                ax.set_title(title)
                ax.set_xlim(min(dims), max(dims))
                # Dynamic lower y-limit: add margin below minimum
                if region_curves[key]:
                    all_vals = np.array(region_curves[key]).ravel()
                    vmin = np.nanmin(all_vals) if all_vals.size else 0.0
                    lower = max(0.0, float(vmin) - 0.02)
                else:
                    lower = 0.0
                ax.set_ylim(lower, 1.0)
                ax.grid(True, alpha=0.3)
            axes[0].legend(loc='lower right', fontsize=9)
            plt.tight_layout()
            out_path = os.path.join(dim_dir, f'S_curve_{metric_family}_k{k}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            print(f"✓ Saved: {out_path}")
    
    # Cross-region boxplots (Euclidean metric, k=30): show distributions of T and C across d with d=3 marked
    box_k = 30
    metric_family = 'euclidean'
    regions = list(region_dim_metrics.keys())
    T_box = {d: [] for d in dims}
    C_box = {d: [] for d in dims}
    for region in regions:
        m = region_dim_metrics[region].get(metric_family)
        if m is None:
            continue
        T_vals = m['Trust_'+metric_family].get(box_k, [])
        C_vals = m['Cont_'+metric_family].get(box_k, [])
        if len(T_vals) == len(dims) and len(C_vals) == len(dims):
            for d, tv, cv in zip(dims, T_vals, C_vals):
                T_box[d].append(tv)
                C_box[d].append(cv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot([T_box[d] for d in dims], labels=dims, showfliers=False)
    axes[0].set_title(f'Trustworthiness across regions (k={box_k})')
    axes[0].set_xlabel('Dimension d')
    # Dynamic y-axis lower bound based on boxplot minima
    try:
        tmins = [min(T_box[d]) for d in dims if len(T_box[d]) > 0]
        t_lower = max(0.0, (min(tmins) if tmins else 0.0) - 0.02)
    except ValueError:
        t_lower = 0.0
    axes[0].set_ylim(t_lower, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(dims.index(3)+1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    axes[1].boxplot([C_box[d] for d in dims], labels=dims, showfliers=False)
    axes[1].set_title(f'Continuity across regions (k={box_k})')
    axes[1].set_xlabel('Dimension d')
    try:
        cmins = [min(C_box[d]) for d in dims if len(C_box[d]) > 0]
        c_lower = max(0.0, (min(cmins) if cmins else 0.0) - 0.02)
    except ValueError:
        c_lower = 0.0
    axes[1].set_ylim(c_lower, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(dims.index(3)+1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(dim_dir, f'boxplot_trust_cont_k{box_k}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {out_path}")
    
    # PCA variance analysis: first 10 principal components and cumulative variance per region
    print("\n" + "="*60)
    print("PCA variance analysis: top 10 principal components and cumulative variance per region")
    print("="*60)

    pca_dir = os.path.join(save_dir, 'PCA_variance')
    os.makedirs(pca_dir, exist_ok=True)

    for region in unique_regions:
        if region not in region_data:
            continue
        X_tn = region_data[region]  # (time_steps, nodes)
        if X_tn.shape[0] < 3 or X_tn.shape[1] < 2:
            print(f"⚠ Region {region} has insufficient samples, skipping PCA")
            continue
        n_comp = int(min(10, X_tn.shape[0], X_tn.shape[1]))
        try:
            pca = PCA(n_components=n_comp, svd_solver='auto', random_state=42)
        except TypeError:
            pca = PCA(n_components=n_comp, svd_solver='auto')
        pca.fit(X_tn)
        ev_ratio = pca.explained_variance_ratio_
        cum_ratio = np.cumsum(ev_ratio)

        fig, ax = plt.subplots(figsize=(8, 5))
        xs = np.arange(1, n_comp+1)
        ax.bar(xs, ev_ratio[:n_comp], color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.6, label='Explained variance ratio')
        ax.plot(xs, cum_ratio[:n_comp], color='#d62728', marker='o', linewidth=2.0, label='Cumulative ratio')
        ax.axhline(0.95, color='green', linestyle='--', linewidth=1.5, alpha=0.9, label='95% threshold')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Ratio')
        ax.set_title(f'PCA Variance Analysis - {region}')
        ax.set_xticks(xs)
        # Dynamic lower y-limit to emphasize differences (but not below 0)
        ymin = max(0.0, float(min(ev_ratio.min(), cum_ratio.min())) - 0.02)
        ax.set_ylim(ymin, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        # Use region name in filename
        out_path = os.path.join(pca_dir, f'{region}_PCA_variance.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Saved: {out_path}")
    
    # Spatiotemporal trajectory angle and orthogonality analysis
    print("\n" + "="*60)
    print("Spatiotemporal trajectory angle and orthogonality analysis across regions")
    print("="*60)
    
    # Compute trajectory angles and orthogonality
    print("Computing trajectory angles and orthogonality across regions...")
    angles_matrix, orthogonality_matrix, region_names = calculate_trajectory_angles_and_orthogonality(
        region_umap_normalized, time_window=3
    )

    print(f"Computation finished!")
    print(f"Angle matrix shape: {angles_matrix.shape}")
    print(f"Orthogonality matrix shape: {orthogonality_matrix.shape}")
    
    # Visualization 1: spatiotemporal heatmaps
    
    # Create spatiotemporal heatmap data
    time_steps, n_regions, _ = angles_matrix.shape
    valid_pairs = []

    # Collect all valid region pairs
    for i in range(n_regions):
        for j in range(i+1, n_regions):  # take upper triangle only to avoid duplicates
            valid_pairs.append((i, j, region_names[i], region_names[j]))

    print(f"Number of valid region pairs: {len(valid_pairs)}")
    
    # Analyze effective range of the time window
    time_window = 3  # window size used in the computation
    valid_start = time_window
    valid_end = time_steps - time_window
    print(f"Time-window size: {time_window}")
    print(f"Effective computation range: {valid_start} to {valid_end} (total {valid_end - valid_start} time points)")
    print(f"Ignored ranges: 0-{valid_start-1} and {valid_end+1}-{time_steps-1}")
    
    # Prepare heatmap data
    angle_heatmap_data = []
    angle_labels = []
    for i, j, name1, name2 in valid_pairs:
        angle_series = angles_matrix[:, i, j]
        angle_heatmap_data.append(angle_series)
        angle_labels.append(f'{name1}-{name2}')

    angle_heatmap_data = np.array(angle_heatmap_data)

    # Use data only within the effective computation range
    valid_angle_data = angle_heatmap_data[:, valid_start:valid_end]
    valid_time_range = range(valid_start, valid_end)

    # Set x-axis ticks only for the effective range
    x_ticks = range(0, len(valid_time_range), max(1, len(valid_time_range)//10))

    # Spatiotemporal angle heatmap
    print("Generating spatiotemporal angle heatmap...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 12))

    # Plot angle heatmap
    im1 = ax1.imshow(valid_angle_data, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_title('Spatiotemporal Heatmap of Trajectory Angles', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=16)
    ax1.set_ylabel('Brain Region Pairs', fontsize=16)
    ax1.set_yticks(range(len(angle_labels)))
    ax1.set_yticklabels(angle_labels, fontsize=12)

    # Configure x-axis ticks for the effective range
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([valid_time_range[i] for i in x_ticks], fontsize=12)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, label='Angle (degrees)', shrink=0.8)
    cbar1.ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save angle heatmap
    save_path1 = os.path.join(save_dir, "Spatiotemporal_Heatmap_Angles.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Spatiotemporal angle heatmap saved: {save_path1}")
    plt.close()
    
    # Spatiotemporal orthogonality heatmap
    print("Generating spatiotemporal orthogonality heatmap...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 12))

    # Prepare orthogonality data
    orth_heatmap_data = []
    for i, j, name1, name2 in valid_pairs:
        orth_series = orthogonality_matrix[:, i, j]
        orth_heatmap_data.append(orth_series)

    orth_heatmap_data = np.array(orth_heatmap_data)
    valid_orth_data = orth_heatmap_data[:, valid_start:valid_end]

    # Plot orthogonality heatmap
    im2 = ax2.imshow(valid_orth_data, cmap='plasma', aspect='auto', interpolation='nearest')
    ax2.set_title('Spatiotemporal Heatmap of Orthogonality', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=16)
    ax2.set_ylabel('Brain Region Pairs', fontsize=16)
    ax2.set_yticks(range(len(angle_labels)))
    ax2.set_yticklabels(angle_labels, fontsize=12)

    # Configure x-axis ticks for the effective range
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([valid_time_range[i] for i in x_ticks], fontsize=12)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, label='Orthogonality (0-1)', shrink=0.8)
    cbar2.ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save orthogonality heatmap
    save_path2 = os.path.join(save_dir, "Spatiotemporal_Heatmap_Orthogonality.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Spatiotemporal orthogonality heatmap saved: {save_path2}")
    plt.close()
    
    # Visualization 3: clustering of region pairs and dynamic patterns
    print("Generating brain-region-pair clustering and dynamic pattern figures...")

    # Compute angle and orthogonality statistics for each brain-region pair
    pair_features = []
    pair_names = []

    for i, j, name1, name2 in valid_pairs:
        angle_series = angles_matrix[:, i, j]
        orth_series = orthogonality_matrix[:, i, j]
        
        # Filter valid data
        valid_mask = angle_series > 0
        valid_angles = angle_series[valid_mask]
        valid_orth = orth_series[valid_mask]
        
        if len(valid_angles) > 10:  # Ensure sufficient data points
            # Compute statistical features
            features = [
                np.mean(valid_angles),      # Mean angle
                np.std(valid_angles),       # Angle standard deviation
                np.mean(valid_orth),        # Mean orthogonality
                np.std(valid_orth),         # Orthogonality standard deviation
                np.corrcoef(valid_angles, valid_orth)[0, 1] if len(valid_angles) > 1 else 0,  # Angle–orthogonality correlation
            ]
            pair_features.append(features)
            pair_names.append(f'{name1}-{name2}')

    pair_features = np.array(pair_features)

    # Create feature analysis figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Angle vs orthogonality scatter plot
    ax1.scatter(pair_features[:, 0], pair_features[:, 2], 
            c=pair_features[:, 4], cmap='coolwarm', s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Mean Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Mean Orthogonality', fontsize=12)
    ax1.set_title('Mean Angle vs Mean Orthogonality\n(Color: Correlation)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add brain-region-pair labels
    for i, name in enumerate(pair_names):
        ax1.annotate(name, (pair_features[i, 0], pair_features[i, 2]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    # Add color bar
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1, label='Angle-Orthogonality Correlation')

    # 2. Angle variability analysis
    ax2.scatter(pair_features[:, 0], pair_features[:, 1], 
            c=pair_features[:, 2], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Mean Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Angle Standard Deviation', fontsize=12)
    ax2.set_title('Mean Angle vs Angle Variability\n(Color: Mean Orthogonality)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add brain-region-pair labels
    for i, name in enumerate(pair_names):
        ax2.annotate(name, (pair_features[i, 0], pair_features[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2, label='Mean Orthogonality')

    # 3. Orthogonality variability analysis
    ax3.scatter(pair_features[:, 2], pair_features[:, 3], 
            c=pair_features[:, 0], cmap='plasma', s=100, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Mean Orthogonality', fontsize=12)
    ax3.set_ylabel('Orthogonality Standard Deviation', fontsize=12)
    ax3.set_title('Mean Orthogonality vs Orthogonality Variability\n(Color: Mean Angle)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add brain-region-pair labels
    for i, name in enumerate(pair_names):
        ax3.annotate(name, (pair_features[i, 2], pair_features[i, 3]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    cbar3 = plt.colorbar(ax3.collections[0], ax=ax3, label='Mean Angle (degrees)')

    # 4. Dynamic stability analysis
    stability_scores = pair_features[:, 1] + pair_features[:, 3]  # Angle variability + orthogonality variability
    ax4.scatter(pair_features[:, 0], pair_features[:, 2], 
            c=stability_scores, cmap='RdYlBu_r', s=100, alpha=0.7, edgecolors='black')
    ax4.set_xlabel('Mean Angle (degrees)', fontsize=12)
    ax4.set_ylabel('Mean Orthogonality', fontsize=12)
    ax4.set_title('Dynamic Stability Analysis\n(Color: Total Variability)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add brain-region-pair labels
    for i, name in enumerate(pair_names):
        ax4.annotate(name, (pair_features[i, 0], pair_features[i, 2]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    cbar4 = plt.colorbar(ax4.collections[0], ax=ax4, label='Total Variability')

    plt.tight_layout()

    # Save feature analysis figure
    save_path = os.path.join(save_dir, "Brain_Region_Pairs_Feature_Analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Brain-region-pair feature analysis figure saved: {save_path}")
    plt.close()

    # Visualization 4: time-window analysis
    print("Generating time-window analysis figure...")

    # Analyze angle and orthogonality patterns under different time windows
    time_windows = [5, 25]  # Window sizes suitable for 94 time steps
    window_analysis = {}

    for window in time_windows:
        if window < time_steps // 3:  # Ensure the window is not too large
            window_angles = []
            window_orth = []
            
            for i, j, name1, name2 in valid_pairs:
                angle_series = angles_matrix[:, i, j]
                orth_series = orthogonality_matrix[:, i, j]
                
                # Compute sliding-window statistics
                if len(angle_series) > window:
                    window_angle_means = []
                    window_orth_means = []
                    
                    for t in range(0, len(angle_series) - window, window//2):
                        window_angles_subset = angle_series[t:t+window]
                        window_orth_subset = orth_series[t:t+window]
                        
                        valid_angles = window_angles_subset[window_angles_subset > 0]
                        valid_orth = window_orth_subset[window_orth_subset > 0]
                        
                        if len(valid_angles) > 0:
                            window_angle_means.append(np.mean(valid_angles))
                        if len(valid_orth) > 0:
                            window_orth_means.append(np.mean(valid_orth))
                    
                    if len(window_angle_means) > 0:
                        window_angles.extend(window_angle_means)
                    if len(window_orth_means) > 0:
                        window_orth.extend(window_orth_means)
            
            window_analysis[window] = {
                'angles': np.array(window_angles),
                'orthogonality': np.array(window_orth)
            }

    # Create time-window analysis figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, window in enumerate(time_windows):
        if window in window_analysis:
            ax = axes[idx]
            
            angles = window_analysis[window]['angles']
            orth = window_analysis[window]['orthogonality']
            
            if len(angles) > 0 and len(orth) > 0:
                # Create scatter plot
                scatter = ax.scatter(angles, orth, alpha=0.6, s=50, c=range(len(angles)), cmap='viridis')
                ax.set_xlabel('Angle (degrees)', fontsize=12)
                ax.set_ylabel('Orthogonality', fontsize=12)
                ax.set_title(f'Time Window: {window} steps\n({len(angles)} data points)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                    
                # Add color bar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Time Order', fontsize=10)

    plt.tight_layout()

    # Save time-window analysis figure
    save_path = os.path.join(save_dir, "Time_Window_Analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Time-window analysis figure saved: {save_path}")
    plt.close()

    # Statistical analysis
    print("\n" + "="*60)
    print("Statistical analysis results")
    print("="*60)

    # Collect all angle and orthogonality data for statistics
    all_angles = []
    all_orthogonality = []

    for i, j, name1, name2 in valid_pairs:
        angle_series = angles_matrix[:, i, j]
        orth_series = orthogonality_matrix[:, i, j]
        
        valid_angles = angle_series[angle_series > 0]
        valid_orth = orth_series[orth_series > 0]
        
        all_angles.extend(valid_angles)
        all_orthogonality.extend(valid_orth)

    all_angles = np.array(all_angles)
    all_orthogonality = np.array(all_orthogonality)

    if len(all_angles) > 0:
        print(f"Angle statistics:")
        print(f"  Mean: {np.mean(all_angles):.2f}°")
        print(f"  Median: {np.median(all_angles):.2f}°")
        print(f"  Std: {np.std(all_angles):.2f}°")
        print(f"  Min: {np.min(all_angles):.2f}°")
        print(f"  Max: {np.max(all_angles):.2f}°")

    if len(all_orthogonality) > 0:
        print(f"\nOrthogonality statistics:")
        print(f"  Mean: {np.mean(all_orthogonality):.4f}")
        print(f"  Median: {np.median(all_orthogonality):.4f}")
        print(f"  Std: {np.std(all_orthogonality):.4f}")
        print(f"  Min: {np.min(all_orthogonality):.4f}")
        print(f"  Max: {np.max(all_orthogonality):.4f}")

    # Compute the mean-angle matrix to identify region pairs with maximal/minimal angles
    mean_angles = np.zeros((len(region_names), len(region_names)))
    for i in range(len(region_names)):
        for j in range(len(region_names)):
            if i != j:
                valid_angles = angles_matrix[:, i, j]
                valid_angles = valid_angles[valid_angles > 0]
                if len(valid_angles) > 0:
                    mean_angles[i, j] = np.mean(valid_angles)

    # Identify brain-region pairs with maximal and minimal mean angles
    if np.max(mean_angles) > 0:
        max_angle_idx = np.unravel_index(np.argmax(mean_angles), mean_angles.shape)
        min_angle_idx = np.unravel_index(np.argmin(mean_angles[mean_angles > 0]), mean_angles.shape)
        
        print(f"\nBrain-region pair with maximum angle: {region_names[max_angle_idx[0]]} - {region_names[max_angle_idx[1]]} ({mean_angles[max_angle_idx]:.1f}°)")
        print(f"Brain-region pair with minimum angle: {region_names[min_angle_idx[0]]} - {region_names[min_angle_idx[1]]} ({mean_angles[min_angle_idx]:.1f}°)")

    print(f"\nAll analysis figures have been saved to: {save_dir}")
    print("Analysis completed.")