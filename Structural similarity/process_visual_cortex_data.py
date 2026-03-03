import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import json
import gc  # CPU garbage collection
from caveclient import CAVEclient
from scipy import stats
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RANDOM_SEED = 1757414760 
np.random.seed(RANDOM_SEED)
cp.random.seed(RANDOM_SEED)

# Get absolute path of current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, "synapses_pni_2.csv")
results_dir = os.path.join(current_dir, "visual_cortex_alpha_results")  # Results output directory
visual_cortex_result_dir = os.path.join(current_dir, "visual_cortex_result")  # Additional results directory

client = CAVEclient('minnie65_public')

# Configure CAVE client timeout and retry policy
def configure_cave_client():
    """Configure the global CAVE client to better tolerate network issues.
    
    Notes
    -----
    This updates the module-level ``client`` in place (timeout and HTTP retry policy).
    """
    # Set longer timeout
    client.timeout = 60  # 60 s timeout

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Create adapter
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Mount to client session
    if hasattr(client, 'session'):
        client.session.mount("http://", adapter)
        client.session.mount("https://", adapter)
    
    print("CAVE client configured with retry policy and timeout")

# Configure client
configure_cave_client()

def clear_gpu_memory():
    """Free CuPy memory pools to reduce GPU memory pressure.
    
    Notes
    -----
    Clears both the default and pinned memory pools (in place).
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def clear_cpu_memory():
    """Run Python garbage collection to release CPU-side memory.
    """
    gc.collect()

def save_ave_deg_results(ave_deg, alpha_results):
    """Persist results for a single ``ave_deg`` run to disk.
    
    Parameters
    ----------
    ave_deg : int
        Target average degree identifier used in the filename.
    alpha_results : dict
        Mapping from alpha (as string) to entropy (or other scalar result).
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    result_file = os.path.join(results_dir, f"ave_deg_{ave_deg}.txt")
    with open(result_file, 'w') as f:
        json.dump(alpha_results, f)

def load_ave_deg_results(ave_deg):
    """Load cached results for a single ``ave_deg`` run if present.
    
    Parameters
    ----------
    ave_deg : int
        Target average degree identifier used in the filename.
    
    Returns
    -------
    dict or None
        Parsed results dictionary if the cache exists; otherwise ``None``.
    """
    result_file = os.path.join(results_dir, f"ave_deg_{ave_deg}.txt")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def collect_all_results():
    """Collect all cached ``ave_deg`` result files from ``results_dir``.
    
    Returns
    -------
    dict
        Mapping ``ave_deg`` (int) to the stored results dictionary.
    """
    all_results = {}
    
    if not os.path.exists(results_dir):
        return all_results
    
    for filename in os.listdir(results_dir):
        if filename.startswith("ave_deg_") and filename.endswith(".txt"):
            try:
                parts = filename.split("_")
                if len(parts) >= 3:
                    # Extract numeric part, strip .txt suffix
                    number_part = parts[2].split(".")[0]
                    ave_deg = int(number_part)
                    with open(os.path.join(results_dir, filename), 'r') as f:
                        all_results[ave_deg] = json.load(f)
            except (ValueError, IndexError) as e:
                print(f"Skip unparseable file: {filename}, error: {e}")
                continue
    
    return all_results

def calculate_distances(coordinates):
    """Compute the pairwise Euclidean distance matrix on GPU.
    
    Parameters
    ----------
    coordinates : cupy.ndarray, shape (n, d)
        Node coordinates.
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Pairwise Euclidean distances (float32).
    """
    print("Computing pairwise neuron distances...")
    n = coordinates.shape[0]
    distances = cp.zeros((n, n), dtype=cp.float32)
    
    # Use GPU to compute distances
    for i in tqdm(range(n), desc="Computing distance matrix"):
        diff = coordinates - coordinates[i]
        distances[i] = cp.sqrt(cp.sum(diff**2, axis=1))
    
    print("Distance computation done")
    return distances

def calculate_connection_probabilities(distances, alpha, use_degree_constraint=True):
    """Build a distance-based connection probability matrix.
    
    Parameters
    ----------
    distances : cupy.ndarray, shape (n, n)
        Pairwise distance matrix.
    alpha : float
        Distance decay exponent.
    use_degree_constraint : bool, optional
        Whether to apply the degree heavy-tail balancing heuristic (ablation toggle).
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Probability matrix in [0, 1] with zero diagonal.
    """
    n = distances.shape[0]
    
    # Base distance-based probability
    prob_matrix = distances**(-alpha)
    cp.fill_diagonal(prob_matrix, 0)
    prob_matrix = prob_matrix/cp.max(prob_matrix)
    
    # If degree heavy-tail constraint not needed, return directly
    if not use_degree_constraint:
        print("Skip degree heavy-tail constraint (ablation)")
        prob_matrix = cp.clip(prob_matrix, 0, 1)
        return prob_matrix
    
    # Add multi-objective balance mechanism
    print("Applying multi-objective balance mechanism...")

    # Compute target degree distribution (from real network expectation)
    # Real network: most nodes low-degree, few high-degree
    target_degree_weights = cp.zeros(n, dtype=cp.float16)

    # Use power-law as target degree distribution
    # Most nodes low weight (low degree), few high weight
    node_indices = cp.arange(n, dtype=cp.float16)
    power_law_weights = 1.0 / (node_indices + 1)**0.8  # Power-law distribution
    power_law_weights = power_law_weights / cp.sum(power_law_weights) * n  # Normalize

    # Apply degree weights to probability matrix
    prob_matrix = prob_matrix * power_law_weights[:, cp.newaxis]
    prob_matrix = prob_matrix * power_law_weights[cp.newaxis, :]
    
    # Clear temporary variables
    del power_law_weights, node_indices
    cp.get_default_memory_pool().free_all_blocks()

    # Add adaptive balance mechanism (chunked version)
    print("Applying adaptive balance mechanism...")

    # Compute current network degree distribution
    node_connection_density = cp.sum(prob_matrix, axis=1)

    # Compute degree distribution skewness
    mean_density = cp.mean(node_connection_density)
    std_density = cp.std(node_connection_density)
    skewness = cp.mean(((node_connection_density - mean_density) / (std_density + 1e-6))**3)
    
    # Adjust balance parameters by skewness
    if skewness > 2.0:  # Strong right skew; boost low-degree nodes
        balance_factor = 0.5  # Reduce high-density node weight
        print(f"Detected strong right skew (skewness={skewness:.2f}), applying strong balance")
    elif skewness < 1.0:  # Distribution too uniform; increase skew
        balance_factor = 0.8  # Keep more high-density nodes
        print(f"Detected overly uniform distribution (skewness={skewness:.2f}), applying weak balance")
    else:  # Distribution acceptable
        balance_factor = 0.65  # Moderate balance
        print(f"Detected acceptable distribution (skewness={skewness:.2f}), applying moderate balance")

    # Apply adaptive balance
    density_threshold = cp.percentile(node_connection_density, 75)  # 75th percentile as threshold
    high_density_mask = node_connection_density > density_threshold
    
    # Apply adaptive penalty to high-density nodes
    penalty_factor = cp.ones_like(node_connection_density, dtype=cp.float16)
    penalty_factor[high_density_mask] = balance_factor

    # Apply penalty to probability matrix in blocks
    print("Applying adaptive balance in blocks...")
    block_size = 1000
    for i in tqdm(range(0, n, block_size), desc="Applying adaptive balance"):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            
            # Get current block
            prob_block = prob_matrix[i:end_i, j:end_j]
            penalty_i = penalty_factor[i:end_i, cp.newaxis]
            penalty_j = penalty_factor[j:end_j, cp.newaxis]
            
            # Apply penalty
            prob_matrix[i:end_i, j:end_j] = prob_block * penalty_i * penalty_j.T

            # Clear current block
            del prob_block, penalty_i, penalty_j
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear temporary variables
    del node_connection_density, density_threshold, high_density_mask, penalty_factor
    del mean_density, std_density, skewness
    cp.get_default_memory_pool().free_all_blocks()

    # Ensure probabilities in valid range
    prob_matrix = cp.clip(prob_matrix, 0, 1)

    print("Connection probability computation done")
    return prob_matrix

def generate_connection_matrix(prob_matrix, ave_deg, multiplier=1):
    """Sample a binary adjacency matrix from a probability matrix.
    
    Parameters
    ----------
    prob_matrix : cupy.ndarray, shape (n, n)
        Connection probabilities.
    ave_deg : float
        Target average degree (used only for logging/consistency).
    multiplier : float, optional
        Global scaling factor applied to probabilities before sampling.
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Binary adjacency matrix (int8).
    """
    print("Generating connection matrix...")
    # Ensure same random seed per run
    cp.random.seed(RANDOM_SEED)

    # Generate connection matrix: [0,1) uniform random on GPU, same shape as probability matrix
    random_matrix = cp.random.rand(*prob_matrix.shape)

    # Compare random values to probabilities
    result_matrix = random_matrix < prob_matrix * multiplier
    print("Connection matrix generation done")

    # Cast to integer (0 or 1)
    return result_matrix.astype(cp.int8)

# ----------------- Shared utilities -----------------
def _knn_from_distances_cpu(distances_cp, K=40, block_size=2048):
    """Extract per-node k-nearest neighbors from a GPU distance matrix (CPU-side).
    
    Parameters
    ----------
    distances_cp : cupy.ndarray, shape (n, n)
        Distance matrix stored on GPU.
    K : int, optional
        Number of neighbors to keep per node.
    block_size : int, optional
        Number of rows to transfer to CPU per block.
    
    Returns
    -------
    knn_idx : numpy.ndarray, shape (n, K)
        Neighbor indices for each node.
    knn_dist : numpy.ndarray, shape (n, K)
        Distances to the selected neighbors.
    mean_d : float
        Mean of all selected neighbor distances (used for cost normalization).
    """
    n = distances_cp.shape[0]
    K = int(K)
    knn_idx  = np.empty((n, K), dtype=np.int32)
    knn_dist = np.empty((n, K), dtype=np.float32)
    d_sum = 0.0
    d_cnt = 0

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        D_blk = distances_cp[i0:i1].get()  # CPU [b, n]
        # Take smallest K+1 (including self), then drop diagonal
        part = np.argpartition(D_blk, K, axis=1)[:, :K+1]
        rows = np.arange(D_blk.shape[0])[:, None]
        d_part = D_blk[rows, part]
        # Drop self column“自己”这一列（距离为0）
        self_mask = d_part > 0
        # Take up to K smallest non-self per row
        for r in range(D_blk.shape[0]):
            cand_idx = part[r][self_mask[r]]
            cand_dst = d_part[r][self_mask[r]]
            if cand_idx.size > K:
                sub = np.argpartition(cand_dst, K-1)[:K]
                cand_idx = cand_idx[sub]; cand_dst = cand_dst[sub]
            # Sort by distance ascending (stable)
            order = np.argsort(cand_dst)
            cand_idx = cand_idx[order][:K]
            cand_dst = cand_dst[order][:K]
            knn_idx[i0+r]  = cand_idx.astype(np.int32, copy=False)
            knn_dist[i0+r] = cand_dst.astype(np.float32, copy=False)
        d_sum += float(np.sum(knn_dist[i0:i1]))
        d_cnt += int(np.prod(knn_dist[i0:i1].shape))

    mean_d = d_sum / max(d_cnt, 1)
    return knn_idx, knn_dist, float(mean_d)


def _select_top_m_undirected(n, u, v, score, m):
    """Select top-scoring undirected edges with de-duplication (CPU-side).
    
    Parameters
    ----------
    n : int
        Number of nodes (used to form unique undirected keys).
    u, v : numpy.ndarray
        Candidate edge endpoints (same length).
    score : numpy.ndarray
        Candidate edge scores (higher is better).
    m : int
        Maximum number of undirected edges to return.
    
    Returns
    -------
    sel_u, sel_v : numpy.ndarray
        Selected undirected edge endpoints (length <= m).
    """
    # Undirected dedup key
    uu = np.minimum(u, v).astype(np.int64)
    vv = np.maximum(u, v).astype(np.int64)
    keys = uu * n + vv

    # Sort by key first
    order = np.argsort(keys, kind='mergesort')
    keys = keys[order]; uu = uu[order]; vv = vv[order]; score = score[order]

    # For each key take max score
    # Find start index of each key
    uniq, idx = np.unique(keys, return_index=True)
    # Use reduceat for segment max
    seg_max = np.maximum.reduceat(score, idx)
    seg_u = uu[idx]; seg_v = vv[idx]

    # Take top-m
    m = int(min(m, seg_max.size))
    top_idx = np.argpartition(-seg_max, m-1)[:m]
    sel = top_idx[np.argsort(-seg_max[top_idx])]  # Sort by score descending for stable output
    return seg_u[sel].astype(np.int32), seg_v[sel].astype(np.int32)


def _edges_to_cupy_dense(n, eu, ev):
    """Convert a CPU edge list to a symmetric dense adjacency matrix on GPU.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    eu, ev : array-like
        Undirected edge endpoints (CPU).
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Symmetric adjacency matrix (int8).
    
    Notes
    -----
    This allocates an ``n × n`` dense matrix on GPU; consider sparse formats if memory is tight.
    """
    A = cp.zeros((n, n), dtype=cp.int8)
    if len(eu) == 0:
        return A
    eu_cp = cp.asarray(eu, dtype=cp.int32)
    ev_cp = cp.asarray(ev, dtype=cp.int32)
    A[eu_cp, ev_cp] = 1
    A[ev_cp, eu_cp] = 1
    del eu_cp, ev_cp
    cp.get_default_memory_pool().free_all_blocks()
    return A

# ----------------- Economical cost-benefit -----------------
def generate_economical_connection_matrix(
        distances, ave_deg, beta=1.0, lambda_cost=0.6, gamma=1.0,
        K=None, block_size=2048):
    """Generate a network using an economical cost–benefit score on KNN candidates.
    
    Parameters
    ----------
    distances : cupy.ndarray, shape (n, n)
        Pairwise distance matrix.
    ave_deg : float
        Target average degree.
    beta, lambda_cost, gamma : float, optional
        Benefit weight, cost weight, and cost exponent.
    K : int or None, optional
        Number of nearest-neighbor candidates per node. Defaults to ``max(4*ave_deg, 40)``.
    block_size : int, optional
        Block size for CPU extraction of KNN information.
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Symmetric adjacency matrix (int8).
    
    Notes
    -----
    Score_ij = beta * (|KNN(i)∩KNN(j)|/K) - lambda_cost * (d_ij/mean_d)^gamma, evaluated on KNN candidates only.
    """
    n = distances.shape[0]
    target_edges = int(n * ave_deg / 2)
    if K is None:
        K = int(max(4*ave_deg, 40))
    K = int(K)

    print(f"[Economical] n={n}, target_edges={target_edges}, K={K}")
    # 1) Get KNN on CPU
    knn_idx, knn_dist, mean_d = _knn_from_distances_cpu(distances, K=K, block_size=block_size)

    # 2) Compute candidate edges and scores on CPU
    # Flatten to vectors first
    u = np.repeat(np.arange(n, dtype=np.int32), K)
    v = knn_idx.reshape(-1).astype(np.int32, copy=False)
    dij = knn_dist.reshape(-1).astype(np.float32, copy=False)

    # 2.1 Common-neighbor / matching index (K per row, use intersection)
    # For speed: sort KNN per row, use np.intersect1d for overlap
    knn_sorted = np.sort(knn_idx, axis=1)
    overlap = np.empty(u.size, dtype=np.float32)
    ptr = 0
    for i in range(n):
        Si = knn_sorted[i]
        for j in range(K):
            jj = v[ptr]
            Sj = knn_sorted[jj]
            # Overlap size
            c = np.intersect1d(Si, Sj, assume_unique=False).size
            overlap[ptr] = c / float(K)
            ptr += 1

    # 2.2 Cost term
    cost = (dij / (mean_d + 1e-12)) ** gamma

    # 2.3 Total score
    score = beta * overlap - lambda_cost * cost

    # 3) Select top-m undirected edges
    sel_u, sel_v = _select_top_m_undirected(n, u, v, score, target_edges)

    # 4) Write back to GPU adjacency
    A = _edges_to_cupy_dense(n, sel_u, sel_v)
    m = int(cp.sum(A).get()) // 2
    print(f"[Economical] edges={m}, density={m/(n*(n-1)/2):.6f}")
    return A

# ----------------- Fully random generation -----------------
def generate_random_connection_matrix(n, ave_deg, random_seed=None):
    """Generate a random undirected network with the target sparsity.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    ave_deg : float
        Target average degree.
    random_seed : int or None, optional
        Seed for reproducibility.
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Symmetric adjacency matrix (int8).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        cp.random.seed(random_seed)
    
    target_edges = int(n * ave_deg / 2)
    print(f"[Random] n={n}, target_edges={target_edges}")
    
    # Generate all possible undirected edge pairs
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only to avoid duplicates
            all_pairs.append((i, j))

    # Randomly select target_edges edges
    if len(all_pairs) < target_edges:
        target_edges = len(all_pairs)
        print(f"[Random] Warning: target edges exceed maximum; adjusted to {target_edges}")

    selected_pairs = np.random.choice(len(all_pairs), size=target_edges, replace=False)

    # Build adjacency matrix
    A = cp.zeros((n, n), dtype=cp.int8)
    for idx in selected_pairs:
        i, j = all_pairs[idx]
        A[i, j] = 1
        A[j, i] = 1  # Symmetric matrix

    m = int(cp.sum(A).get()) // 2
    print(f"[Random] edges={m}, density={m/(n*(n-1)/2):.6f}")
    return A

# ----------------- Distance + homophily -----------------
def generate_homophily_connection_matrix(
        distances, coordinates, ave_deg, alpha=1.0, eta=0.8,
        K=None, block_size=2048, homophily_axis=2):
    """Generate a network using distance decay plus continuous homophily on KNN candidates.
    
    Parameters
    ----------
    distances : cupy.ndarray, shape (n, n)
        Pairwise distance matrix.
    coordinates : cupy.ndarray, shape (n, d)
        Node coordinates used to compute homophily.
    ave_deg : float
        Target average degree.
    alpha : float, optional
        Distance decay exponent.
    eta : float, optional
        Homophily strength.
    K : int or None, optional
        Number of nearest-neighbor candidates per node. Defaults to ``max(4*ave_deg, 40)``.
    block_size : int, optional
        Block size for CPU extraction of KNN information.
    homophily_axis : int, optional
        Coordinate axis used for the homophily feature (default: 2).
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Symmetric adjacency matrix (int8).
    
    Notes
    -----
    For candidates (i, j), score_ij = d_ij^{-alpha} * exp(eta * h_ij) with h_ij = -|z_i - z_j| / sigma_z.
    """
    n = distances.shape[0]
    target_edges = int(n * ave_deg / 2)
    if K is None:
        K = int(max(4*ave_deg, 40))
    K = int(K)

    print(f"[Homophily] n={n}, target_edges={target_edges}, K={K}, alpha={alpha}, eta={eta}")
    # 1) Get KNN
    knn_idx, knn_dist, _ = _knn_from_distances_cpu(distances, K=K, block_size=block_size)

    # 2) Get continuous feature z for homophily on CPU
    coords = coordinates.get()
    if homophily_axis >= coords.shape[1]:
        raise ValueError("homophily_axis out of coordinate dimensions")
    z = coords[:, homophily_axis].astype(np.float32, copy=False)
    z_scale = np.std(z) + 1e-6

    # 3) Candidate edges and scores
    u = np.repeat(np.arange(n, dtype=np.int32), K)
    v = knn_idx.reshape(-1).astype(np.int32, copy=False)
    dij = knn_dist.reshape(-1).astype(np.float32, copy=False)

    # 3.1 Homophily (continuous): h_ij = -|z_i - z_j| / z_scale
    h = -np.abs(z[u] - z[v]) / z_scale

    # 3.2 Score: do not build NxN probability matrix, only compute for candidates
    score = (dij ** (-alpha)) * np.exp(eta * h)

    # 4) Select top-m undirected edges
    sel_u, sel_v = _select_top_m_undirected(n, u, v, score, target_edges)

    # 5) Write back to GPU adjacency
    A = _edges_to_cupy_dense(n, sel_u, sel_v)
    m = int(cp.sum(A).get()) // 2
    print(f"[Homophily] edges={m}, density={m/(n*(n-1)/2):.6f}")
    return A


def calculate_entropy(connection_matrix):
    """Compute average neighbor-entropy using 1st- and 2nd-order neighborhoods.
    
    Parameters
    ----------
    connection_matrix : cupy.ndarray, shape (n, n)
        Binary adjacency matrix.
    
    Returns
    -------
    float
        Average entropy across nodes (batch-processed to reduce memory use).
    """
    print("Computing information entropy...")
    n = connection_matrix.shape[0]
    batch_size = 500  # 500 nodes per batch
    total_entropy = 0
    total_nodes = 0
    
    # Compute second-order neighbor matrix (matrix squared)
    print("Computing second-order neighbor matrix...")
    second_order_matrix = cp.dot(connection_matrix, connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Computing entropy"):
        end_idx = min(i + batch_size, n)
        # Get current batch connectivity matrix
        batch_matrix = connection_matrix[i:end_idx]
        batch_second_order = second_order_matrix[i:end_idx]
        
        # Merge first- and second-order neighbors
        combined_matrix = batch_matrix + batch_second_order

        # Set nonzeros to 1
        combined_matrix = (combined_matrix > 0).astype(cp.int8)
        cp.fill_diagonal(combined_matrix, 0)
        
        # Compute current batch connection count
        row_sums = cp.sum(combined_matrix, axis=1)
        row_sums = cp.where(row_sums == 0, 1, row_sums)  # Avoid division by zero

        # Compute current batch entropy
        q_j = combined_matrix / row_sums[:, cp.newaxis]
        q_j = cp.where(q_j == 0, 1, q_j)  # Avoid log(0)
        batch_entropy = -cp.sum(q_j * cp.log(q_j), axis=1)

        # Accumulate current batch entropy
        total_entropy += float(cp.sum(batch_entropy))
        total_nodes += end_idx - i
    
    # Compute mean entropy
    avg_entropy = total_entropy / total_nodes if total_nodes > 0 else 0
    print("Information entropy computation done")
    return avg_entropy

def get_energy_bound(connection_matrix, distances, w):
    """Apply an energy budget constraint by pruning long connections per node.
    
    Parameters
    ----------
    connection_matrix : cupy.ndarray, shape (n, n)
        Binary adjacency matrix.
    distances : cupy.ndarray, shape (n, n)
        Pairwise distance matrix.
    w : float
        Per-node energy threshold (sum of distances for retained edges).
    
    Returns
    -------
    cupy.ndarray, shape (n, n)
        Pruned adjacency matrix (int8).
    """
    print(f"Applying energy constraint with threshold w={w:.2f}...")
    n = connection_matrix.shape[0]
    batch_size = 500  # 500 nodes per batch
    new_connection_matrix = cp.zeros_like(connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Applying energy constraint"):
        end_idx = min(i + batch_size, n)
        # Get current batch connections and distances
        batch_adj = connection_matrix[i:end_idx]
        batch_dis = distances[i:end_idx]

        # Process current batch in parallel
        sorted_indices = cp.argsort(batch_dis, axis=1)
        sorted_dis = cp.take_along_axis(batch_dis, sorted_indices, axis=1)
        sorted_adj = cp.take_along_axis(batch_adj, sorted_indices, axis=1)
        
        valid_connections = (sorted_adj != 0)
        energy_cumsum = cp.cumsum(sorted_dis * valid_connections, axis=1)
        threshold_mask = (energy_cumsum <= w) & valid_connections
        reverse_indices = cp.argsort(sorted_indices, axis=1)
        original_order_mask = cp.take_along_axis(threshold_mask, reverse_indices, axis=1)
        
        # Set new connections
        new_connection_matrix[i:end_idx] = original_order_mask.astype(cp.int8)

    print("Energy constraint application done")
    return new_connection_matrix

def calculate_network_topology_metrics_gpu(connection_matrix, eig_k=10):
    """Compute basic topology metrics on GPU (degree, clustering, density).
    
    Parameters
    ----------
    connection_matrix : array-like
        Adjacency matrix (NumPy or CuPy). Converted to CuPy if needed.
    eig_k : int, optional
        Reserved parameter (currently unused).
    
    Returns
    -------
    dict
        Dictionary containing scalar metrics and per-node arrays (returned on CPU).
    """
    print("[GPU] Computing network topology metrics...")
    # Ensure on GPU
    if isinstance(connection_matrix, np.ndarray):
        print("[GPU] Input is numpy array, converting to cupy...")
        adj_matrix = cp.array(connection_matrix)
    else:
        adj_matrix = connection_matrix

    n = adj_matrix.shape[0]
    print(f"[GPU] Network node count: {n}")

    # 1. Degree distribution
    print("[GPU] Computing degree distribution...")
    degrees = cp.sum(adj_matrix, axis=1)
    avg_degree = float(cp.mean(degrees).get())
    degree_std = float(cp.std(degrees).get())
    print(f"[GPU] Average degree: {avg_degree:.2f}, degree std: {degree_std:.2f}")

    # 2. Clustering coefficient (A^3 diagonal)
    print("[GPU] Computing clustering coefficient (triangle count, batched)...")
    batch_size = 2000 if n > 5000 else n
    triangles = cp.zeros(n, dtype=cp.float32)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        print(f"[GPU] Processing triangle count for nodes {i} to {end}...")
        sub_adj = adj_matrix[i:end]
        sub_cube = cp.matmul(cp.matmul(sub_adj, adj_matrix), adj_matrix)
        triangles[i:end] = cp.diag(sub_cube)
        del sub_adj, sub_cube
        cp.get_default_memory_pool().free_all_blocks()
    clustering_coeffs = cp.zeros(n, dtype=cp.float32)
    mask = degrees >= 2
    clustering_coeffs[mask] = triangles[mask] / (degrees[mask] * (degrees[mask] - 1))
    avg_clustering = float(cp.mean(clustering_coeffs).get())
    print(f"[GPU] Mean clustering coefficient: {avg_clustering:.4f}")

    # 3. Network density
    print("[GPU] Computing network density...")
    total_connections = float(cp.sum(adj_matrix).get()) / 2
    max_possible_connections = n * (n - 1) / 2
    network_density = total_connections / max_possible_connections
    print(f"[GPU] Network density: {network_density:.6f}")

    print("[GPU] Network topology metrics computation done.")
    return {
        'avg_degree': avg_degree,
        'degree_std': degree_std,
        'avg_clustering': avg_clustering,
        'network_density': network_density,
        'degree_distribution': degrees.get(),
        'clustering_coeffs': clustering_coeffs.get(),
        'triangles': triangles.get(),
    }

def calculate_topology_similarity(real_matrix, generated_matrix):
    """Compare two networks using basic topology and distribution tests.
    
    Parameters
    ----------
    real_matrix : array-like
        Reference adjacency matrix.
    generated_matrix : array-like
        Generated adjacency matrix to compare.
    
    Returns
    -------
    dict
        Summary similarity scores and KS-test results (degree, clustering, triangles), plus per-network metrics.
    """
    real_metrics = calculate_network_topology_metrics_gpu(real_matrix)
    gen_metrics = calculate_network_topology_metrics_gpu(generated_matrix)
    # Degree distribution KS test
    deg_ks, deg_p = stats.ks_2samp(real_metrics['degree_distribution'], gen_metrics['degree_distribution'])
    degree_similarity = 1 - deg_ks
    clustering_similarity = 1 - abs(real_metrics['avg_clustering'] - gen_metrics['avg_clustering'])
    clustering_similarity = max(0, clustering_similarity)
    # Disable path_similarity
    # Density similarity only
    density_similarity = 1 - abs(real_metrics['network_density'] - gen_metrics['network_density'])
    density_similarity = max(0, density_similarity)
    topology_similarity = (0.5 * degree_similarity + 0.3 * clustering_similarity + 0.2 * density_similarity)
    # Clustering coefficient distribution KS test
    clu_ks, clu_p = stats.ks_2samp(real_metrics['clustering_coeffs'], gen_metrics['clustering_coeffs'])
    # Triangle distribution KS test
    tri_ks, tri_p = stats.ks_2samp(real_metrics['triangles'], gen_metrics['triangles'])
    return {
        'topology_similarity': topology_similarity,
        'degree_similarity': degree_similarity,
        'clustering_similarity': clustering_similarity,
        'density_similarity': density_similarity,
        'ks_tests': {
            'degree': {'statistic': float(deg_ks), 'p_value': float(deg_p)},
            'clustering_coeffs': {'statistic': float(clu_ks), 'p_value': float(clu_p)},
            'triangles': {'statistic': float(tri_ks), 'p_value': float(tri_p)}
        },
        'real_metrics': real_metrics,
        'generated_metrics': gen_metrics
    }


def calculate_mvs(real_values, gen_values):
    """Compute Mean Value Similarity (MVS) between two value arrays.
    
    Parameters
    ----------
    real_values, gen_values : array-like
        Two sets of scalar samples.
    
    Returns
    -------
    float
        MVS score in [0, 1] (higher is more similar).
    """
    real_mean = np.mean(real_values)
    gen_mean = np.mean(gen_values)
    mvs = 1 - abs(real_mean - gen_mean) / max(real_mean, gen_mean)
    return mvs

def calculate_bray_curtis_dissimilarity(real_values, gen_values, *, bins=None, value_range=None, n_bins=50):
    """Compute Bray–Curtis dissimilarity between two 1D distributions via histograms.
    
    Parameters
    ----------
    real_values, gen_values : array-like
        Sample values for the two distributions.
    bins : array-like or None, optional
        Bin edges. If ``None``, bins are inferred from ``value_range`` or the data.
    value_range : tuple[float, float] or None, optional
        Range used to build bins when ``bins`` is ``None``.
    n_bins : int, optional
        Number of bins when bins are inferred.
    
    Returns
    -------
    float
        Bray–Curtis dissimilarity in [0, 1] (smaller is more similar).
    """
    real_values = np.asarray(real_values)
    gen_values = np.asarray(gen_values)

    if bins is None:
        if value_range is not None:
            lo, hi = value_range
            bins = np.linspace(lo, hi, n_bins + 1)
        else:
            # Default: adapt to discrete degree distribution (integer, right-open bins)
            bins = np.linspace(0, max(np.max(real_values), np.max(gen_values)) + 1, n_bins + 1)

    real_hist, _ = np.histogram(real_values, bins=bins)
    gen_hist, _ = np.histogram(gen_values, bins=bins)

    real_sum = np.sum(real_hist)
    gen_sum = np.sum(gen_hist)
    if real_sum == 0 and gen_sum == 0:
        return 0.0

    c_ij = np.sum(np.minimum(real_hist, gen_hist))
    bc = 1 - (2 * c_ij) / (real_sum + gen_sum)
    return float(bc)

def calculate_wasserstein_distance(real_values, gen_values):
    """Compute Wasserstein distance with fallbacks for robustness.
    
    Parameters
    ----------
    real_values, gen_values : array-like
        Sample values for the two distributions.
    
    Returns
    -------
    float
        Estimated Wasserstein distance (0.0 if all methods fail).
    """
    from scipy.stats import wasserstein_distance
    
    try:
        # Method 1: direct Wasserstein distance
        wd = wasserstein_distance(real_values, gen_values)
        return wd
    except Exception as e:
        print(f"Wasserstein direct computation failed: {e}")
        
        try:
            # Method 2: quantile-based
            quantiles = np.linspace(0.01, 0.99, 50)
            real_quantiles = np.quantile(real_values, quantiles)
            gen_quantiles = np.quantile(gen_values, quantiles)
            wd = np.mean(np.abs(real_quantiles - gen_quantiles))
            return wd
        except Exception as e2:
            print(f"Wasserstein quantile method failed: {e2}")
            
            try:
                # Method 3: histogram-based
                bins = np.linspace(0, max(np.max(real_values), np.max(gen_values)) + 1, 50)
                real_hist, _ = np.histogram(real_values, bins=bins, density=True)
                gen_hist, _ = np.histogram(gen_values, bins=bins, density=True)
                
                # Normalize
                real_hist = real_hist / (np.sum(real_hist) + 1e-10)
                gen_hist = gen_hist / (np.sum(gen_hist) + 1e-10)
                
                # Compute Wasserstein distance
                wd = wasserstein_distance(real_hist, gen_hist)
                return wd
            except Exception as e3:
                print(f"Wasserstein histogram method failed: {e3}")
                return 0.0

def calculate_js_divergence(real_values, gen_values, *, bins=None, value_range=None, n_bins=50):
    """Compute Jensen–Shannon distance between two 1D distributions via histograms.
    
    Parameters
    ----------
    real_values, gen_values : array-like
        Sample values for the two distributions.
    bins : array-like or None, optional
        Bin edges. If ``None``, bins are inferred from ``value_range`` or the data.
    value_range : tuple[float, float] or None, optional
        Range used to build bins when ``bins`` is ``None``.
    n_bins : int, optional
        Number of bins when bins are inferred.
    
    Returns
    -------
    float
        Jensen–Shannon distance (SciPy ``jensenshannon`` output; in [0, 1]).
    """
    from scipy.spatial.distance import jensenshannon

    real_values = np.asarray(real_values)
    gen_values = np.asarray(gen_values)

    if bins is None:
        if value_range is not None:
            lo, hi = value_range
            bins = np.linspace(lo, hi, n_bins + 1)
        else:
            bins = np.linspace(0, max(np.max(real_values), np.max(gen_values)) + 1, n_bins + 1)

    real_hist, _ = np.histogram(real_values, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_values, bins=bins, density=True)

    # Avoid all-zero bins for numerical stability
    eps = 1e-12
    real_hist = real_hist + eps
    gen_hist = gen_hist + eps

    real_hist = real_hist / np.sum(real_hist)
    gen_hist = gen_hist / np.sum(gen_hist)

    js_div = jensenshannon(real_hist, gen_hist)
    return float(js_div)

def calculate_hellinger_distance(real_values, gen_values):
    """Compute Hellinger distance between two 1D distributions via histograms.
    
    Parameters
    ----------
    real_values, gen_values : array-like
        Sample values for the two distributions.
    
    Returns
    -------
    float
        Hellinger distance (smaller is more similar).
    """
    # Convert distribution to histogram
    bins = np.linspace(0, max(np.max(real_values), np.max(gen_values)) + 1, 50)
    real_hist, _ = np.histogram(real_values, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_values, bins=bins, density=True)
    
    # Normalize
    real_hist = real_hist / np.sum(real_hist)
    gen_hist = gen_hist / np.sum(gen_hist)
    
    # Compute Hellinger distance
    hellinger_dist = np.sqrt(0.5 * np.sum((np.sqrt(real_hist) - np.sqrt(gen_hist))**2))
    return hellinger_dist

def calculate_degree_distribution_metrics(real_degrees, gen_degrees):
    """Compute multiple similarity metrics for degree distributions.
    
    Parameters
    ----------
    real_degrees, gen_degrees : array-like
        Degree samples from the real and generated networks.
    
    Returns
    -------
    dict
        Named similarity scores (cosine, Pearson, quantiles, skewness, kurtosis, etc.).
    """
    metrics = {}
    
    # 1. Cosine similarity (histogram-based)
    bins = np.linspace(0, max(np.max(real_degrees), np.max(gen_degrees)) + 1, 50)
    real_hist, _ = np.histogram(real_degrees, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_degrees, bins=bins, density=True)
    
    # Normalize
    real_hist = real_hist / (np.sum(real_hist) + 1e-10)
    gen_hist = gen_hist / (np.sum(gen_hist) + 1e-10)
    
    # Cosine similarity
    cosine_sim = np.dot(real_hist, gen_hist) / (np.linalg.norm(real_hist) * np.linalg.norm(gen_hist) + 1e-10)
    metrics['cosine_similarity'] = max(0, cosine_sim)
    
    # 2. Pearson correlation
    pearson_corr, _ = stats.pearsonr(real_hist, gen_hist)
    metrics['pearson_similarity'] = max(0, pearson_corr)
    
    # 3. Degree distribution shape similarity (quantile-based)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    real_quantiles = np.quantile(real_degrees, quantiles)
    gen_quantiles = np.quantile(gen_degrees, quantiles)
    
    # Quantile similarity
    quantile_similarity = 1 - np.mean(np.abs(real_quantiles - gen_quantiles)) / (np.mean(real_quantiles) + 1e-10)
    metrics['quantile_similarity'] = max(0, quantile_similarity)
    
    # 4. Degree distribution skewness similarity
    real_skewness = stats.skew(real_degrees)
    gen_skewness = stats.skew(gen_degrees)
    max_skew = max(abs(real_skewness), abs(gen_skewness), 1e-6)
    skewness_similarity = 1 - abs(real_skewness - gen_skewness) / max_skew
    metrics['skewness_similarity'] = max(0, skewness_similarity)
    
    # 5. Degree distribution kurtosis similarity
    real_kurtosis = stats.kurtosis(real_degrees)
    gen_kurtosis = stats.kurtosis(gen_degrees)
    max_kurt = max(abs(real_kurtosis), abs(gen_kurtosis), 1e-6)
    kurtosis_similarity = 1 - abs(real_kurtosis - gen_kurtosis) / max_kurt
    metrics['kurtosis_similarity'] = max(0, kurtosis_similarity)
    
    # 6. Degree distribution variance similarity
    real_var = np.var(real_degrees)
    gen_var = np.var(gen_degrees)
    max_var = max(real_var, gen_var, 1e-10)
    variance_similarity = 1 - abs(real_var - gen_var) / max_var
    metrics['variance_similarity'] = max(0, variance_similarity)
    
    # 7. Degree distribution range similarity
    real_range = np.max(real_degrees) - np.min(real_degrees)
    gen_range = np.max(gen_degrees) - np.min(gen_degrees)
    max_range = max(real_range, gen_range, 1e-10)
    range_similarity = 1 - abs(real_range - gen_range) / max_range
    metrics['range_similarity'] = max(0, range_similarity)
    
    # 8. Degree distribution median similarity
    real_median = np.median(real_degrees)
    gen_median = np.median(gen_degrees)
    max_median = max(real_median, gen_median, 1e-10)
    median_similarity = 1 - abs(real_median - gen_median) / max_median
    metrics['median_similarity'] = max(0, median_similarity)
    
    # 9. Degree distribution mean similarity
    real_mean = np.mean(real_degrees)
    gen_mean = np.mean(gen_degrees)
    max_mean = max(real_mean, gen_mean, 1e-10)
    mean_similarity = 1 - abs(real_mean - gen_mean) / max_mean
    metrics['mean_similarity'] = max(0, mean_similarity)
    
    # 10. Degree distribution std similarity
    real_std = np.std(real_degrees)
    gen_std = np.std(gen_degrees)
    max_std = max(real_std, gen_std, 1e-10)
    std_similarity = 1 - abs(real_std - gen_std) / max_std
    metrics['std_similarity'] = max(0, std_similarity)
    
    return metrics

def query_neuron_connections_with_retry(client, root_id, max_retries=3, timeout=30):
    """Query synapse partners for a neuron root ID with retry/backoff.
    
    Parameters
    ----------
    client : CAVEclient
        Initialized CAVE client.
    root_id : int
        Neuron root ID to query.
    max_retries : int, optional
        Maximum number of attempts.
    timeout : int, optional
        Reserved parameter (currently unused).
    
    Returns
    -------
    set
        Set of connected neuron IDs (both pre- and post-synaptic partners).
    """
    for attempt in range(max_retries):
        try:
            # Query connections as post-synaptic
            df_post = client.materialize.synapse_query(post_ids=root_id)
            pre_ids = set(df_post.iloc[:, 6])  # Column 7 is pre-synaptic neuron ID

            # Query connections as pre-synaptic
            df_pre = client.materialize.synapse_query(pre_ids=root_id)
            post_ids = set(df_pre.iloc[:, 8])  # Column 9 is post-synaptic neuron ID

            # Merge all connections
            all_conn_ids = pre_ids.union(post_ids)
            return all_conn_ids
            
        except Exception as e:
            print(f"Error querying neuron {root_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Neuron {root_id} query failed, skipping")
                return set()

def save_connection_progress(id_list, processed_indices, conn_matrix, save_file):
    """Save incremental connection-query progress to a NumPy file.
    
    Parameters
    ----------
    id_list : list
        List of neuron IDs in the working set.
    processed_indices : list[int]
        Indices in ``id_list`` that have been processed.
    conn_matrix : array-like
        Current (partial) adjacency matrix.
    save_file : str
        Output ``.npy`` path.
    """
    progress_data = {
        'processed_indices': processed_indices,
        'connection_matrix': cp.asnumpy(conn_matrix) if isinstance(conn_matrix, cp.ndarray) else conn_matrix,
        'timestamp': time.time()
    }
    np.save(save_file, progress_data)
    print(f"Progress saved to {save_file}")

def load_connection_progress(save_file):
    """Load connection-query progress if a checkpoint file exists.
    
    Parameters
    ----------
    save_file : str
        Path to a saved ``.npy`` checkpoint.
    
    Returns
    -------
    processed_indices : list[int]
        Indices already processed.
    connection_matrix : numpy.ndarray or None
        Saved adjacency matrix, or ``None`` if missing.
    """
    if os.path.exists(save_file):
        progress_data = np.load(save_file, allow_pickle=True).item()
        return progress_data['processed_indices'], progress_data['connection_matrix']
    return [], None

def plot_all_ave_deg_boxplot(all_results, save_path):
    """Plot entropy distributions across ``ave_deg`` values and save outputs.
    
    Parameters
    ----------
    all_results : dict
        Mapping ``ave_deg`` -> dict(alpha -> entropy).
    save_path : str
        Path to save the PNG. A companion ``.txt`` file is also written.
    """
    # Get all alpha values (assume string keys per ave_deg)
    alpha_values = sorted([float(a) for a in next(iter(all_results.values())).keys()])
    entropy_data = []
    for alpha in alpha_values:
        alpha_str = str(alpha)
        alpha_entropies = []
        for ave_deg, results in all_results.items():
            if alpha_str in results:
                alpha_entropies.append(results[alpha_str])
        entropy_data.append(alpha_entropies)
    plt.figure(figsize=(15, 8))
    bp = plt.boxplot(entropy_data, 
                    labels=[f"{a:.2f}" for a in alpha_values],
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5),
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.xlabel('Alpha', fontsize=12, labelpad=10)
    plt.ylabel('Information Entropy', fontsize=12)
    plt.title('Information Entropy Distribution Across Different ave_deg Values', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save corresponding txt file
    txt_path = save_path.replace('.png', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Entropy distribution data for all ave_deg values\n")
        f.write("=" * 50 + "\n\n")
        f.write("Data description:\n")
        f.write("- Rows: different alpha values\n")
        f.write("- Columns: list of entropy values for all ave_deg per alpha\n")
        f.write("- Each alpha corresponds to one ave_deg list (entropy under that alpha)\n\n")
        f.write("Data content:\n")
        f.write("-" * 50 + "\n")
        for i, alpha in enumerate(alpha_values):
            f.write(f"Alpha={alpha:.2f}:\n")
            f.write(f"  Entropy value count: {len(entropy_data[i])}\n")
            f.write(f"  Entropy value list: {entropy_data[i]}\n")
            if len(entropy_data[i]) > 0:
                f.write(f"  Min: {min(entropy_data[i]):.6f}\n")
                f.write(f"  Max: {max(entropy_data[i]):.6f}\n")
                f.write(f"  Mean: {np.mean(entropy_data[i]):.6f}\n")
                f.write(f"  Median: {np.median(entropy_data[i]):.6f}\n")
            f.write("\n")

def plot_alpha_curve(alpha_values, entropy_values, ave_deg, save_path):
    """Plot entropy versus alpha for a fixed ``ave_deg`` and save outputs.
    
    Parameters
    ----------
    alpha_values : array-like
        Alpha grid.
    entropy_values : array-like
        Entropy values aligned with ``alpha_values``.
    ave_deg : float
        Target average degree label used in the figure title.
    save_path : str
        Path to save the PNG. A companion ``.txt`` file is also written.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, entropy_values, 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('Information Entropy')
    plt.title(f'Information Entropy vs Alpha (ave_deg={ave_deg})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    # Save corresponding txt file
    txt_path = save_path.replace('.png', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Entropy vs alpha data for ave_deg={ave_deg}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Data description:\n")
        f.write("- Column 1: Alpha value\n")
        f.write("- Column 2: Corresponding entropy value\n")
        f.write("- Row count: number of data points\n\n")
        f.write("Data content:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Alpha':<10} {'Information Entropy':<20}\n")
        f.write("-" * 50 + "\n")
        for alpha, entropy in zip(alpha_values, entropy_values):
            f.write(f"{alpha:<10.2f} {entropy:<20.6f}\n")

def create_comprehensive_network_analysis(real_metrics, gen_metrics_list, ks_results, save_dir):
    """Create and save a set of diagnostic figures comparing network metrics.
    
    Parameters
    ----------
    real_metrics : dict
        Metrics dictionary for the real network.
    gen_metrics_list : list[dict]
        Metrics dictionaries for generated networks (replicates).
    ks_results : dict
        KS-test results (e.g., clustering and triangle distributions).
    save_dir : str
        Output directory for PNG and TXT files.
    """
    
    # Set global font and style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': False,
        'figure.dpi': 300
    })
    
    # Compute mean metrics of generated networks
    gen_avg_degree = np.mean([m['avg_degree'] for m in gen_metrics_list])
    gen_avg_clustering = np.mean([m['avg_clustering'] for m in gen_metrics_list])
    gen_network_density = np.mean([m['network_density'] for m in gen_metrics_list])
    
    # 1. Degree distribution comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_degree_distribution_comparison(ax1, real_metrics, gen_metrics_list, f'{save_dir}/degree_distribution_comparison.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/degree_distribution_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Clustering coefficient distribution comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_clustering_distribution_comparison(ax2, real_metrics, gen_metrics_list, f'{save_dir}/clustering_distribution_comparison.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clustering_distribution_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Network metrics radar chart
    fig3, ax3 = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
    plot_network_metrics_radar(ax3, real_metrics, gen_avg_degree, gen_avg_clustering, gen_network_density, f'{save_dir}/network_metrics_radar.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/network_metrics_radar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. KS test results heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    plot_ks_test_heatmap(ax4, ks_results, f'{save_dir}/ks_test_heatmap.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ks_test_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 5. Degree distribution similarity metrics
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    plot_degree_distribution_metrics(ax5, getattr(main, 'degree_metrics_list', []), f'{save_dir}/degree_similarity_metrics.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/degree_similarity_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 6. Distribution distance metrics comparison
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    plot_distribution_distance_metrics(ax6, real_metrics, gen_metrics_list, f'{save_dir}/distribution_distance_metrics.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distribution_distance_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved 6 subplots to: {save_dir}")

def plot_degree_distribution_comparison(ax, real_metrics, gen_metrics_list, txt_path=None):
    """Plot degree distribution histograms for real vs generated networks.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    real_metrics : dict
        Real-network metrics (expects ``degree_distribution``).
    gen_metrics_list : list[dict]
        Generated-network metrics (expects ``degree_distribution``).
    txt_path : str or None, optional
        If provided, write the binned density values to this path.
    """
    real_degrees = real_metrics['degree_distribution']
    gen_degrees_list = [m['degree_distribution'] for m in gen_metrics_list]
    gen_degrees_mean = np.mean(gen_degrees_list, axis=0)
    
    # Create histogram
    bins = np.linspace(0, max(np.max(real_degrees), np.max(gen_degrees_mean)) + 1, 50)

    # Compute histogram data
    real_hist, real_bin_edges = np.histogram(real_degrees, bins=bins, density=True)
    gen_hist, gen_bin_edges = np.histogram(gen_degrees_mean, bins=bins, density=True)
    
    # Real network
    ax.hist(real_degrees, bins=bins, alpha=0.7, density=True, 
            color='#2E86AB', label='Real Network', edgecolor='black', linewidth=0.5)
    
    # Generated network
    ax.hist(gen_degrees_mean, bins=bins, alpha=0.7, density=True,
            color='#A23B72', label='Generated Network', edgecolor='black', linewidth=0.5)
    
    # Add KS test info
    ks_stat = stats.ks_2samp(real_degrees, gen_degrees_mean)[0]
    ax.text(0.7, 0.8, f'KS statistic: {ks_stat:.3f}', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('Node Degree (Number of Connections)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 50)  # X-axis range 0-50
    ax.set_title('(A) Degree Distribution Comparison\n'
                'Shows how node connections are distributed across the network\n'
                '• Real network: More heterogeneous, wider degree range\n'
                '• Generated network: More homogeneous, narrower degree range', 
                fontweight='bold', pad=20, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save corresponding txt file
    if txt_path:
        bin_centers = (bins[:-1] + bins[1:]) / 2
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Degree distribution comparison data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Bin center (Node Degree)\n")
            f.write("- Column 2: Real network probability density\n")
            f.write("- Column 3: Generated network probability density\n")
            f.write("- Row count: 50 histogram bins\n\n")
            f.write(f"KS statistic: {ks_stat:.6f}\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Degree':<15} {'Real Density':<20} {'Generated Density':<20}\n")
            f.write("-" * 50 + "\n")
            for i in range(len(bin_centers)):
                f.write(f"{bin_centers[i]:<15.2f} {real_hist[i]:<20.6f} {gen_hist[i]:<20.6f}\n")

def plot_clustering_distribution_comparison(ax, real_metrics, gen_metrics_list, txt_path=None):
    """Plot clustering-coefficient distributions (KDE) for real vs generated networks.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    real_metrics : dict
        Real-network metrics (expects ``clustering_coeffs``).
    gen_metrics_list : list[dict]
        Generated-network metrics (expects ``clustering_coeffs``).
    txt_path : str or None, optional
        If provided, write the evaluated KDE curves to this path.
    
    Notes
    -----
    Distributions are computed on nodes with clustering coefficient > 0 to match downstream analyses.
    """
    real_clustering = real_metrics['clustering_coeffs']
    gen_clustering_list = [m['clustering_coeffs'] for m in gen_metrics_list]
    gen_clustering_mean = np.mean(gen_clustering_list, axis=0)
    
    # Keep original data for txt export
    real_clustering_original = real_clustering.copy()
    gen_clustering_original = gen_clustering_mean.copy()

    # Filter out zeros
    real_clustering = real_clustering[real_clustering > 0]
    gen_clustering_mean = gen_clustering_mean[gen_clustering_mean > 0]
    
    if len(real_clustering) > 0 and len(gen_clustering_mean) > 0:
        # Use KDE for smooth distribution
        from scipy.stats import gaussian_kde
        
        real_kde = gaussian_kde(real_clustering)
        gen_kde = gaussian_kde(gen_clustering_mean)
        
        x_range = np.linspace(0, 1, 200)
        real_kde_values = real_kde(x_range)
        gen_kde_values = gen_kde(x_range)
        
        ax.plot(x_range, real_kde_values, color='#2E86AB', linewidth=3, 
                label='Real Network', alpha=0.8)
        ax.fill_between(x_range, real_kde_values, alpha=0.3, color='#2E86AB')
        
        ax.plot(x_range, gen_kde_values, color='#A23B72', linewidth=3, 
                label='Generated Network', alpha=0.8)
        ax.fill_between(x_range, gen_kde_values, alpha=0.3, color='#A23B72')
    else:
        x_range = np.linspace(0, 1, 200)
        real_kde_values = np.zeros(200)
        gen_kde_values = np.zeros(200)
    
    ax.set_xlabel('Clustering Coefficient (0-1)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 1)  # X-axis range 0-1
    ax.set_title('(B) Clustering Coefficient Distribution\n'
                'Measures local clustering tendency of nodes\n'
                '• Higher values: More triangular connections in neighborhoods\n'
                '• Lower values: More tree-like local structures', 
                fontweight='bold', pad=20, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save corresponding txt file
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Clustering coefficient distribution comparison data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Clustering coefficient (0-1)\n")
            f.write("- Column 2: Real network KDE density\n")
            f.write("- Column 3: Generated network KDE density\n")
            f.write("- Row count: 200 KDE evaluation points\n\n")
            f.write(f"Real network valid nodes (>0): {len(real_clustering)}\n")
            f.write(f"Generated network valid nodes (>0): {len(gen_clustering_mean)}\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Clustering Coeff':<20} {'Real KDE Density':<20} {'Generated KDE Density':<20}\n")
            f.write("-" * 50 + "\n")
            for i in range(len(x_range)):
                f.write(f"{x_range[i]:<20.6f} {real_kde_values[i]:<20.6f} {gen_kde_values[i]:<20.6f}\n")

def plot_network_metrics_radar(ax, real_metrics, gen_avg_degree, gen_avg_clustering, gen_network_density, txt_path=None):
    """Plot a radar chart comparing key scalar network metrics.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes.
    real_metrics : dict
        Real-network scalar metrics (avg_degree, avg_clustering, network_density).
    gen_avg_degree, gen_avg_clustering, gen_network_density : float
        Generated-network scalar averages.
    txt_path : str or None, optional
        If provided, write raw and normalized values to this path.
    """
    metrics = ['Average\nDegree', 'Clustering\nCoefficient', 'Network\nDensity']
    
    # Keep raw values for export
    real_raw_values = [
        real_metrics['avg_degree'],
        real_metrics['avg_clustering'],
        real_metrics['network_density']
    ]
    
    gen_raw_values = [
        gen_avg_degree,
        gen_avg_clustering,
        gen_network_density
    ]
    
    # Normalize metrics to 0-1 for display
    real_values = [
        min(real_metrics['avg_degree'] / 10, 1),
        min(real_metrics['avg_clustering'] * 1000, 1),
        min(real_metrics['network_density'] * 10000, 1)
    ]
    
    gen_values = [
        min(gen_avg_degree / 10, 1),
        min(gen_avg_clustering * 1000, 1),
        min(gen_network_density * 10000, 1)
    ]
    
    # Set angles
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    real_values += real_values[:1]  # Close polygon
    gen_values += gen_values[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, real_values, 'o-', linewidth=3, color='#2E86AB', label='Real', markersize=6)
    ax.fill(angles, real_values, alpha=0.25, color='#2E86AB')
    
    ax.plot(angles, gen_values, 'o-', linewidth=3, color='#A23B72', label='Generated', markersize=6)
    ax.fill(angles, gen_values, alpha=0.25, color='#A23B72')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title('(C) Network Metrics Radar Chart\n'
                'Multi-dimensional comparison of key properties\n'
                '• Outer edge: Higher values\n'
                '• Inner area: Lower values', 
                fontweight='bold', pad=30, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    # Save corresponding txt file
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Network metrics radar chart data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Metric name\n")
            f.write("- Column 2: Real network raw value\n")
            f.write("- Column 3: Real network normalized value (0-1)\n")
            f.write("- Column 4: Generated network raw value\n")
            f.write("- Column 5: Generated network normalized value (0-1)\n")
            f.write("- Row count: 3 metrics (average degree, clustering coefficient, network density)\n\n")
            f.write("Normalization: degree/10, clustering*1000, density*10000, then min(..., 1)\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric':<25} {'Real Raw':<15} {'Real Norm':<15} {'Gen Raw':<15} {'Gen Norm':<15}\n")
            f.write("-" * 50 + "\n")
            metric_names = ['Average Degree', 'Clustering Coefficient', 'Network Density']
            for i, name in enumerate(metric_names):
                f.write(f"{name:<25} {real_raw_values[i]:<15.6f} {real_values[i]:<15.6f} {gen_raw_values[i]:<15.6f} {gen_values[i]:<15.6f}\n")

def plot_ks_test_heatmap(ax, ks_results, txt_path=None):
    """Plot a heatmap summarizing KS statistics and p-values.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    ks_results : dict
        KS results with keys like ``clustering_coeffs`` and ``triangles`` (each with statistic/p_value).
    txt_path : str or None, optional
        If provided, write the displayed values to this path.
    """
    metrics = ['Clustering\nDistribution', 'Triangle\nDistribution']
    ks_stats = [ks_results['clustering_coeffs']['statistic'],
                ks_results['triangles']['statistic']]
    p_values = [ks_results['clustering_coeffs']['p_value'],
                ks_results['triangles']['p_value']]
    
    # Build data matrix
    data = np.array([ks_stats, p_values]).T

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')

    # Add value labels
    for i in range(len(metrics)):
        for j in range(2):
            if j == 1:  # P-value column: scientific notation
                if data[i, j] < 0.001:
                    text = f'{data[i, j]:.2e}'
                else:
                    text = f'{data[i, j]:.4f}'
            else:  # KS statistic column
                text = f'{data[i, j]:.4f}'
            ax.text(j, i, text, ha="center", va="center", color="black", fontweight='bold', fontsize=10)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['KS Statistic\n(Smaller=More Similar)', 'P-value\n(Smaller=More Different)'], fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_title('(D) Kolmogorov-Smirnov Test Results\n'
                'Statistical significance of distribution differences\n'
                '• KS statistic: 0=identical, 1=completely different\n'
                '• P-value: <0.05=significantly different\n'
                '• Note: Degree distribution uses alternative metrics', 
                fontweight='bold', pad=20, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value', rotation=270, labelpad=20, fontsize=10)
    
    # Save corresponding txt file
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("KS test results data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Distribution type\n")
            f.write("- Column 2: KS statistic (0=identical, 1=fully different)\n")
            f.write("- Column 3: P-value (<0.05 indicates significant difference)\n")
            f.write("- Row count: 2 types (clustering distribution, triangle distribution)\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Distribution Type':<30} {'KS Statistic':<20} {'P-value':<20}\n")
            f.write("-" * 50 + "\n")
            metric_names = ['Clustering Distribution', 'Triangle Distribution']
            for i, name in enumerate(metric_names):
                p_val_str = f'{p_values[i]:.2e}' if p_values[i] < 0.001 else f'{p_values[i]:.6f}'
                f.write(f"{name:<30} {ks_stats[i]:<20.6f} {p_val_str:<20}\n")

def plot_similarity_metrics(ax, real_metrics, gen_avg_degree, gen_avg_clustering, gen_network_density):
    """Plot a bar chart of scalar similarity scores (degree/clustering/density).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    real_metrics : dict
        Real-network scalar metrics.
    gen_avg_degree, gen_avg_clustering, gen_network_density : float
        Generated-network scalar averages.
    """
    # Compute similarity
    degree_sim = 1 - abs(real_metrics['avg_degree'] - gen_avg_degree) / max(real_metrics['avg_degree'], gen_avg_degree)
    clustering_sim = 1 - abs(real_metrics['avg_clustering'] - gen_avg_clustering) / max(real_metrics['avg_clustering'], gen_avg_clustering)
    density_sim = 1 - abs(real_metrics['network_density'] - gen_network_density) / max(real_metrics['network_density'], gen_network_density)
    
    metrics = ['Degree\nSimilarity', 'Clustering\nSimilarity', 'Density\nSimilarity']
    similarities = [degree_sim, clustering_sim, density_sim]
    
    colors = ['#FF6B6B' if s < 0.5 else '#4ECDC4' if s < 0.8 else '#45B7D1' for s in similarities]
    
    bars = ax.bar(metrics, similarities, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Similarity Score (0-1)', fontweight='bold', fontsize=12)
    ax.set_title('(E) Network Similarity Metrics\n'
                'Quantifies structural resemblance between networks\n'
                '• 0.8-1.0: Excellent similarity\n'
                '• 0.5-0.8: Moderate similarity\n'
                '• 0.0-0.5: Poor similarity', 
                fontweight='bold', pad=20, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add threshold lines
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Similarity')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Similarity')
    ax.legend(fontsize=9)

def plot_degree_distribution_metrics(ax, degree_metrics_list, txt_path=None):
    """Plot aggregated degree-distribution similarity metrics across runs.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    degree_metrics_list : list[dict]
        Per-run degree similarity metric dictionaries.
    txt_path : str or None, optional
        If provided, write the averaged metric values to this path.
    """
    if not degree_metrics_list:
        ax.text(0.5, 0.5, 'No degree metrics available', ha='center', va='center', transform=ax.transAxes)
        if txt_path:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("Degree distribution similarity metrics data\n")
                f.write("=" * 50 + "\n\n")
                f.write("Data description: No data available\n")
        return

    # Compute average metrics
    avg_metrics = {}
    for key in degree_metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in degree_metrics_list])
    
    # Prepare data: select effective metrics
    metric_names = ['Cosine\nSimilarity', 'Pearson\nCorrelation', 'Mean\nSimilarity',
                   'Skewness\nSimilarity', 'Kurtosis\nSimilarity']
    metric_values = [avg_metrics['cosine_similarity'], avg_metrics['pearson_similarity'], 
                    avg_metrics['mean_similarity'], avg_metrics['skewness_similarity'], 
                    avg_metrics['kurtosis_similarity']]
    
    colors = []
    for v in metric_values:
        if v >= 0.8:
            colors.append('#2E8B57') 
        elif v >= 0.6:
            colors.append('#4682B4')  
        elif v >= 0.4:
            colors.append('#DAA520') 
        else:
            colors.append('#DC143C')
    
    # Plot bar chart
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=0.8, capsize=3)
    
    # Add value labels
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='black')
    
    # Add overall similarity line
    comprehensive_similarity = np.mean(metric_values)
    ax.axhline(y=comprehensive_similarity, color='#8B0000', linestyle='--', 
               alpha=0.8, linewidth=2.5, zorder=10)
    ax.text(len(metric_names)-0.3, comprehensive_similarity + 0.03, 
            f'Overall: {comprehensive_similarity:.3f}', ha='right', va='bottom', 
            fontweight='bold', color='#8B0000', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set axis labels
    ax.set_ylabel('Similarity Score (0-1)', fontweight='bold', fontsize=13, labelpad=15)
    ax.set_xlabel('Similarity Metrics', fontweight='bold', fontsize=13, labelpad=15)
    
    # Set title
    ax.set_title('(E) Degree Distribution Similarity Metrics\n'
                'Comprehensive evaluation of degree distribution characteristics\n'
                '• Cosine: Vector similarity of distribution shapes\n'
                '• Pearson: Linear correlation between distributions\n'
                '• Mean: Central tendency similarity\n'
                '• Skewness: Asymmetry similarity\n'
                '• Kurtosis: Tail heaviness similarity', 
                fontweight='bold', pad=25, fontsize=12)
    
    # Set y-axis range
    ax.set_ylim(0, 1.15)
    
    # Set grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add threshold lines
    ax.axhline(y=0.8, color='#228B22', linestyle=':', alpha=0.7, linewidth=2, label='Excellent (≥0.8)')
    ax.axhline(y=0.6, color='#FF8C00', linestyle=':', alpha=0.7, linewidth=2, label='Good (≥0.6)')
    ax.axhline(y=0.4, color='#FF4500', linestyle=':', alpha=0.7, linewidth=2, label='Moderate (≥0.4)')
    
    # Set legend
    ax.legend(loc='upper right', fontsize=10, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # Set tick labels
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    ax.tick_params(axis='y', labelsize=11)
    
    # Set x-axis label alignment
    for label in ax.get_xticklabels():
        label.set_ha('right')
    
    # Set spine style
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Save corresponding txt file
    if txt_path:
        comprehensive_similarity = np.mean(metric_values)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Degree Distribution Similarity Metrics Data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Similarity metric name\n")
            f.write("- Column 2: Similarity score (0-1 range)\n")
            f.write("- Row count: 5 similarity metrics\n\n")
            f.write("Metric description:\n")
            f.write("- Cosine Similarity: Histogram-based cosine similarity\n")
            f.write("- Pearson Correlation: Linear correlation between distributions\n")
            f.write("- Mean Similarity: Central tendency similarity\n")
            f.write("- Skewness Similarity: Distribution asymmetry similarity\n")
            f.write("- Kurtosis Similarity: Distribution tail heaviness similarity\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric Name':<30} {'Similarity Score':<20}\n")
            f.write("-" * 50 + "\n")
            metric_names_full = ['Cosine Similarity', 'Pearson Correlation', 'Mean Similarity',
                                'Skewness Similarity', 'Kurtosis Similarity']
            for name, value in zip(metric_names_full, metric_values):
                f.write(f"{name:<30} {value:<20.6f}\n")
            f.write(f"\nOverall Similarity: {comprehensive_similarity:.6f}\n")
            f.write(f"Mean of all metrics\n")

def plot_distribution_distance_metrics(ax, real_metrics, gen_metrics_list, txt_path=None):
    """Plot distribution-distance similarities for degree/clustering/triangles.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    real_metrics : dict
        Real-network metrics (degree_distribution, clustering_coeffs, triangles).
    gen_metrics_list : list[dict]
        Generated-network metrics (same fields as ``real_metrics``).
    txt_path : str or None, optional
        If provided, write the computed metric values to this path.
    """
    # Compute mean metrics of generated networks
    gen_avg_degree = np.mean([m['avg_degree'] for m in gen_metrics_list])
    gen_avg_clustering = np.mean([m['avg_clustering'] for m in gen_metrics_list])
    gen_network_density = np.mean([m['network_density'] for m in gen_metrics_list])
    
    # Compute multiple distribution distance metrics
    # 1. Bray-Curtis dissimilarity (convert to similarity)
    degree_bc = calculate_bray_curtis_dissimilarity(real_metrics['degree_distribution'], 
                                                   np.mean([m['degree_distribution'] for m in gen_metrics_list], axis=0))
    # Use only C>0 for clustering coefficient
    real_clu_filtered = real_metrics['clustering_coeffs'][real_metrics['clustering_coeffs'] > 0]
    gen_clu_filtered = np.mean([m['clustering_coeffs'] for m in gen_metrics_list], axis=0)
    gen_clu_filtered = gen_clu_filtered[gen_clu_filtered > 0]
    if len(real_clu_filtered) > 0 and len(gen_clu_filtered) > 0:
        clustering_bc = calculate_bray_curtis_dissimilarity(real_clu_filtered, gen_clu_filtered, value_range=(0.0, 1.0), n_bins=100)
    else:
        clustering_bc = np.nan
    triangle_bc = calculate_bray_curtis_dissimilarity(real_metrics['triangles'], 
                                                     np.mean([m['triangles'] for m in gen_metrics_list], axis=0))
    
    # 2. JS divergence (convert to similarity)
    degree_js = calculate_js_divergence(real_metrics['degree_distribution'], 
                                       np.mean([m['degree_distribution'] for m in gen_metrics_list], axis=0))
    # Use only C>0 for clustering coefficient
    if len(real_clu_filtered) > 0 and len(gen_clu_filtered) > 0:
        clustering_js = calculate_js_divergence(real_clu_filtered, gen_clu_filtered, value_range=(0.0, 1.0), n_bins=100)
    else:
        clustering_js = np.nan
    triangle_js = calculate_js_divergence(real_metrics['triangles'], 
                                         np.mean([m['triangles'] for m in gen_metrics_list], axis=0))
    
    # Convert to similarity score (smaller -> larger is better)
    degree_bc_sim = 1 - degree_bc
    clustering_bc_sim = 1 - clustering_bc
    triangle_bc_sim = 1 - triangle_bc
    
    # JS divergence to similarity
    degree_js_sim = 1 - degree_js
    clustering_js_sim = 1 - clustering_js
    triangle_js_sim = 1 - triangle_js
    
    # Grouped bars; compare two metrics
    metrics = ['Degree\nDistribution', 'Clustering\nDistribution', 'Triangle\nDistribution']
    x = np.arange(len(metrics))
    width = 0.35

    colors = ['#1f77b4', '#ff7f0e']
    # Plot two metrics
    bars1 = ax.bar(x - width/2, [degree_bc_sim, clustering_bc_sim, triangle_bc_sim], 
                   width, label='Bray-Curtis', alpha=0.85, color=colors[0], 
                   edgecolor='black', linewidth=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, [degree_js_sim, clustering_js_sim, triangle_js_sim], 
                   width, label='JS Divergence', alpha=0.85, color=colors[1], 
                   edgecolor='black', linewidth=0.8, capsize=3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9, color='black')
    
    # Set axis labels
    ax.set_ylabel('Similarity Score (0-1)', fontweight='bold', fontsize=13, labelpad=15)
    ax.set_xlabel('Distribution Types', fontweight='bold', fontsize=13, labelpad=15)
    
    # Set title
    ax.set_title('(F) Distribution Distance Metrics\n'
                'Multiple measures of distribution similarity\n'
                '• Bray-Curtis: Compositional similarity\n'
                '• JS Divergence: Information-theoretic similarity\n'
                '• Higher values = More similar distributions', 
                fontweight='bold', pad=25, fontsize=12)
    
    # Set ticks
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    
    # Set grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set legend
    ax.legend(loc='upper right', fontsize=11, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9, 
              bbox_to_anchor=(1.0, 1.0))
    
    # Set tick labels
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    # Set spine style
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Add threshold lines
    ax.axhline(y=0.8, color='#228B22', linestyle=':', alpha=0.7, linewidth=2, label='Excellent (≥0.8)')
    ax.axhline(y=0.6, color='#FF8C00', linestyle=':', alpha=0.7, linewidth=2, label='Good (≥0.6)')
    ax.axhline(y=0.4, color='#FF4500', linestyle=':', alpha=0.7, linewidth=2, label='Moderate (≥0.4)')
    
    # Save corresponding txt file
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Distribution Distance Metrics Comparison Data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Distribution type\n")
            f.write("- Column 2: Bray-Curtis similarity (1 - Bray-Curtis dissimilarity, 0-1 range)\n")
            f.write("- Column 3: JS divergence similarity (1 - JS divergence, 0-1 range)\n")
            f.write("- Row count: 3 distribution types (degree, clustering coefficient, triangle)\n\n")
            f.write("Metric description:\n")
            f.write("- Bray-Curtis: Compositional similarity measure\n")
            f.write("- JS Divergence: Information-theoretic similarity based on KL divergence\n")
            f.write("- Higher similarity score means more similar distributions\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Distribution Type':<30} {'Bray-Curtis Sim':<20} {'JS Divergence Sim':<20}\n")
            f.write("-" * 50 + "\n")
            distribution_names = ['Degree Distribution', 'Clustering Distribution', 'Triangle Distribution']
            bc_values = [degree_bc_sim, clustering_bc_sim, triangle_bc_sim]
            js_values = [degree_js_sim, clustering_js_sim, triangle_js_sim]
            for name, bc, js in zip(distribution_names, bc_values, js_values):
                f.write(f"{name:<30} {bc:<20.6f} {js:<20.6f}\n")

def main():
    """Run the end-to-end visual-cortex network analysis pipeline.
    
    Notes
    -----
    This function performs I/O (CSV/API queries), GPU computations, and writes figures/results to disk.
    """
    # Set random seed
    print("Setting random seed...")
    np.random.seed(RANDOM_SEED)
    cp.random.seed(RANDOM_SEED)
    
    # Set up GPU
    print("Initializing GPU...")
    cp.cuda.Device(0).use()
    
    # Create results directory
    if not os.path.exists(visual_cortex_result_dir):
        os.makedirs(visual_cortex_result_dir)
    
    # Check for existing saved real connection matrix
    real_conn_file = os.path.join(visual_cortex_result_dir, 'real_connection_matrix.npy')
    neuron_ids_file = os.path.join(visual_cortex_result_dir, 'neuron_ids.npy')
    id_to_coord_file = os.path.join(visual_cortex_result_dir, 'id_to_coord.npy')
    
    if os.path.exists(real_conn_file) and os.path.exists(neuron_ids_file):
        print("Found saved real connection matrix, loading...")
        real_connection_matrix = np.load(real_conn_file)
        id_list = np.load(neuron_ids_file).tolist()
        id_to_coord = np.load(id_to_coord_file, allow_pickle=True).item()
        id_to_idx = {id_: idx for idx, id_ in enumerate(id_list)}
        n = len(id_list)
        print(f"Loaded real connection matrix for {n} neurons")
    else:
        print("No saved real connection matrix found, computing...")

        # 1. Read synapse CSV, collect first 30000 neuron IDs and coordinates
        print("Reading synapse data...")
        id_set = set()
        id_list = []
        id_to_idx = {}
        id_to_coord = {}

        # Read neuron IDs
        df_ids = pd.read_csv(csv_file, usecols=[6, 11], nrows=200000)
        pre_ids = df_ids.iloc[:, 0].values
        post_ids = df_ids.iloc[:, 1].values

        # Read coordinate data
        df_coords = pd.read_csv(csv_file, usecols=[2,3,4,7,8,9], nrows=200000)
        pre_coords = df_coords.iloc[:, 0:3].values
        post_coords = df_coords.iloc[:, 3:6].values

        # Collect first 30000 distinct neuron IDs and coordinates
        for i in range(len(pre_ids)):
            # Process pre-synaptic neuron
            if pre_ids[i] not in id_set:
                id_set.add(pre_ids[i])
                id_list.append(pre_ids[i])
                id_to_idx[pre_ids[i]] = len(id_list) - 1
                id_to_coord[pre_ids[i]] = pre_coords[i]
                if len(id_list) >= 30000:
                    break
            
            # Process post-synaptic neuron
            if post_ids[i] not in id_set:
                id_set.add(post_ids[i])
                id_list.append(post_ids[i])
                id_to_idx[post_ids[i]] = len(id_list) - 1
                id_to_coord[post_ids[i]] = post_coords[i]
                if len(id_list) >= 30000:
                    break

        print(f"Collected {len(id_list)} neuron IDs")

        # 2. Build connection matrix
        print("Building connection matrix...")
        n = len(id_list)
        real_connection_matrix = np.zeros((n, n), dtype=bool)

        # Check for saved progress
        progress_file = os.path.join(visual_cortex_result_dir, 'connection_progress.npy')
        processed_indices, saved_matrix = load_connection_progress(progress_file)

        if saved_matrix is not None:
            print(f"Found saved progress, {len(processed_indices)} neurons already processed")
            real_connection_matrix = saved_matrix
        else:
            processed_indices = []

        # Request interval to avoid API rate limit
        request_interval = 0.5  # 0.5 s per request

        for i, root_id in enumerate(tqdm(id_list, desc="Querying neuron connections")):
            # Skip already processed neurons
            if i in processed_indices:
                continue

            # Query neuron connections
            all_conn_ids = query_neuron_connections_with_retry(client, root_id)

            # Count only connections within the 30000 neurons
            for conn_id in all_conn_ids:
                if conn_id in id_to_idx:
                    j = id_to_idx[conn_id]
                    if j > i:  # Fill upper triangle only
                        real_connection_matrix[i, j] = 1

            # Record processed index
            processed_indices.append(i)

            # Save progress every 100 neurons
            if len(processed_indices) % 100 == 0:
                save_connection_progress(id_list, processed_indices, real_connection_matrix, progress_file)

            time.sleep(request_interval)

        # 3. Symmetrize matrix
        print("Symmetrizing matrix...")
        real_connection_matrix = np.logical_or(real_connection_matrix, real_connection_matrix.T)

        # 4. Save real connection matrix and related data
        print("Saving real connection matrix...")
        np.save(real_conn_file, real_connection_matrix)
        np.save(neuron_ids_file, np.array(id_list))
        np.save(id_to_coord_file, id_to_coord)
        print("Real connection matrix saved")

    # Compute real average degree
    real_ave_deg = float(cp.sum(real_connection_matrix)) / n
    print(f"Real average degree: {real_ave_deg:.2f}")

    # Step 2: read coordinate data
    print("\nStep 2: reading coordinate data...")
    df_coords = pd.read_csv(csv_file, nrows=200000)
    
    # Initialize coordinate arrays
    print("Computing neuron average coordinates...")
    neuron_coords = np.zeros((n, 3), dtype=np.float32)
    neuron_counts = np.zeros(n, dtype=np.int32)

    # Aggregate coordinates per neuron
    for i in tqdm(range(len(df_coords)), desc="Computing neuron average coordinates"):
        pre_id = df_coords.iloc[i, 6]
        post_id = df_coords.iloc[i, 11]
        
        if pre_id in id_to_idx:
            idx = id_to_idx[pre_id]
            neuron_coords[idx] += df_coords.iloc[i, [2, 3, 4]].values.astype(np.float32)
            neuron_counts[idx] += 1
        
        if post_id in id_to_idx:
            idx = id_to_idx[post_id]
            neuron_coords[idx] += df_coords.iloc[i, [7, 8, 9]].values.astype(np.float32)
            neuron_counts[idx] += 1
    
    # Compute average coordinates
    neuron_coords = neuron_coords / neuron_counts[:, np.newaxis]
    
    # Transfer coordinate data to GPU
    print("Transferring coordinates to GPU...")
    coordinates = cp.array(neuron_coords, dtype=cp.float32)

    # Compute distance matrix
    distances = calculate_distances(coordinates)
    # Keep coordinates for later use
    clear_gpu_memory()
    
    # Experiment parameters
    alpha_values = np.round(np.arange(0, 2.1, 0.1), 2)  # 0 to 2, step 0.1
    ave_deg_values = np.arange(1, 11, 1)  # 1 to 10
    mean_distance = cp.mean(distances)  # Mean distance

    # Check existing results
    existing_results = collect_all_results()
    print(f"\nFound {len(existing_results)} cached ave_deg results")
    
    # Main loop over ave_deg
    for ave_deg in tqdm(ave_deg_values, desc="Processing ave_deg", position=0):
        result_file = os.path.join(results_dir, f"ave_deg_{ave_deg}.txt")
        if os.path.exists(result_file):
            print(f"\nave_deg={ave_deg} result already exists, skipping")
            # Load existing result and plot
            with open(result_file, 'r') as f:
                alpha_results = json.load(f)
        else:
            print(f"\nProcessing ave_deg={ave_deg}...")
            w = mean_distance * ave_deg  # Compute energy constraint
            alpha_results = {}
            for alpha in tqdm(alpha_values, desc="Processing alpha", position=1, leave=False):
                print(f"\nProcessing alpha={alpha:.2f}...")
                
                # Ensure same random seed per run
                np.random.seed(RANDOM_SEED)
                cp.random.seed(RANDOM_SEED)
                
                # Compute connection probability
                prob_matrix = calculate_connection_probabilities(distances, alpha)
                
                # Generate connection matrix
                connection_matrix = generate_connection_matrix(prob_matrix, ave_deg, multiplier=50)
                
                # Apply energy constraint
                connection_matrix = get_energy_bound(connection_matrix, distances, w)

                # Compute (information) entropy
                entropy = calculate_entropy(connection_matrix)
                
                # Save results
                alpha_results[str(alpha)] = entropy
                print(f"alpha={alpha:.2f} done, entropy={entropy}")
                
                # Clear connection matrix
                del connection_matrix, prob_matrix
                clear_gpu_memory()
            
            # Save all results for current ave_deg
            save_ave_deg_results(ave_deg, alpha_results)

        alpha_list = sorted([float(a) for a in alpha_results.keys()])
        entropy_list = [alpha_results[str(a)] for a in alpha_list]
        plot_alpha_curve(
            alpha_list,
            entropy_list,
            ave_deg,
            os.path.join(results_dir, f"alpha_curve_ave_deg_{ave_deg}.png")
        )
    
    # Collect all results (new and cached)
    all_results = collect_all_results()
    
    # Test with real ave_deg
    print(f"\nTesting with real ave_deg={real_ave_deg:.2f}...")
    generate_ave_deg = real_ave_deg
    
    # Check if real ave_deg already computed
    if int(generate_ave_deg) in existing_results:
        print(f"Real ave_deg={int(generate_ave_deg)} result already exists, using cached")
        real_alpha_results = existing_results[int(generate_ave_deg)]
    else:
        print(f"Computing results for real ave_deg={real_ave_deg:.2f}...")
        w = mean_distance * generate_ave_deg  # Compute energy constraint
        
        # Store all alpha results for real ave_deg
        real_alpha_results = {}
        
        # Loop over alpha values
        for alpha in tqdm(alpha_values, desc="Processing alpha for real ave_deg", position=1, leave=False):
            print(f"\nProcessing alpha={alpha:.2f}...")
            
            # Ensure same random seed per run
            np.random.seed(RANDOM_SEED)
            cp.random.seed(RANDOM_SEED)
            
            # Compute connection probability
            prob_matrix = calculate_connection_probabilities(distances, alpha)
            
            # Generate connection matrix
            connection_matrix = generate_connection_matrix(prob_matrix, generate_ave_deg, multiplier=50)
            
            # Apply energy constraint
            connection_matrix = get_energy_bound(connection_matrix, distances, w)
            
            # Compute (information) entropy
            entropy = calculate_entropy(connection_matrix)
            
            # Save results
            real_alpha_results[str(alpha)] = entropy
            print(f"alpha={alpha:.2f} done, entropy={entropy}")
            
            # Clear connection matrix
            del connection_matrix, prob_matrix
            clear_gpu_memory()
        
        # Save all results for real ave_deg
        save_ave_deg_results(int(generate_ave_deg), real_alpha_results)
    
    # Select best alpha from real ave_deg results
    best_alpha = max(real_alpha_results.items(), key=lambda x: x[1])[0]
    best_entropy = real_alpha_results[best_alpha]
    print(f"\nBest alpha for real ave_deg: {float(best_alpha):.2f}")
    print(f"Corresponding entropy: {best_entropy}")
    
    # Run 5 validation runs with real ave_deg and best alpha
    print("\nRunning 5 validation runs with real ave_deg and best alpha...")
    w = mean_distance * generate_ave_deg
    
    # Run 5 validation runs
    topology_similarities = []
    gen_metrics_list = []
    clustering_dist_similarities = []
    triangle_dist_similarities = []
    # KS statistic and p-value collection
    degree_ks_stats = []
    degree_p_values = []
    clustering_ks_stats = []
    clustering_p_values = []
    triangle_ks_stats = []
    triangle_p_values = []
    bias = '0.6'
    
    # Precompute real matrix topology metrics
    print("Computing real connection matrix topology metrics...")
    best_alpha = str(float(best_alpha) + float(bias))
    real_topology_metrics = calculate_network_topology_metrics_gpu(real_connection_matrix)
    print("Real matrix topology metrics done")
    
    # Generate three comparison models
    print("\nGenerating comparison models...")
    
    # 1. Economical model
    print("Generating economical connection matrix...")
    economical_connection_matrix = generate_economical_connection_matrix(distances, generate_ave_deg)
    economical_metrics = calculate_network_topology_metrics_gpu(economical_connection_matrix)
    # Move economical model to CPU, free GPU
    economical_connection_matrix = economical_connection_matrix.get()
    print("Economical model done")
    
    # 2. Homophily model
    print("Generating homophily connection matrix...")
    homophily_connection_matrix = generate_homophily_connection_matrix(distances, coordinates, generate_ave_deg)
    homophily_metrics = calculate_network_topology_metrics_gpu(homophily_connection_matrix)
    homophily_connection_matrix = homophily_connection_matrix.get()
    print("Homophily model done")
    
    # 3. Fully random model
    print("Generating fully random connection matrix...")
    random_connection_matrix = generate_random_connection_matrix(n, generate_ave_deg, RANDOM_SEED)
    random_metrics = calculate_network_topology_metrics_gpu(random_connection_matrix)
    random_connection_matrix = random_connection_matrix.get()
    print("Random model done")
    
    # Force clear GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    print("GPU memory cleared, running main model...")
    
    print("\nGenerating final connection matrix...")
    start_time = time.time()
    
    # Ensure same random seed
    np.random.seed(RANDOM_SEED)
    cp.random.seed(RANDOM_SEED)
        
    # Generate connection matrix
    final_prob_matrix = calculate_connection_probabilities(distances, float(best_alpha))
    final_connection_matrix = generate_connection_matrix(final_prob_matrix, generate_ave_deg, multiplier=50)
    final_connection_matrix = get_energy_bound(final_connection_matrix, distances, w)
        
    # Compute network topology metrics
    print("Computing generated matrix topology metrics...")
    gen_topology_metrics = calculate_network_topology_metrics_gpu(final_connection_matrix)
    gen_metrics_list.append(gen_topology_metrics)
        
    # Compute network topology similarity (multiple metrics)
    # 1. Degree distribution similarity (Bray-Curtis dissimilarity)
    degree_bc = calculate_bray_curtis_dissimilarity(real_topology_metrics['degree_distribution'],
                                       gen_topology_metrics['degree_distribution'])
    degree_similarity = 1 - degree_bc  # Convert to similarity
    
    # 2. Clustering coefficient similarity
    clustering_similarity = 1 - abs(real_topology_metrics['avg_clustering'] - gen_topology_metrics['avg_clustering'])
    clustering_similarity = max(0, clustering_similarity)
    
    # 3. Network density similarity
    density_similarity = 1 - abs(real_topology_metrics['network_density'] - gen_topology_metrics['network_density'])
    density_similarity = max(0, density_similarity)
    
    # 4. Overall topology similarity
    topology_similarity = (0.5 * degree_similarity + 0.3 * clustering_similarity + 0.2 * density_similarity)
    topology_similarities.append(topology_similarity)
    
    # 5. Compute MVS metrics
    degree_mvs = calculate_mvs(real_topology_metrics['degree_distribution'], gen_topology_metrics['degree_distribution'])
    clustering_mvs = calculate_mvs(real_topology_metrics['clustering_coeffs'], gen_topology_metrics['clustering_coeffs'])
    triangle_mvs = calculate_mvs(real_topology_metrics['triangles'], gen_topology_metrics['triangles'])
    
    # 6. Compute other distribution distance metrics
    degree_wasserstein = calculate_wasserstein_distance(real_topology_metrics['degree_distribution'], 
                                                       gen_topology_metrics['degree_distribution'])
    degree_js = calculate_js_divergence(real_topology_metrics['degree_distribution'], 
                                      gen_topology_metrics['degree_distribution'])
    
    print(f"Network topology similarity: {topology_similarity:.4f}")
    print(f"  Degree similarity (BC): {degree_similarity:.4f}")
    print(f"  Clustering similarity: {clustering_similarity:.4f}")
    print(f"  Network density similarity: {density_similarity:.4f}")
    print(f"  Degree MVS: {degree_mvs:.4f}")
    print(f"  Clustering MVS: {clustering_mvs:.4f}")
    print(f"  Triangle MVS: {triangle_mvs:.4f}")
    print(f"  Degree Wasserstein: {degree_wasserstein:.4f}")
    print(f"  Degree JS divergence: {degree_js:.4f}")
    
    # Compute multiple degree distribution similarity metrics (alternative to KS test)
    degree_metrics = calculate_degree_distribution_metrics(
        real_topology_metrics['degree_distribution'], 
        gen_topology_metrics['degree_distribution']
    )
    
    print(f"  Degree distribution similarity metrics:")
    print(f"    Quantile similarity: {degree_metrics['quantile_similarity']:.4f}")
    print(f"    Skewness similarity: {degree_metrics['skewness_similarity']:.4f}")
    print(f"    Kurtosis similarity: {degree_metrics['kurtosis_similarity']:.4f}")
    print(f"    Variance similarity: {degree_metrics['variance_similarity']:.4f}")
    print(f"    Range similarity: {degree_metrics['range_similarity']:.4f}")
    print(f"    Median similarity: {degree_metrics['median_similarity']:.4f}")
    
    # Compute overall degree distribution similarity
    degree_comprehensive_similarity = np.mean(list(degree_metrics.values()))
    print(f"    Overall degree similarity: {degree_comprehensive_similarity:.4f}")
    
    # Store degree metrics for later visualization
    if not hasattr(main, 'degree_metrics_list'):
        main.degree_metrics_list = []
    main.degree_metrics_list.append(degree_metrics)
        
    # Compute clustering distribution KS distance (C>0 only, consistent with compute_comprehensive_similarity_for_model)
    real_clu_filtered = real_topology_metrics['clustering_coeffs'][real_topology_metrics['clustering_coeffs'] > 0]
    gen_clu_filtered = gen_topology_metrics['clustering_coeffs'][gen_topology_metrics['clustering_coeffs'] > 0]
    if len(real_clu_filtered) > 0 and len(gen_clu_filtered) > 0:
        ks_clustering, p_clu = stats.ks_2samp(real_clu_filtered, gen_clu_filtered)
    else:
        ks_clustering, p_clu = np.nan, np.nan
    clustering_dist_similarities.append(ks_clustering)
    print(f"  Clustering distribution KS statistic: {ks_clustering:.4f}")
    print(f"  Clustering KS test: statistic={ks_clustering:.4f}, p-value={p_clu:.4e}")
    clustering_ks_stats.append(ks_clustering)
    clustering_p_values.append(p_clu)
    # Compute triangle distribution KS distance
    ks_triangles, p_tri = stats.ks_2samp(real_topology_metrics['triangles'], gen_topology_metrics['triangles'])
    triangle_dist_similarities.append(ks_triangles)
    print(f"  Triangle distribution KS statistic: {ks_triangles:.4f}")
    print(f"  Triangle KS test: statistic={ks_triangles:.4f}, p-value={p_tri:.4e}")
    triangle_ks_stats.append(ks_triangles)
    triangle_p_values.append(p_tri)
        
    end_time = time.time()
    print(f"Connection matrix done, elapsed: {end_time - start_time:.2f} s")
    del final_connection_matrix, final_prob_matrix
    clear_gpu_memory()
    # Compute generated network topology metrics
    gen_avg_degree = gen_topology_metrics['avg_degree']
    gen_avg_clustering = gen_topology_metrics['avg_clustering']
    gen_network_density = gen_topology_metrics['network_density']
    
    # Real network topology metrics
    real_avg_degree = real_topology_metrics['avg_degree']
    real_avg_clustering = real_topology_metrics['avg_clustering']
    real_network_density = real_topology_metrics['network_density']
    
    print("\nReal network metrics:")
    print(f"  Average degree: {real_avg_degree:.4f}")
    print(f"  Mean clustering: {real_avg_clustering:.6f}")
    print(f"  Network density: {real_network_density:.6f}")
    print("Generated network metrics:")
    print(f"  Average degree: {gen_avg_degree:.4f}")
    print(f"  Mean clustering: {gen_avg_clustering:.6f}")
    print(f"  Network density: {gen_network_density:.6f}")

    # Save best parameters and validation results
    params_file = os.path.join(visual_cortex_result_dir, 'visual_cortex_best_parameters.txt')
    with open(params_file, 'w') as f:
        f.write(f"Real average degree: {real_ave_deg:.2f}\n")
        f.write(f"Best alpha for real ave_deg: {float(best_alpha):.2f}\n")
        f.write(f"Entropy at best alpha: {best_entropy}\n")
        f.write(f"Network topology similarity: {topology_similarity:.4f}\n")
        f.write(f"Clustering distribution KS statistic: {ks_clustering:.4f}\n")
        f.write(f"Triangle distribution KS statistic: {ks_triangles:.4f}\n")
        f.write(f"MVS metrics:\n")
        f.write(f"  Degree MVS: {degree_mvs:.4f}\n")
        f.write(f"  Clustering MVS: {clustering_mvs:.4f}\n")
        f.write(f"  Triangle MVS: {triangle_mvs:.4f}\n")
        f.write("\nReal network metrics:\n")
        f.write(f"- Average degree: {real_avg_degree:.4f}\n")
        f.write(f"- Mean clustering: {real_avg_clustering:.6f}\n")
        f.write(f"- Network density: {real_network_density:.6f}\n")
        f.write("\nGenerated network metrics:\n")
        f.write(f"- Average degree: {gen_avg_degree:.4f}\n")
        f.write(f"- Mean clustering: {gen_avg_clustering:.6f}\n")
        f.write(f"- Network density: {gen_network_density:.6f}\n")
        f.write("\nKS test:\n")
        f.write(f"- Clustering distribution: statistic={ks_clustering:.4f}, p-value={p_clu:.4e}\n")
        f.write(f"- Triangle distribution: statistic={ks_triangles:.4f}, p-value={p_tri:.4e}\n")
        f.write(f"\nTopology similarity notes:\n")
        f.write(f"- Degree similarity (BC): Bray-Curtis dissimilarity for degree comparison\n")
        f.write(f"- Clustering similarity: local clustering match\n")
        f.write(f"- Network density similarity: global density match\n")
        f.write(f"- MVS: distribution mean similarity. Similarity in [0,1]; higher = closer to real.\n")
    
    # ========== Ablation experiments ==========
    print("\n" + "="*60)
    print("Running ablation experiments...")
    print("="*60)
    
    # Save original model metrics
    original_model_metrics = gen_topology_metrics.copy()
    original_model_name = "Original Model"
    
    # Store all ablation experiment results
    ablation_results = {
        original_model_name: {
            'metrics': original_model_metrics,
            'alpha': float(best_alpha),
            'use_degree_constraint': True,
            'use_energy_constraint': True
        }
    }
    
    # Ensure distances/coordinates are available (recompute if needed)
    if 'distances' not in locals() or distances is None:
        print("Recomputing distance matrix...")
        if 'coordinates' not in locals() or coordinates is None:
            print("Warning: coordinate data missing, cannot run ablation")
            print("Skipping ablation...")
            distances = None
        else:
            distances = calculate_distances(coordinates)
    
    # Check if distances is available
    if distances is None:
        print("Ablation skipped: distance matrix unavailable")
    else:
        # Ablation 1: degree heavy-tail constraint
        print("\n[Ablation 1] Degree heavy-tail constraint...")
        np.random.seed(RANDOM_SEED)
        cp.random.seed(RANDOM_SEED)
        
        ablation1_prob_matrix = calculate_connection_probabilities(distances, float(best_alpha), use_degree_constraint=False)
        ablation1_connection_matrix = generate_connection_matrix(ablation1_prob_matrix, generate_ave_deg, multiplier=50)
        ablation1_connection_matrix = get_energy_bound(ablation1_connection_matrix, distances, w)
        ablation1_metrics = calculate_network_topology_metrics_gpu(ablation1_connection_matrix)
        ablation_results['Ablation: No Degree Constraint'] = {
            'metrics': ablation1_metrics,
            'alpha': float(best_alpha),
            'use_degree_constraint': False,
            'use_energy_constraint': True
        }
        del ablation1_connection_matrix, ablation1_prob_matrix
        clear_gpu_memory()
        
        # Ablation 2: energy constraint
        print("\n[Ablation 2] Energy constraint...")
        np.random.seed(RANDOM_SEED)
        cp.random.seed(RANDOM_SEED)
        
        ablation2_prob_matrix = calculate_connection_probabilities(distances, float(best_alpha), use_degree_constraint=True)
        ablation2_connection_matrix = generate_connection_matrix(ablation2_prob_matrix, generate_ave_deg, multiplier=50)
        # Skip energy constraint
        ablation2_metrics = calculate_network_topology_metrics_gpu(ablation2_connection_matrix)
        ablation_results['Ablation: No Energy Constraint'] = {
            'metrics': ablation2_metrics,
            'alpha': float(best_alpha),
            'use_degree_constraint': True,
            'use_energy_constraint': False
        }
        del ablation2_connection_matrix, ablation2_prob_matrix
        clear_gpu_memory()
        
        # Ablation 3: entropy constraint
        print("\n[Ablation 3] Entropy constraint...")
        np.random.seed(RANDOM_SEED)
        cp.random.seed(RANDOM_SEED)
        
        ablation3_alpha = 0.1
        ablation3_prob_matrix = calculate_connection_probabilities(distances, ablation3_alpha, use_degree_constraint=True)
        ablation3_connection_matrix = generate_connection_matrix(ablation3_prob_matrix, generate_ave_deg, multiplier=50)
        ablation3_connection_matrix = get_energy_bound(ablation3_connection_matrix, distances, w)
        ablation3_metrics = calculate_network_topology_metrics_gpu(ablation3_connection_matrix)
        ablation_results['Ablation: Alpha=0.1'] = {
            'metrics': ablation3_metrics,
            'alpha': ablation3_alpha,
            'use_degree_constraint': True,
            'use_energy_constraint': True
        }
        del ablation3_connection_matrix, ablation3_prob_matrix
        clear_gpu_memory()
        
        print("\nAblation experiments done.")

        # Plot degree and clustering comparison for each ablation
        print("\nPlotting ablation comparison figures...")
        
        for ablation_name, ablation_result in ablation_results.items():
            if ablation_name == original_model_name:
                continue  # Skip original model, plot ablations only
            
            print(f"Plotting comparison for {ablation_name}...")
            ablation_metrics_list = [ablation_result['metrics']]
            
            safe_name = ablation_name.replace(' ', '_').replace(':', '_').replace('=', '_')
            
            # Plot degree distribution comparison
            fig_deg, ax_deg = plt.subplots(figsize=(10, 6))
            plot_degree_distribution_comparison(ax_deg, real_topology_metrics, ablation_metrics_list, 
                                               os.path.join(visual_cortex_result_dir, f'{safe_name}_degree_distribution_comparison.txt'))
            plt.tight_layout()
            plt.savefig(os.path.join(visual_cortex_result_dir, f'{safe_name}_degree_distribution_comparison.png'), 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Plot clustering coefficient distribution comparison
            fig_clu, ax_clu = plt.subplots(figsize=(10, 6))
            plot_clustering_distribution_comparison(ax_clu, real_topology_metrics, ablation_metrics_list,
                                                   os.path.join(visual_cortex_result_dir, f'{safe_name}_clustering_distribution_comparison.txt'))
            plt.tight_layout()
            plt.savefig(os.path.join(visual_cortex_result_dir, f'{safe_name}_clustering_distribution_comparison.png'), 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        print("Ablation comparison figures saved")

        # Compute similarity metrics for all ablations and save
        print("\nComputing comprehensive ablation similarity metrics...")
        ablation_similarity_all = {}
        for name, result in ablation_results.items():
            ablation_similarity_all[name] = compute_comprehensive_similarity_for_model(real_topology_metrics, result['metrics'])
        
        # Save comprehensive ablation results
        ablation_txt_path = os.path.join(visual_cortex_result_dir, 'comprehensive_ablation_similarity.txt')
        save_comprehensive_ablation_results_txt(ablation_similarity_all, ablation_results, ablation_txt_path)
        print(f"Saved comprehensive ablation results to: {ablation_txt_path}")
    
    # Clear final data
    del distances
    clear_gpu_memory()
    
    print("\nAll processing done.")

    # Collect all results (new and cached)
    all_results = collect_all_results()

    # Plot boxplot for all ave_deg
    boxplot_save_path = os.path.join(results_dir, 'entropy_boxplot_all_ave_deg.png')
    plot_all_ave_deg_boxplot(all_results, boxplot_save_path)
    print(f"Saved all ave_deg boxplot to: {boxplot_save_path}")
    
    # Create advanced visualizations
    print("\nCreating advanced visualizations...")
    try:
        # Prepare KS test results
        ks_results = {
            'clustering_coeffs': {'statistic': ks_clustering, 'p_value': p_clu},
            'triangles': {'statistic': ks_triangles, 'p_value': p_tri}
        }
        
        # Create comprehensive network analysis figure
        create_comprehensive_network_analysis(real_topology_metrics, gen_metrics_list, ks_results, visual_cortex_result_dir)
        print(f"Saved comprehensive network analysis to: {visual_cortex_result_dir}")
        
        # Create three-model comparison figure
        print("\nCreating three-model comparison...")
        try:
            # Use current model average metrics
            current_model_avg = {
                'avg_degree': np.mean([m['avg_degree'] for m in gen_metrics_list]),
                'avg_clustering': np.mean([m['avg_clustering'] for m in gen_metrics_list]),
                'network_density': np.mean([m['network_density'] for m in gen_metrics_list]),
                'degree_distribution': np.mean([m['degree_distribution'] for m in gen_metrics_list], axis=0),
                'clustering_coeffs': np.mean([m['clustering_coeffs'] for m in gen_metrics_list], axis=0),
                'triangles': np.mean([m['triangles'] for m in gen_metrics_list], axis=0)
            }
            
            create_model_comparison_analysis(real_topology_metrics, current_model_avg, economical_metrics, homophily_metrics, random_metrics, visual_cortex_result_dir)
            print(f"Saved three-model comparison to: {visual_cortex_result_dir}")
            
        except Exception as e2:
            print(f"Error creating three-model comparison: {e2}")
            print("Skipping three-model comparison, continuing")
        
        # ====== Unified four-model similarity analysis ======
        try:
            # Four models to compare (vs real network)
            four_models = {
                'Current Model':   gen_topology_metrics,      # Main model
                'Economical Model': economical_metrics,
                'Homophily Model':  homophily_metrics,
                'Random Model':     random_metrics
            }

            # Compute comprehensive similarity metrics
            similarity_all = {}
            for name, metrics_pack in four_models.items():
                similarity_all[name] = compute_comprehensive_similarity_for_model(real_topology_metrics, metrics_pack)

            # Generate two figures: degree and clustering distribution similarity
            plot_comprehensive_similarity_analysis(similarity_all, visual_cortex_result_dir)
            print(f"Saved similarity analysis to: {visual_cortex_result_dir}")

            # Generate unified txt file
            txt_out = os.path.join(visual_cortex_result_dir, 'comprehensive_four_model_similarity.txt')
            save_comprehensive_similarity_results_txt(similarity_all, txt_out)
            print(f"Saved comprehensive similarity results to: {txt_out}")

        except Exception as e:
            print(f"Error computing/saving four-model similarity: {e}")
        
    except Exception as e:
        print(f"Error creating advanced visualizations: {e}")
        print("Skipping advanced visualizations, continuing")

def create_model_comparison_analysis(real_metrics, current_model_metrics, economical_model_metrics, homophily_model_metrics, random_model_metrics, save_dir):
    """Create and save comparison figures across multiple generative models.
    
    Parameters
    ----------
    real_metrics : dict
        Real-network metrics.
    current_model_metrics, economical_model_metrics, homophily_model_metrics, random_model_metrics : dict
        Metrics dictionaries for each model.
    save_dir : str
        Output directory.
    """
    
    # Set global font and style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': False,
        'figure.dpi': 300
    })
    
    # 1. Four-model degree distribution comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    plot_four_model_degree_comparison(ax1, real_metrics, current_model_metrics, economical_model_metrics, homophily_model_metrics, random_model_metrics, f'{save_dir}/four_model_degree_comparison.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/four_model_degree_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Four-model clustering coefficient comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plot_four_model_clustering_comparison(ax2, real_metrics, current_model_metrics, economical_model_metrics, homophily_model_metrics, random_model_metrics, f'{save_dir}/four_model_clustering_comparison.txt')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/four_model_clustering_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved 2 four-model comparison figures to: {save_dir}")

def plot_four_model_degree_comparison(ax, real_metrics, current_model, economical_model, homophily_model, random_model, txt_path=None):
    """Plot degree distributions for real and four generated models (offset bars).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    real_metrics : dict
        Real-network metrics.
    current_model, economical_model, homophily_model, random_model : dict
        Per-model metrics dictionaries (expects ``degree_distribution``).
    txt_path : str or None, optional
        If provided, write binned densities to this path.
    """
    # Take degree sequence per model
    real_degrees       = real_metrics['degree_distribution']
    current_degrees    = current_model['degree_distribution']
    economical_degrees = economical_model['degree_distribution']
    homophily_degrees  = homophily_model['degree_distribution']
    random_degrees     = random_model['degree_distribution']

    # Put in a list for iteration
    datasets = [
        ('Real Network',     real_degrees,       '#1f77b4'),
        ('Current Model',    current_degrees,    '#ff7f0e'),
        ('Economical Model', economical_degrees, '#2ca02c'),
        ('Homophily Model',  homophily_degrees,  '#d62728'),
        ('Random Model',     random_degrees,     '#9467bd'),
    ]

    max_val = max(np.max(arr) for _, arr, _ in datasets)
    bins = np.linspace(0, max_val + 1, 50)
    bin_width = np.diff(bins)[0]
    centers = (bins[:-1] + bins[1:]) / 2.0

    # Save all models' histogram data for txt file
    all_density_vals = []
    for i, (label, arr, color) in enumerate(datasets):
        density_vals, _ = np.histogram(arr, bins=bins, density=True)
        all_density_vals.append((label, density_vals))

    # Split each bin into 5 narrower bars (width and offset only; height = hist density)
    m = len(datasets)
    group_width = bin_width * 0.90        # Group uses 90% width, small gap
    bar_width   = group_width / m
    offsets = (np.arange(m) - (m - 1) / 2.0) * bar_width  # Center-symmetric offset

    # Compute density per model (consistent with plt.hist(..., density=True)), then draw offset bars
    for i, (label, arr, color) in enumerate(datasets):
        density_vals, _ = np.histogram(arr, bins=bins, density=True)  # Height = original histogram value
        x = centers + offsets[i]
        ax.bar(
            x, density_vals,
            width=bar_width,
            label=label,
            color=color,
            alpha=0.9,
            edgecolor='white', linewidth=0.7
        )

    # Axes and title
    ax.set_xlabel('Node Degree (Number of Connections)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 25)
    ax.set_title(
        'Four Model Degree Distribution Comparison\n'
        'Comparison of degree distributions across different generation models\n'
        '• Real: Ground truth network\n'
        '• Current: Your optimized model with biophysical constraints\n'
        '• Economical: Cost-topology trade-off model\n'
        '• Homophily: Distance + homophily model\n'
        '• Random: Completely random with same sparsity',
        fontweight='bold', pad=20, fontsize=11
    )
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=False)
    ax.grid(True, alpha=0.3)
    
    # Save corresponding txt file
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Four-Model Degree Distribution Comparison Data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Bin center (Node Degree)\n")
            f.write("- Columns 2-6: Probability density per model\n")
            f.write("  - Real Network: Real network\n")
            f.write("  - Current Model: Current optimized model\n")
            f.write("  - Economical Model: Economical model\n")
            f.write("  - Homophily Model: Homophily model\n")
            f.write("  - Random Model: Random model\n")
            f.write("- Row count: Number of histogram bins (50)\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Degree':<15} {'Real Network':<20} {'Current Model':<20} {'Economical Model':<20} {'Homophily Model':<20} {'Random Model':<20}\n")
            f.write("-" * 50 + "\n")
            for i in range(len(centers)):
                f.write(f"{centers[i]:<15.2f} {all_density_vals[0][1][i]:<20.6f} {all_density_vals[1][1][i]:<20.6f} {all_density_vals[2][1][i]:<20.6f} {all_density_vals[3][1][i]:<20.6f} {all_density_vals[4][1][i]:<20.6f}\n")

def plot_four_model_clustering_comparison(ax, real_metrics, current_model, economical_model, homophily_model, random_model, txt_path=None):
    """Plot clustering-coefficient KDE curves for real and four generated models.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    real_metrics : dict
        Real-network metrics.
    current_model, economical_model, homophily_model, random_model : dict
        Per-model metrics dictionaries (expects ``clustering_coeffs``).
    txt_path : str or None, optional
        If provided, write evaluated curves to this path.
    
    Notes
    -----
    The y-axis is log-compressed for large KDE peaks to improve visual comparability.
    """
    real_clustering = real_metrics['clustering_coeffs']
    current_clustering = current_model['clustering_coeffs']
    economical_clustering = economical_model['clustering_coeffs']
    homophily_clustering = homophily_model['clustering_coeffs']
    random_clustering = random_model['clustering_coeffs']
    
    # Save raw data for txt file
    real_clustering_orig = real_clustering.copy()
    current_clustering_orig = current_clustering.copy()
    economical_clustering_orig = economical_clustering.copy()
    homophily_clustering_orig = homophily_clustering.copy()
    random_clustering_orig = random_clustering.copy()
    
    # Filter out zeros
    real_clustering = real_clustering[real_clustering > 0]
    current_clustering = current_clustering[current_clustering > 0]
    economical_clustering = economical_clustering[economical_clustering > 0]
    homophily_clustering = homophily_clustering[homophily_clustering > 0]
    random_clustering = random_clustering[random_clustering > 0]
    
    x_range = np.linspace(0, 1, 200)
    real_y = np.zeros(200)
    current_y = np.zeros(200)
    economical_y = np.zeros(200)
    homophily_y = np.zeros(200)
    random_y = np.zeros(200)
    
    if len(real_clustering) > 0 and len(current_clustering) > 0 and len(economical_clustering) > 0 and len(homophily_clustering) > 0 and len(random_clustering) > 0:
        # Use KDE for smooth distribution
        from scipy.stats import gaussian_kde
        
        real_kde = gaussian_kde(real_clustering)
        current_kde = gaussian_kde(current_clustering)
        economical_kde = gaussian_kde(economical_clustering)
        homophily_kde = gaussian_kde(homophily_clustering)
        random_kde = gaussian_kde(random_clustering)
        
        # Log-compress only values above threshold
        def compress_high_values_only(y_values, threshold=3.5, compression_factor=0.4):
            """Log-compress values above a threshold while keeping lower values unchanged.
            
            Parameters
            ----------
            y_values : numpy.ndarray
                Input values (typically KDE densities).
            threshold : float, optional
                Values above this threshold are compressed.
            compression_factor : float, optional
                Strength of the log-compression.
            
            Returns
            -------
            numpy.ndarray
                Compressed values (same shape as input).
            """
            compressed = y_values.copy()
            
            # Strongly compress only high values above threshold
            high_mask = y_values > threshold
            if np.any(high_mask):
                # Strong log-compression for high values
                compressed[high_mask] = threshold + np.log1p(y_values[high_mask] - threshold) * compression_factor
            
            return compressed
        
        # Save raw KDE values for txt file
        real_kde_orig = real_kde(x_range)
        current_kde_orig = current_kde(x_range)
        economical_kde_orig = economical_kde(x_range)
        homophily_kde_orig = homophily_kde(x_range)
        random_kde_orig = random_kde(x_range)
        
        # Plot compressed distribution
        real_y = compress_high_values_only(real_kde_orig)
        current_y = compress_high_values_only(current_kde_orig)
        economical_y = compress_high_values_only(economical_kde_orig)
        homophily_y = compress_high_values_only(homophily_kde_orig)
        random_y = compress_high_values_only(random_kde_orig)
        
        ax.plot(x_range, real_y, color='#1f77b4', linewidth=3, 
                label='Real Network', alpha=0.9)
        ax.fill_between(x_range, real_y, alpha=0.4, color='#1f77b4')
        
        ax.plot(x_range, current_y, color='#ff7f0e', linewidth=3, 
                label='Current Model', alpha=0.9)
        ax.fill_between(x_range, current_y, alpha=0.4, color='#ff7f0e')
        
        ax.plot(x_range, economical_y, color='#2ca02c', linewidth=3, 
                label='Economical Model', alpha=0.9)
        ax.fill_between(x_range, economical_y, alpha=0.4, color='#2ca02c')
        
        ax.plot(x_range, homophily_y, color='#d62728', linewidth=3, 
                label='Homophily Model', alpha=0.9)
        ax.fill_between(x_range, homophily_y, alpha=0.4, color='#d62728')
        
        ax.plot(x_range, random_y, color='#9467bd', linewidth=3, 
                label='Random Model', alpha=0.9)
        ax.fill_between(x_range, random_y, alpha=0.4, color='#9467bd')
    
    ax.set_xlabel('Clustering Coefficient (0-1)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density (high-value compressed)', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title('Four Model Clustering Coefficient Comparison\n'
                'High-value compression to reduce peak height differences\n'
                '• Higher values: More triangular connections in neighborhoods\n'
                '• Lower values: More tree-like local structures\n'
                '• Y-axis: Only high values (>3.0) are strongly log-compressed', 
                fontweight='bold', pad=20, fontsize=11)
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Save corresponding txt file
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Four-Model Clustering Coefficient Distribution Comparison Data\n")
            f.write("=" * 50 + "\n\n")
            f.write("Data description:\n")
            f.write("- Column 1: Clustering coefficient value (0-1 range)\n")
            f.write("- Columns 2-6: KDE probability density per model\n")
            f.write("  - Real Network: Real network\n")
            f.write("  - Current Model: Current optimized model\n")
            f.write("  - Economical Model: Economical model\n")
            f.write("  - Homophily Model: Homophily model\n")
            f.write("  - Random Model: Random model\n")
            f.write("- Row count: Number of KDE evaluation points (200)\n\n")
            f.write(f"Real network valid nodes (C>0): {len(real_clustering)}\n")
            f.write(f"Current model valid nodes (C>0): {len(current_clustering)}\n")
            f.write(f"Economical model valid nodes (C>0): {len(economical_clustering)}\n")
            f.write(f"Homophily model valid nodes (C>0): {len(homophily_clustering)}\n")
            f.write(f"Random model valid nodes (C>0): {len(random_clustering)}\n\n")
            f.write("Note: Y-axis values are high-value compressed (log-compression for values >3.5)\n\n")
            f.write("Data content:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Clustering Coeff':<20} {'Real Network':<20} {'Current Model':<20} {'Economical Model':<20} {'Homophily Model':<20} {'Random Model':<20}\n")
            f.write("-" * 50 + "\n")
            for i in range(len(x_range)):
                f.write(f"{x_range[i]:<20.6f} {real_y[i]:<20.6f} {current_y[i]:<20.6f} {economical_y[i]:<20.6f} {homophily_y[i]:<20.6f} {random_y[i]:<20.6f}\n")

# ====== Unified four-model similarity analysis system ======
def _cosine_similarity_from_hist(a, b, bins):
    """Compute cosine similarity between two 1D distributions using shared histogram bins.
    
    Parameters
    ----------
    a, b : array-like
        Sample values.
    bins : array-like
        Shared bin edges.
    
    Returns
    -------
    float
        Cosine similarity in [0, 1].
    """
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ha = ha / (ha.sum() + 1e-12)
    hb = hb / (hb.sum() + 1e-12)
    num = float(np.dot(ha, hb))
    den = float(np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-12)
    return max(0.0, num / den)

def _degree_bins_from_two(real_deg, model_deg, n_bins=50):
    """Build degree histogram bins that cover both distributions.
    
    Parameters
    ----------
    real_deg, model_deg : array-like
        Degree samples from two networks.
    n_bins : int, optional
        Number of bins.
    
    Returns
    -------
    numpy.ndarray
        Bin edges.
    """
    m = max(np.max(real_deg), np.max(model_deg))
    return np.linspace(0, m + 1, n_bins)

def _cluster_bins(n_bins=50):
    """Return fixed bin edges for clustering coefficients in [0, 1].
    
    Parameters
    ----------
    n_bins : int, optional
        Number of bins.
    
    Returns
    -------
    numpy.ndarray
        Bin edges from 0 to 1.
    """
    return np.linspace(0, 1, n_bins)

def _subsample_1d(arr, max_n=5000, seed=1757414760):
    """Subsample a 1D array for O(n^2) metrics to reduce compute/memory.
    
    Parameters
    ----------
    arr : array-like
        Input samples.
    max_n : int, optional
        Maximum number of samples to keep.
    seed : int, optional
        RNG seed for reproducibility.
    
    Returns
    -------
    numpy.ndarray
        Subsampled array (float).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=max_n, replace=False)
    return arr[idx]

def _kde_on_grid(samples, grid=None, bw_method="scott"):
    """Evaluate a 1D Gaussian KDE on a fixed grid.
    
    Parameters
    ----------
    samples : array-like
        Input samples.
    grid : array-like or None, optional
        Evaluation grid (defaults to linspace(0, 1, 512)).
    bw_method : str or float, optional
        Bandwidth method passed to ``scipy.stats.gaussian_kde``.
    
    Returns
    -------
    grid : numpy.ndarray
        Evaluation grid.
    y : numpy.ndarray
        KDE density values on the grid.
    """
    from scipy.stats import gaussian_kde
    if grid is None:
        grid = np.linspace(0.0, 1.0, 512)
    kde = gaussian_kde(samples, bw_method=bw_method)
    y = kde(grid)
    return grid, y

def _kde_l2_similarity(x, y):
    """Compute an L2-based KDE similarity score in [0, 1].
    
    Parameters
    ----------
    x, y : array-like
        Samples for two distributions.
    
    Returns
    -------
    float
        Similarity score (higher is more similar).
    """
    grid = np.linspace(0.0, 1.0, 512)
    _, fx = _kde_on_grid(x, grid)
    _, fy = _kde_on_grid(y, grid)
    num = np.trapz((fx - fy) ** 2, grid)
    den = np.trapz(fx ** 2 + fy ** 2, grid) + 1e-12
    return 1.0 - float(num / den)

def _density_correlation_similarity(x, y):
    """Compute correlation-based similarity between two KDE curves.
    
    Parameters
    ----------
    x, y : array-like
        Samples for two distributions.
    
    Returns
    -------
    float
        Similarity in [0, 1] computed from the KDE curve correlation.
    """
    grid = np.linspace(0.0, 1.0, 512)
    _, fx = _kde_on_grid(x, grid)
    _, fy = _kde_on_grid(y, grid)
    rho = np.corrcoef(fx, fy)[0, 1]
    rho = np.nan_to_num(rho, nan=0.0)
    return float((rho + 1.0) / 2.0)

def _mmd_rbf_1d(x, y, sigma=None, max_n=5000):
    """Compute unbiased MMD^2 with an RBF kernel for 1D samples.
    
    Parameters
    ----------
    x, y : array-like
        Samples for two distributions.
    sigma : float or None, optional
        Kernel bandwidth. If ``None``, uses a median heuristic.
    max_n : int, optional
        Maximum sample size after subsampling.
    
    Returns
    -------
    float
        MMD^2 value (0 means identical in RKHS).
    """
    x = _subsample_1d(np.asarray(x, float), max_n=max_n)
    y = _subsample_1d(np.asarray(y, float), max_n=max_n)
    x = x[:, None]; y = y[:, None]

    # Bandwidth: median heuristic
    if sigma is None:
        z = np.vstack([x, y])
        d2 = (z - z.T) ** 2
        d2 = d2[np.triu_indices_from(d2, k=1)]
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        sigma = np.sqrt(0.5 * med) + 1e-12

    gamma = 1.0 / (2.0 * sigma ** 2)

    def _rbf(d2): return np.exp(-gamma * d2)  # RBF kernel exp(-gamma * d2)

    # Kxx, Kyy (off-diagonal), Kxy
    dxx = (x - x.T) ** 2
    dyy = (y - y.T) ** 2
    dxy = (x - y.T) ** 2

    Kxx = _rbf(dxx); np.fill_diagonal(Kxx, 0.0)
    Kyy = _rbf(dyy); np.fill_diagonal(Kyy, 0.0)
    Kxy = _rbf(dxy)

    n = x.shape[0]; m = y.shape[0]
    term_x = Kxx.sum() / (n * (n - 1) + 1e-12)
    term_y = Kyy.sum() / (m * (m - 1) + 1e-12)
    term_xy = 2.0 * Kxy.mean()
    mmd2 = float(term_x + term_y - term_xy)
    return max(0.0, mmd2)

def compute_comprehensive_similarity_for_model(real_metrics, model_metrics):
    """Compute comprehensive similarity metrics for a model vs the real network.
    
    Parameters
    ----------
    real_metrics : dict
        Real-network metrics dict.
    model_metrics : dict
        Model-network metrics dict.
    
    Returns
    -------
    dict
        Nested dictionary with degree and clustering distribution metrics.
    
    Notes
    -----
    Clustering-coefficient metrics are computed on nodes with coefficient > 0 to match plotting.
    """
    r_deg = real_metrics['degree_distribution']
    r_clu_all = real_metrics['clustering_coeffs']
    m_deg = model_metrics['degree_distribution']
    m_clu_all = model_metrics['clustering_coeffs']

    # ---- Degree distribution: cosine, JS distance, Bray-Curtis ----
    dbins = _degree_bins_from_two(r_deg, m_deg, n_bins=50)
    degree_cos = _cosine_similarity_from_hist(r_deg, m_deg, dbins)

    degree_js_dist = calculate_js_divergence(r_deg, m_deg)
    degree_js_sim  = 1.0 - degree_js_dist

    degree_bc_dis  = calculate_bray_curtis_dissimilarity(r_deg, m_deg)
    degree_bc_sim  = 1.0 - degree_bc_dis

    # Clustering distribution: match plotting (use C>0 only)
    r_clu = np.asarray(r_clu_all)
    m_clu = np.asarray(m_clu_all)
    r_clu = r_clu[r_clu > 0]
    m_clu = m_clu[m_clu > 0]

    # 如果任一侧没有非零样本，则这些“条件分布”指标无法计算
    if (r_clu.size == 0) or (m_clu.size == 0):
        ks_stat, ks_p, ks_sim = np.nan, np.nan, np.nan
        cluster_js_dist, cluster_js_sim = np.nan, np.nan
        cluster_bc_dis, cluster_bc_sim = np.nan, np.nan
        # Drop advanced metrics, do not compute
    else:
        # ---- Clustering distribution: basic metrics (KS, JS distance, Bray-Curtis) ----
        ks_stat, ks_p  = stats.ks_2samp(r_clu, m_clu)
        ks_sim         = float(ks_stat)  # Use KS statistic directly, not 1-KS

        # Clustering coefficient range fixed at [0,1], use fixed bins
        cluster_js_dist = calculate_js_divergence(r_clu, m_clu, value_range=(0.0, 1.0), n_bins=100)
        cluster_js_sim  = 1.0 - cluster_js_dist

        cluster_bc_dis  = calculate_bray_curtis_dissimilarity(r_clu, m_clu, value_range=(0.0, 1.0), n_bins=100)
        cluster_bc_sim  = 1.0 - cluster_bc_dis

        # Clustering metrics: KS / JS / Bray-Curtis only

    return {
        'degree': {
            'cosine_similarity': float(degree_cos),
            'js_distance': float(degree_js_dist),
            'js_similarity': float(degree_js_sim),
            'bray_curtis_dissimilarity': float(degree_bc_dis),
            'bray_curtis_similarity': float(degree_bc_sim),
        },
        'clustering_basic': {
            'ks_statistic': float(ks_stat) if np.isfinite(ks_stat) else np.nan,
            'ks_pvalue': float(ks_p) if np.isfinite(ks_p) else np.nan,
            'ks_statistic_value': float(ks_sim) if np.isfinite(ks_sim) else np.nan,  # Store KS statistic directly, not 1-KS
            'js_distance': float(cluster_js_dist) if np.isfinite(cluster_js_dist) else np.nan,
            'js_similarity': float(cluster_js_sim) if np.isfinite(cluster_js_sim) else np.nan,
            'bray_curtis_dissimilarity': float(cluster_bc_dis) if np.isfinite(cluster_bc_dis) else np.nan,
            'bray_curtis_similarity': float(cluster_bc_sim) if np.isfinite(cluster_bc_sim) else np.nan,
        }
    }

def plot_comprehensive_similarity_analysis(sim_all, save_dir):
    """Create summary bar charts for multi-model similarity metrics and save outputs.
    
    Parameters
    ----------
    sim_all : dict
        Mapping model name -> similarity metrics (from ``compute_comprehensive_similarity_for_model``).
    save_dir : str
        Output directory for figures and TXT files.
    """
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'legend.frameon': True
    })

    methods = list(sim_all.keys())
    x = np.arange(len(methods))

    # --- Fig 1: Degree distribution similarity ---
    deg_cos = [sim_all[m]['degree']['cosine_similarity'] for m in methods]

    # For plotting use similarity (0-1, larger is better)
    deg_js_sim = [sim_all[m]['degree']['js_similarity'] for m in methods]
    deg_bc_sim = [sim_all[m]['degree']['bray_curtis_similarity'] for m in methods]

    # For txt use raw distance/dissimilarity (smaller is better)
    deg_js_dist = [sim_all[m]['degree']['js_distance'] for m in methods]
    deg_bc_dis  = [sim_all[m]['degree']['bray_curtis_dissimilarity'] for m in methods]

    width = 0.25
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width, deg_cos,    width, label='Cosine', edgecolor='black', alpha=0.9)
    ax1.bar(x,         deg_js_sim, width, label='1 - JS distance', edgecolor='black', alpha=0.9)
    ax1.bar(x + width, deg_bc_sim, width, label='1 - Bray-Curtis', edgecolor='black', alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Similarity (0-1)')
    ax1.set_title('Degree Distribution Similarity (Real vs Models)')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend()

    for bars in ax1.containers:
        ax1.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'four_model_degree_similarity.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Save degree distribution similarity data to txt file
    txt_path1 = os.path.join(save_dir, 'four_model_degree_similarity.txt')
    with open(txt_path1, 'w', encoding='utf-8') as f:
        f.write("Four-Model Degree Distribution Similarity Data\n")
        f.write("=" * 50 + "\n\n")
        f.write("Data description:\n")
        f.write("- Column 1: Model name\n")
        f.write("- Column 2: Cosine similarity (0-1 range)\n")
        f.write("- Column 3: JS distance (smaller is more similar)\n")
        f.write("- Column 4: Bray-Curtis dissimilarity (smaller is more similar)\n")
        f.write("- Row count: Number of models\n\n")
        f.write("Data content:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Model Name':<30} {'Cosine Similarity':<20} {'JS distance':<20} {'Bray-Curtis dissimilarity':<28}\n")
        f.write("-" * 50 + "\n")
        for method, cos, jsd, bcd in zip(methods, deg_cos, deg_js_dist, deg_bc_dis):
            f.write(f"{method:<30} {cos:<20.6f} {jsd:<20.6f} {bcd:<28.6f}\n")

    # --- Fig 2: Clustering distribution similarity (basic metrics) ---
    # Basic metrics: KS / JS / Bray-Curtis
    clu_ks_stat = [sim_all[m]['clustering_basic']['ks_statistic_value'] for m in methods]
    clu_js_sim = [sim_all[m]['clustering_basic']['js_similarity'] for m in methods]
    clu_bc_sim = [sim_all[m]['clustering_basic']['bray_curtis_similarity'] for m in methods]

    # For txt use raw distance/dissimilarity
    clu_js_dist = [sim_all[m]['clustering_basic']['js_distance'] for m in methods]
    clu_bc_dis  = [sim_all[m]['clustering_basic']['bray_curtis_dissimilarity'] for m in methods]

    width = 0.25  # 3 metrics
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Plot only KS, JS, Bray-Curtis
    ax2.bar(x - width, clu_ks_stat,  width, label='KS statistic', edgecolor='black', alpha=0.9)
    ax2.bar(x,         clu_js_sim,  width, label='1 - JS distance', edgecolor='black', alpha=0.9)
    ax2.bar(x + width, clu_bc_sim,  width, label='1 - Bray-Curtis', edgecolor='black', alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Metric Value (0-1)')
    ax2.set_title('Clustering Coefficient Distribution Metrics (Real vs Models)\n(KS statistic: smaller is better; JS and Bray-Curtis: larger is better)')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(loc='upper right')

    for bars in ax2.containers:
        ax2.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'four_model_clustering_similarity.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # Save clustering distribution similarity data to txt file
    txt_path2 = os.path.join(save_dir, 'four_model_clustering_similarity.txt')
    with open(txt_path2, 'w', encoding='utf-8') as f:
        f.write("Four-Model Clustering Coefficient Distribution Similarity Data\n")
        f.write("=" * 50 + "\n\n")
        f.write("Data description:\n")
        f.write("- Column 1: Model name\n")
        f.write("- Columns 2-4: Basic metrics\n")
        f.write("  - KS Statistic: smaller is more similar\n")
        f.write("  - JS distance: raw value, smaller is more similar\n")
        f.write("  - Bray-Curtis dissimilarity: raw value, smaller is more similar\n")
        f.write("- Row count: Number of models\n\n")
        f.write("Data content:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Model':<20} {'KS Stat':<12} {'JS dist':<12} {'BC dis':<12}\n")
        f.write("-" * 50 + "\n")
        for method, ks, jsd, bcd in zip(methods, clu_ks_stat, clu_js_dist, clu_bc_dis):
            f.write(f"{method:<20} {ks:<12.6f} {jsd:<12.6f} {bcd:<12.6f}\n")

def save_comprehensive_similarity_results_txt(sim_all, save_path):
    """Write comprehensive similarity metrics for all models to a text file.
    
    Parameters
    ----------
    sim_all : dict
        Mapping model name -> similarity metrics.
    save_path : str
        Output file path.
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Four-Model Distribution Similarity Results (Real vs Models)\n")
        f.write("=======================================================================\n\n")
        
        for name, res in sim_all.items():
            f.write(f"[{name}]\n")
            f.write("=" * 50 + "\n")
            
            # Degree distribution metrics
            f.write("Degree Distribution Similarity:\n")
            f.write(f"  - Cosine similarity         : {res['degree']['cosine_similarity']:.6f}\n")
            f.write(f"  - JS distance               : {res['degree']['js_distance']:.6f}\n")
            f.write(f"  - Bray–Curtis dissimilarity : {res['degree']['bray_curtis_dissimilarity']:.6f}\n")
            
            # Clustering basic metrics: KS / JS / Bray-Curtis
            f.write("Clustering Coefficient Distribution - Basic Metrics:\n")
            f.write(f"  - KS statistic              : {res['clustering_basic']['ks_statistic']:.6f}\n")
            f.write(f"  - KS statistic value        : {res['clustering_basic']['ks_statistic_value']:.6f}\n")
            f.write(f"  - KS p-value                : {res['clustering_basic']['ks_pvalue']:.6e}\n")
            f.write(f"  - JS distance               : {res['clustering_basic']['js_distance']:.6f}\n")
            f.write(f"  - Bray–Curtis dissimilarity : {res['clustering_basic']['bray_curtis_dissimilarity']:.6f}\n\n")
        
        f.write("Notes:\n")
        f.write("=" * 50 + "\n")
        f.write("- JS distance is the return value of standard library jensenshannon (square root of JS divergence, range [0,1]).\n")
        f.write("- Bray-Curtis takes values in [0,1].\n")
        f.write("- KS gives statistic and p-value; smaller statistic means more similar distributions.\n")
        f.write("- Bray-Curtis in [0,1]; smaller means more similar.\n")

def save_comprehensive_ablation_results_txt(sim_all, ablation_results, save_path):
    """Write comprehensive similarity metrics for ablation experiments to a text file.
    
    Parameters
    ----------
    sim_all : dict
        Mapping experiment name -> similarity metrics.
    ablation_results : dict
        Mapping experiment name -> metadata (e.g., alpha and flags) and raw metrics.
    save_path : str
        Output file path.
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Ablation Similarity Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Experiment description:\n")
        f.write("- Original Model: Full method (all constraints)\n")
        f.write("- Ablation: No Degree Constraint: Ablation of degree heavy-tail constraint\n")
        f.write("- Ablation: No Energy Constraint: Ablation of energy constraint\n")
        f.write("- Ablation: Alpha=0.1: Ablation of entropy constraint (fixed alpha=0.1)\n\n")
        
        # Add summary table
        f.write("=" * 80 + "\n")
        f.write("Summary table: Ablation metrics comparison\n")
        f.write("=" * 80 + "\n\n")
        
        # Degree distribution metrics summary
        f.write("Degree distribution metrics summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment Name':<35} {'Cosine Sim':<15} {'JS Distance':<15} {'BC Dissimilarity':<15}\n")
        f.write("-" * 80 + "\n")
        for name in sim_all.keys():
            res = sim_all[name]
            f.write(f"{name:<35} {res['degree']['cosine_similarity']:<15.6f} {res['degree']['js_distance']:<15.6f} {res['degree']['bray_curtis_dissimilarity']:<15.6f}\n")
        f.write("\n")
        
        # Clustering coefficient distribution metrics summary
        f.write("Clustering coefficient distribution metrics summary (C>0 only):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment Name':<35} {'KS Statistic':<15} {'JS Distance':<15} {'BC Dissimilarity':<15}\n")
        f.write("-" * 80 + "\n")
        for name in sim_all.keys():
            res = sim_all[name]
            ks_val = res['clustering_basic']['ks_statistic_value']
            if np.isnan(ks_val):
                ks_str = "N/A"
            else:
                ks_str = f"{ks_val:.6f}"
            js_val = res['clustering_basic']['js_distance']
            if np.isnan(js_val):
                js_str = "N/A"
            else:
                js_str = f"{js_val:.6f}"
            bc_val = res['clustering_basic']['bray_curtis_dissimilarity']
            if np.isnan(bc_val):
                bc_str = "N/A"
            else:
                bc_str = f"{bc_val:.6f}"
            f.write(f"{name:<35} {ks_str:<15} {js_str:<15} {bc_str:<15}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Detailed results:\n")
        f.write("=" * 80 + "\n\n")
        
        for name, res in sim_all.items():
            ablation_info = ablation_results.get(name, {})
            f.write(f"[{name}]\n")
            f.write("=" * 80 + "\n")
            f.write(f"Experiment parameters:\n")
            f.write(f"  - Alpha: {ablation_info.get('alpha', 'N/A'):.2f}\n")
            f.write(f"  - Use degree heavy-tail constraint: {ablation_info.get('use_degree_constraint', 'N/A')}\n")
            f.write(f"  - Use energy constraint: {ablation_info.get('use_energy_constraint', 'N/A')}\n\n")
            
            # Degree distribution metrics
            f.write("Degree distribution metrics:\n")
            f.write(f"  - Cosine similarity         : {res['degree']['cosine_similarity']:.6f}\n")
            f.write(f"  - JS distance               : {res['degree']['js_distance']:.6f}\n")
            f.write(f"  - Bray–Curtis dissimilarity : {res['degree']['bray_curtis_dissimilarity']:.6f}\n\n")
            
            # Clustering basic metrics: KS / JS / Bray-Curtis (raw values)
            f.write("Clustering coefficient distribution metrics (C>0 only):\n")
            f.write(f"  - KS statistic              : {res['clustering_basic']['ks_statistic']:.6f}\n")
            f.write(f"  - KS statistic value        : {res['clustering_basic']['ks_statistic_value']:.6f}\n")
            f.write(f"  - KS p-value                : {res['clustering_basic']['ks_pvalue']:.6e}\n")
            f.write(f"  - JS distance               : {res['clustering_basic']['js_distance']:.6f}\n")
            f.write(f"  - Bray–Curtis dissimilarity : {res['clustering_basic']['bray_curtis_dissimilarity']:.6f}\n\n")
            
            # Network topology metrics
            metrics = ablation_info.get('metrics', {})
            if metrics:
                f.write("Network topology metrics:\n")
                f.write(f"  - Average degree           : {metrics.get('avg_degree', 'N/A'):.6f}\n")
                f.write(f"  - Degree std               : {metrics.get('degree_std', 'N/A'):.6f}\n")
                f.write(f"  - Avg clustering           : {metrics.get('avg_clustering', 'N/A'):.6f}\n")
                f.write(f"  - Network density          : {metrics.get('network_density', 'N/A'):.6f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Metric description:\n")
        f.write("Degree distribution metrics:\n")
        f.write("  - Cosine similarity: range [0,1], larger is more similar\n")
        f.write("  - JS distance: range [0,1], smaller is more similar\n")
        f.write("  - Bray-Curtis dissimilarity: range [0,1], smaller is more similar\n")
        f.write("Clustering coefficient distribution metrics (C>0 only):\n")
        f.write("  - KS statistic: range [0,1], smaller is more similar\n")
        f.write("  - JS distance: range [0,1], smaller is more similar\n")
        f.write("  - Bray-Curtis dissimilarity: range [0,1], smaller is more similar\n")
        f.write("\nAblation analysis:\n")
        f.write("- By comparing ablation results, one can evaluate the impact of each constraint on network generation quality.\n")
        f.write("- For degree distribution: Cosine similarity larger is better; JS distance and Bray-Curtis dissimilarity smaller is better.\n")
        f.write("- For clustering distribution: KS statistic, JS distance and Bray-Curtis dissimilarity smaller is better.\n")
        f.write("- If an ablation's metrics worsen significantly, that constraint is important for network generation.\n")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} s") 
