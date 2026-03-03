import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import json
import gc  # Garbage collector
from caveclient import CAVEclient
from scipy import stats
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RANDOM_SEED = 1757414760
print(f"Random seed: {RANDOM_SEED}")
np.random.seed(RANDOM_SEED)
cp.random.seed(RANDOM_SEED)

# Paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, "synapses_pni_2.csv")
results_dir = os.path.join(current_dir, "visual_cortex_alpha_results")  # Cached alpha sweep results
visual_cortex_result_dir = os.path.join(current_dir, "neuron_count_experiment_results")  # Output directory

client = CAVEclient('minnie65_public')

# Configure CAVE client networking (timeout/retry)
def configure_cave_client():
    """Configure the global CAVE client session (timeout and HTTP retries).

    Notes
    -----
    Updates the module-level ``client`` in place.
    """
    # Increase request timeout
    client.timeout = 60  # seconds
    
    # Retry policy for transient HTTP errors
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Requests adapter with retries
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    # Attach adapter to the client session
    if hasattr(client, 'session'):
        client.session.mount("http://", adapter)
        client.session.mount("https://", adapter)
    
    print("CAVE client configured with retry policy and timeout")

# Apply networking configuration
configure_cave_client()

def clear_gpu_memory():
    """Release CuPy memory pools to reduce GPU memory pressure.

    Returns
    -------
    None
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def clear_cpu_memory():
    """Trigger Python garbage collection to reduce CPU memory pressure.

    Returns
    -------
    None
    """
    gc.collect()

def load_ave_deg_results(ave_deg):
    """Load cached alpha-sweep results for a given average degree.

    Parameters
    ----------
    ave_deg : int
        Average-degree identifier used in the cache filename.

    Returns
    -------
    dict or None
        Parsed JSON payload if the cache exists; otherwise ``None``.
    """
    result_file = os.path.join(results_dir, f"ave_deg_{ave_deg}.txt")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def collect_all_results():
    """Collect all cached alpha-sweep results under ``results_dir``.

    Returns
    -------
    dict[int, dict]
        Mapping ``ave_deg -> results`` loaded from JSON files.
    """
    all_results = {}
    
    if not os.path.exists(results_dir):
        return all_results
    
    for filename in os.listdir(results_dir):
        if filename.startswith("ave_deg_") and filename.endswith(".txt"):
            try:
                # Robust filename parsing
                parts = filename.split("_")
                if len(parts) >= 3:
                    # Extract ave_deg (strip .txt)
                    number_part = parts[2].split(".")[0]
                    ave_deg = int(number_part)
                    with open(os.path.join(results_dir, filename), 'r') as f:
                        all_results[ave_deg] = json.load(f)
            except (ValueError, IndexError) as e:
                print(f"Skipping unparseable file: {filename}, error: {e}")
                continue
    
    return all_results

def calculate_distances(coordinates):
    """Compute the full pairwise Euclidean distance matrix on GPU.

    Parameters
    ----------
    coordinates : cupy.ndarray
        Array of shape ``(n, d)`` containing neuron coordinates on GPU.

    Returns
    -------
    cupy.ndarray
        Pairwise Euclidean distances with shape ``(n, n)``.
    """
    print("Computing pairwise neuron distances...")
    n = coordinates.shape[0]
    distances = cp.zeros((n, n), dtype=cp.float32)
    
    # GPU distance matrix (row-wise)
    for i in tqdm(range(n), desc="Computing distance matrix"):
        diff = coordinates - coordinates[i]
        distances[i] = cp.sqrt(cp.sum(diff**2, axis=1))
    
    print("Distance computation completed")
    return distances

def calculate_connection_probabilities(distances, alpha):
    """Compute a distance-based connection probability matrix with balancing heuristics.

    The base probability decays with distance as ``d^{-alpha}``, then applies (i) a
    heavy-tailed node reweighting and (ii) an adaptive hub penalty based on the
    skewness of node-wise probability mass.

    Parameters
    ----------
    distances : cupy.ndarray
        Pairwise distance matrix of shape ``(n, n)``.
    alpha : float
        Distance decay exponent.

    Returns
    -------
    cupy.ndarray
        Probability matrix in ``[0, 1]`` with a zero diagonal.
    """
    print(f"Computing connection probability for alpha={alpha:.2f}...")
    n = distances.shape[0]
    
    # Step 1: distance-based prior
    prob_matrix = distances**(-alpha)
    cp.fill_diagonal(prob_matrix, 0)
    prob_matrix = prob_matrix/cp.max(prob_matrix)
    
    # Step 2: degree-shape reweighting (power-law)
    print("Applying multi-objective balance...")
    
    # Target degree-shape prior (heuristic)
    # Encourage many low-degree nodes and a few hubs
    target_degree_weights = cp.zeros(n, dtype=cp.float16)
    
    # Power-law weights as degree prior
    # Heavy-tailed weighting across nodes
    node_indices = cp.arange(n, dtype=cp.float16)
    power_law_weights = 1.0 / (node_indices + 1)**0.8  # Power-law
    power_law_weights = power_law_weights / cp.sum(power_law_weights) * n  # Normalize
    
    # Apply node weights to pairwise probabilities
    prob_matrix = prob_matrix * power_law_weights[:, cp.newaxis]
    prob_matrix = prob_matrix * power_law_weights[cp.newaxis, :]
    
    # Free temporary GPU buffers
    del power_law_weights, node_indices
    cp.get_default_memory_pool().free_all_blocks()
    
    # Step 3: adaptive hub penalty (blockwise)
    print("Applying adaptive balance...")
    
    # Node-wise probability mass (proxy for degree)
    node_connection_density = cp.sum(prob_matrix, axis=1)
    
    # Skewness of node-wise mass
    mean_density = cp.mean(node_connection_density)
    std_density = cp.std(node_connection_density)
    skewness = cp.mean(((node_connection_density - mean_density) / (std_density + 1e-6))**3)
    
    # Choose penalty strength from skewness
    if skewness > 2.0:  # Too right-skewed: penalize hubs
        balance_factor = 0.5  # Downweight high-density nodes
        print(f"Detected heavy right skew (skewness={skewness:.2f}), applying strong balance")
    elif skewness < 1.0:  # Too uniform: keep more hubs
        balance_factor = 0.8  # Mild penalty
        print(f"Detected overly uniform distribution (skewness={skewness:.2f}), applying weak balance")
    else:  # Moderate penalty
        balance_factor = 0.65  # Balance
        print(f"Detected moderate distribution (skewness={skewness:.2f}), applying moderate balance")
    
    # Apply hub penalty
    density_threshold = cp.percentile(node_connection_density, 75)  # 75th percentile threshold
    high_density_mask = node_connection_density > density_threshold
    
    # Per-node penalty factors
    penalty_factor = cp.ones_like(node_connection_density, dtype=cp.float16)
    penalty_factor[high_density_mask] = balance_factor
    
    # Apply penalties blockwise (memory bound)
    print("Applying adaptive balance in blocks...")
    block_size = 1000
    for i in tqdm(range(0, n, block_size), desc="Applying adaptive balance"):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)
            
            # Current block
            prob_block = prob_matrix[i:end_i, j:end_j]
            penalty_i = penalty_factor[i:end_i, cp.newaxis]
            penalty_j = penalty_factor[j:end_j, cp.newaxis]
            
            # Apply penalty
            prob_matrix[i:end_i, j:end_j] = prob_block * penalty_i * penalty_j.T
            
            # Release block temporaries
            del prob_block, penalty_i, penalty_j
            cp.get_default_memory_pool().free_all_blocks()
    
    # Free temporary GPU buffers
    del node_connection_density, density_threshold, high_density_mask, penalty_factor
    del mean_density, std_density, skewness
    cp.get_default_memory_pool().free_all_blocks()
    
    # Clip to [0, 1]
    prob_matrix = cp.clip(prob_matrix, 0, 1)

    print("Connection probability computation completed")
    return prob_matrix

def generate_connection_matrix(prob_matrix, ave_deg, multiplier=50):
    """Sample a binary adjacency matrix from connection probabilities.

    Parameters
    ----------
    prob_matrix : cupy.ndarray
        Connection probability matrix of shape ``(n, n)``.
    ave_deg : float
        Kept for API compatibility; not used in this function.
    multiplier : float, default=50
        Scales probabilities before sampling: an edge is sampled if
        ``U < prob_matrix * multiplier``.

    Returns
    -------
    cupy.ndarray
        Binary adjacency matrix (dtype ``int8``).
    """
    print("Generating connection matrix...")
    # Deterministic sampling
    cp.random.seed(RANDOM_SEED)
    
    # Sample adjacency by thresholding uniform noise
    # U ~ Uniform[0, 1) on GPU
    random_matrix = cp.random.rand(*prob_matrix.shape)
    
    # Edge if U < p * multiplier
    result_matrix = random_matrix < prob_matrix * multiplier
    print("Connection matrix generation completed")
    
    # Return as int8 (0/1)
    return result_matrix.astype(cp.int8)

def calculate_entropy(connection_matrix):
    """Compute average Shannon entropy over combined 1st/2nd-order neighborhoods.

    This computes per-node entropy on the normalized neighbor distribution after
    merging first- and second-order reachability, then averages across nodes.
    Uses batching to reduce peak GPU memory usage.

    Parameters
    ----------
    connection_matrix : cupy.ndarray
        Binary adjacency matrix of shape ``(n, n)``.

    Returns
    -------
    float
        Average entropy across nodes.
    """
    print("Computing entropy...")
    n = connection_matrix.shape[0]
    batch_size = 500  # Batch size (memory bound)
    total_entropy = 0
    total_nodes = 0
    
    # Second-order reachability via A^2
    print("Computing second-order neighbor matrix...")
    second_order_matrix = cp.dot(connection_matrix, connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Computing entropy"):
        end_idx = min(i + batch_size, n)
        # Slice current batch
        batch_matrix = connection_matrix[i:end_idx]
        batch_second_order = second_order_matrix[i:end_idx]
        
        # Combine 1st- and 2nd-order neighbors
        combined_matrix = batch_matrix + batch_second_order
        
        # Binarize reachability
        combined_matrix = (combined_matrix > 0).astype(cp.int8)
        cp.fill_diagonal(combined_matrix, 0)
        
        # Row normalization
        row_sums = cp.sum(combined_matrix, axis=1)
        row_sums = cp.where(row_sums == 0, 1, row_sums)  # Avoid divide-by-zero
        
        # Entropy per node
        q_j = combined_matrix / row_sums[:, cp.newaxis]
        q_j = cp.where(q_j == 0, 1, q_j)  # Avoid log(0)
        batch_entropy = -cp.sum(q_j * cp.log(q_j), axis=1)
        
        # Accumulate
        total_entropy += float(cp.sum(batch_entropy))
        total_nodes += end_idx - i
    
    # Average over nodes
    avg_entropy = total_entropy / total_nodes if total_nodes > 0 else 0
    print("Entropy computation completed")
    return avg_entropy

def get_energy_bound(connection_matrix, distances, w):
    """Apply a per-node distance (energy) budget to prune connections.

    For each node, neighbors are sorted by distance and edges are retained until
    the cumulative distance of retained edges exceeds ``w``.

    Parameters
    ----------
    connection_matrix : cupy.ndarray
        Binary adjacency matrix of shape ``(n, n)``.
    distances : cupy.ndarray
        Pairwise distances of shape ``(n, n)``.
    w : float
        Energy threshold (distance budget).

    Returns
    -------
    cupy.ndarray
        Pruned binary adjacency matrix (dtype ``int8``).
    """
    print(f"Applying energy constraint (threshold w={w:.2f})...")
    n = connection_matrix.shape[0]
    batch_size = 500  # Batch size (memory bound)
    new_connection_matrix = cp.zeros_like(connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Applying energy constraint"):
        end_idx = min(i + batch_size, n)
        # Batch adjacency/distances
        batch_adj = connection_matrix[i:end_idx]
        batch_dis = distances[i:end_idx]
        
        # Sort neighbors by distance
        sorted_indices = cp.argsort(batch_dis, axis=1)
        sorted_dis = cp.take_along_axis(batch_dis, sorted_indices, axis=1)
        sorted_adj = cp.take_along_axis(batch_adj, sorted_indices, axis=1)
        
        valid_connections = (sorted_adj != 0)
        energy_cumsum = cp.cumsum(sorted_dis * valid_connections, axis=1)
        threshold_mask = (energy_cumsum <= w) & valid_connections
        reverse_indices = cp.argsort(sorted_indices, axis=1)
        original_order_mask = cp.take_along_axis(threshold_mask, reverse_indices, axis=1)
        
        # Keep closest edges under energy budget
        new_connection_matrix[i:end_idx] = original_order_mask.astype(cp.int8)
    
    print("Energy constraint applied")
    return new_connection_matrix

def calculate_network_topology_metrics_gpu(connection_matrix, eig_k=10):
    # Compute topology metrics on GPU (degree, clustering, density).
    # Args: connection_matrix (n,n) binary adj (NumPy/CuPy); eig_k unused.
    # Returns: dict of summary stats and per-node arrays (moved to CPU).
    print("[GPU] Computing network topology metrics...")
    # Ensure CuPy array
    if isinstance(connection_matrix, np.ndarray):
        print("[GPU] Input is NumPy array, converting to CuPy...")
        adj_matrix = cp.array(connection_matrix)
    else:
        adj_matrix = connection_matrix

    n = adj_matrix.shape[0]
    print(f"[GPU] Network node count: {n}")

    # 1) Degree distribution
    print("[GPU] Computing degree distribution...")
    degrees = cp.sum(adj_matrix, axis=1)
    avg_degree = float(cp.mean(degrees).get())
    degree_std = float(cp.std(degrees).get())
    print(f"[GPU] Mean degree: {avg_degree:.2f}, degree std: {degree_std:.2f}")

    # 2) Clustering coefficient (diag(A^3))
    print("[GPU] Computing clustering coefficient (triangle count, batched)...")
    batch_size = 2000 if n > 4000 else n
    triangles = cp.zeros(n, dtype=cp.float32)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        print(f"[GPU] Triangle count for nodes {i} to {end}...")
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

    # 3) Network density
    print("[GPU] Computing network density...")
    total_connections = float(cp.sum(adj_matrix).get()) / 2
    max_possible_connections = n * (n - 1) / 2
    network_density = total_connections / max_possible_connections
    print(f"[GPU] Network density: {network_density:.6f}")

    print("[GPU] Network topology metrics completed")
    return {
        'avg_degree': avg_degree,
        'degree_std': degree_std,
        'avg_clustering': avg_clustering,
        'network_density': network_density,
        'degree_distribution': degrees.get(),
        'clustering_coeffs': clustering_coeffs.get(),
        'triangles': triangles.get(),
    }

def query_neuron_connections_with_retry(client, root_id, max_retries=3, timeout=30):
    """Query synaptic partners for a neuron with retries.

    Parameters
    ----------
    client : caveclient.CAVEclient
        Configured CAVE client.
    root_id : int
        Neuron root ID to query.
    max_retries : int, default=3
        Maximum retry attempts.
    timeout : int, default=30
        Reserved for API compatibility (not used directly).

    Returns
    -------
    set
        Set of partner neuron root IDs connected to ``root_id``.
    """
    for attempt in range(max_retries):
        try:
            # As postsynaptic: query presynaptic partners
            df_post = client.materialize.synapse_query(post_ids=root_id)
            pre_ids = set(df_post.iloc[:, 6])  # presynaptic root_id column
            
            # As presynaptic: query postsynaptic partners
            df_pre = client.materialize.synapse_query(pre_ids=root_id)
            post_ids = set(df_pre.iloc[:, 8])  # postsynaptic root_id column
            
            # Union partners
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
    """Checkpoint long-running connection queries to a NumPy file.

    Parameters
    ----------
    id_list : list
        List of neuron root IDs (for reference).
    processed_indices : list[int]
        Indices in ``id_list`` that have been processed.
    conn_matrix : array-like
        Current adjacency matrix (NumPy or CuPy).
    save_file : str
        Output ``.npy`` path.

    Returns
    -------
    None
    """
    progress_data = {
        'processed_indices': processed_indices,
        'connection_matrix': cp.asnumpy(conn_matrix) if isinstance(conn_matrix, cp.ndarray) else conn_matrix,
        'timestamp': time.time()
    }
    np.save(save_file, progress_data)
    print(f"Progress saved to {save_file}")

def load_connection_progress(save_file):
    """Load a previously saved connection-query checkpoint.

    Parameters
    ----------
    save_file : str
        Path to the ``.npy`` checkpoint.

    Returns
    -------
    tuple[list[int], array-like | None]
        ``(processed_indices, connection_matrix)``; returns ``([], None)`` if the
        checkpoint does not exist.
    """
    if os.path.exists(save_file):
        progress_data = np.load(save_file, allow_pickle=True).item()
        return progress_data['processed_indices'], progress_data['connection_matrix']
    return [], None

def create_split_violin_plot(all_neuron_count_results, save_dir):
    """Create a split violin-style histogram plot for degree distributions.

    Parameters
    ----------
    all_neuron_count_results : dict
        Mapping ``neuron_count -> results`` where each results dict contains
        ``real_metrics`` and ``gen_metrics`` with ``degree_distribution`` arrays.
    save_dir : str
        Directory to save the figure and sidecar text files.

    Returns
    -------
    None
    """
    print("Creating split violin plot...")
    
    # Matplotlib style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300
    })
    
    # Collect results by neuron count
    neuron_counts = sorted(all_neuron_count_results.keys())
    print(f"Neuron counts: {neuron_counts}")
    
    # Figure canvas
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layout: split violin per neuron count
    base_position = 1.5  # Base x-position
    spacing = 2.0        # X-spacing
    positions = {count: base_position + pos * spacing for pos, count in enumerate(neuron_counts)}
    max_bar_width = spacing / 2  # Max half-width
    
    # Step 1: shared bins (global degree range)
    global_max_degree = 0
    for neuron_count in neuron_counts:
        real_degrees = all_neuron_count_results[neuron_count]['real_metrics']['degree_distribution']
        gen_degrees = all_neuron_count_results[neuron_count]['gen_metrics']['degree_distribution']
        max_degree = max(np.max(real_degrees), np.max(gen_degrees))
        global_max_degree = max(global_max_degree, max_degree)
    
    # Shared histogram bins
    unified_bins = np.linspace(0, global_max_degree + 1, 50)
    unified_bin_centers = (unified_bins[:-1] + unified_bins[1:]) / 2
    unified_bin_diff = np.diff(unified_bins)[0]
    
    # Step 2: histograms (shared bins)
    all_histograms = {}
    all_max_density = 0
    
    for neuron_count in neuron_counts:
        print(f"Processing data for {neuron_count} neurons...")
        
        real_degrees = all_neuron_count_results[neuron_count]['real_metrics']['degree_distribution']
        gen_degrees = all_neuron_count_results[neuron_count]['gen_metrics']['degree_distribution']
        
        real_hist, _ = np.histogram(real_degrees, bins=unified_bins, density=True)
        gen_hist, _ = np.histogram(gen_degrees, bins=unified_bins, density=True)
        
        all_max_density = max(all_max_density, np.max(real_hist), np.max(gen_hist))
        
        all_histograms[neuron_count] = {
            'real_hist': real_hist,
            'gen_hist': gen_hist
        }
    
    # Step 3: shared scaling
    if all_max_density > 0:
        unified_scale_factor = max_bar_width / all_max_density
    else:
        unified_scale_factor = 1.0
    
    print(f"Max density: {all_max_density:.6f}, max bar width: {max_bar_width:.2f}, scale factor: {unified_scale_factor:.6f}")
    
    # Step 4: render
    for i, neuron_count in enumerate(neuron_counts):
        hist_data = all_histograms[neuron_count]
        real_hist = hist_data['real_hist']
        gen_hist = hist_data['gen_hist']
        center_x = positions[neuron_count]
        
        real_hist_scaled = real_hist * unified_scale_factor
        gen_hist_scaled = gen_hist * unified_scale_factor
        
        mask = unified_bin_centers <= 25
        
        # Left: real network
        left_positions = center_x - real_hist_scaled[mask]
        ax.barh(
            unified_bin_centers[mask],
            real_hist_scaled[mask],
            height=unified_bin_diff,
            left=left_positions,
            color='#2E86AB',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            label='Real Network' if i == 0 else ""
        )
        
        # Right: generated network
        ax.barh(
            unified_bin_centers[mask],
            gen_hist_scaled[mask],
            height=unified_bin_diff,
            left=center_x,
            color='#A23B72',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            label='Generated Network' if i == 0 else ""
        )
        
        # Center line
        ax.axvline(x=center_x, color='black', linewidth=0.75, alpha=0.9, zorder=10)
    
    # Axes/labels
    ax.set_xlabel('Neuron Count', fontweight='bold', fontsize=14)
    ax.set_ylabel('Node Degree', fontweight='bold', fontsize=14)
    ax.set_title('Degree Distribution Comparison', fontweight='bold', pad=15, fontsize=14)
    
    x_ticks = list(positions.values())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{count:,}' for count in neuron_counts], fontsize=12)
    
    ax.set_ylim(0, 25)
    max_position = max(positions.values())
    ax.set_xlim(0, max_position + 1)
    
    ax.legend(loc='upper right', fontsize=11, frameon=True,
              fancybox=True, shadow=True, framealpha=0.9)
    ax.grid(False)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'split_violin_degree_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Split violin plot saved to: {save_path}")
    
    # Save sidecar .txt with histogram data
    txt_path = os.path.splitext(save_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Data description for split_violin_degree_distribution.png\n")
        f.write("=" * 60 + "\n\n")
        f.write("This file records the degree-distribution histogram data used to draw split_violin_degree_distribution.png.\n")
        f.write("Each neuron count corresponds to two degree-distribution curves:\n")
        f.write("  - Real Network: node-degree histogram (probability density) of the real connectome\n")
        f.write("  - Generated Network: node-degree histogram (probability density) of the generated connectome\n\n")
        
        f.write("Global settings:\n")
        f.write(f"  Neuron count list: {', '.join(str(c) for c in neuron_counts)}\n")
        f.write(f"  Global max degree (before truncation to 0-25): {global_max_degree:.4f}\n")
        f.write(f"  Histogram bin count: {len(unified_bin_centers)}\n")
        f.write(f"  Histogram bin edges (degree): {unified_bins.tolist()}\n")
        f.write(f"  Histogram bin centers (degree): {unified_bin_centers.tolist()}\n")
        f.write(f"  Unified scale factor (density to horizontal width): {unified_scale_factor:.6f}\n\n")
        
        f.write("Notes:\n")
        f.write("  - Each bin value is the probability density of node degree in that interval (np.histogram(..., density=True)).\n")
        f.write("  - The left/right half-violin shapes in the figure are obtained by scaling these densities with the unified scale factor.\n")
        f.write("  - The figure displays degree in 0-25 only; full histogram data is given here.\n\n")
        
        for neuron_count in neuron_counts:
            hist_data = all_histograms[neuron_count]
            real_hist = hist_data['real_hist']
            gen_hist = hist_data['gen_hist']
            
            f.write("-" * 60 + "\n")
            f.write(f"Neuron count = {neuron_count}\n")
            f.write("  Real Network degree-distribution probability density (by bin order):\n")
            f.write("    " + ", ".join(f"{v:.6e}" for v in real_hist) + "\n")
            f.write("  Generated Network degree-distribution probability density (by bin order):\n")
            f.write("    " + ", ".join(f"{v:.6e}" for v in gen_hist) + "\n\n")
    
    print(f"Figure data and description saved to: {txt_path}")
    # End sidecar export
    
    # Write summary statistics file
    stats_file = os.path.join(save_dir, 'neuron_count_experiment_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Neuron Count Experiment Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        for neuron_count in neuron_counts:
            result = all_neuron_count_results[neuron_count]
            real_metrics = result['real_metrics']
            gen_metrics = result['gen_metrics']
            
            f.write(f"Neuron Count: {neuron_count:,}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Alpha: {result['best_alpha']:.2f}\n")
            f.write(f"Best Entropy: {result['best_entropy']:.4f}\n")
            f.write(f"Real Average Degree: {result['real_ave_deg']:.2f}\n\n")
            
            f.write("Real Network Metrics:\n")
            f.write(f"  - Average Degree: {real_metrics['avg_degree']:.4f}\n")
            f.write(f"  - Degree Std: {real_metrics['degree_std']:.4f}\n")
            f.write(f"  - Max Degree: {np.max(real_metrics['degree_distribution']):.0f}\n")
            f.write(f"  - Min Degree: {np.min(real_metrics['degree_distribution']):.0f}\n\n")
            
            f.write("Generated Network Metrics:\n")
            f.write(f"  - Average Degree: {gen_metrics['avg_degree']:.4f}\n")
            f.write(f"  - Degree Std: {gen_metrics['degree_std']:.4f}\n")
            f.write(f"  - Max Degree: {np.max(gen_metrics['degree_distribution']):.0f}\n")
            f.write(f"  - Min Degree: {np.min(gen_metrics['degree_distribution']):.0f}\n\n")
            
            f.write("=" * 50 + "\n\n")
    
    print(f"Statistics saved to: {stats_file}")

def main():
    # Run neuron-count sweep, generate matched synthetic graphs, and save plots/stats.
    print("Setting random seed...")
    np.random.seed(RANDOM_SEED)
    cp.random.seed(RANDOM_SEED)
    
    # Select GPU device
    print("Initializing GPU...")
    cp.cuda.Device(0).use()
    
    # Ensure output directory exists
    if not os.path.exists(visual_cortex_result_dir):
        os.makedirs(visual_cortex_result_dir)
    
    # Neuron counts to evaluate
    neuron_counts = [10000, 20000, 30000]
    print(f"Running experiments for neuron counts: {neuron_counts}")
    
    # Aggregate results
    all_neuron_count_results = {}
    
    # Run experiment per neuron count
    for target_neuron_count in neuron_counts:
        print(f"\n{'='*60}")
        print(f"Starting experiment for {target_neuron_count} neurons")
        print(f"{'='*60}")
        
        # Load cached real adjacency if present
        real_conn_file = os.path.join(visual_cortex_result_dir, f'real_connection_matrix_{target_neuron_count}.npy')
        neuron_ids_file = os.path.join(visual_cortex_result_dir, f'neuron_ids_{target_neuron_count}.npy')
        id_to_coord_file = os.path.join(visual_cortex_result_dir, f'id_to_coord_{target_neuron_count}.npy')
        
        if os.path.exists(real_conn_file) and os.path.exists(neuron_ids_file):
            print(f"Found cached real connection matrix for {target_neuron_count} neurons, loading...")
            real_connection_matrix = np.load(real_conn_file)
            id_list = np.load(neuron_ids_file).tolist()
            id_to_coord = np.load(id_to_coord_file, allow_pickle=True).item()
            id_to_idx = {id_: idx for idx, id_ in enumerate(id_list)}
            n = len(id_list)
            print(f"Loaded real connection matrix for {n} neurons")
        else:
            print(f"No cached real connection matrix for {target_neuron_count} neurons, computing...")
            
            # Step 1: collect neuron IDs and seed coordinates from CSV
            print("Loading synapse data...")
            id_set = set()
            id_list = []
            id_to_idx = {}
            id_to_coord = {}

            # Read neuron IDs (pre/post root IDs)
            df_ids = pd.read_csv(csv_file, usecols=[6, 11], nrows=200000)
            pre_ids = df_ids.iloc[:, 0].values
            post_ids = df_ids.iloc[:, 1].values

            # Read synapse endpoint coordinates
            df_coords = pd.read_csv(csv_file, usecols=[2,3,4,7,8,9], nrows=200000)
            pre_coords = df_coords.iloc[:, 0:3].values
            post_coords = df_coords.iloc[:, 3:6].values

            # Collect unique neurons up to target count
            for i in range(len(pre_ids)):
                # Add presynaptic neuron if new
                if pre_ids[i] not in id_set:
                    id_set.add(pre_ids[i])
                    id_list.append(pre_ids[i])
                    id_to_idx[pre_ids[i]] = len(id_list) - 1
                    id_to_coord[pre_ids[i]] = pre_coords[i]
                    if len(id_list) >= target_neuron_count:
                        break
                
                # Add postsynaptic neuron if new
                if post_ids[i] not in id_set:
                    id_set.add(post_ids[i])
                    id_list.append(post_ids[i])
                    id_to_idx[post_ids[i]] = len(id_list) - 1
                    id_to_coord[post_ids[i]] = post_coords[i]
                    if len(id_list) >= target_neuron_count:
                        break

            print(f"Collected {len(id_list)} neuron IDs")

            # Step 2: build real adjacency (via CAVE queries)
            print("Building connection matrix...")
            n = len(id_list)
            real_connection_matrix = np.zeros((n, n), dtype=bool)

            # Resume from saved progress if available
            progress_file = os.path.join(visual_cortex_result_dir, f'connection_progress_{target_neuron_count}.npy')
            processed_indices, saved_matrix = load_connection_progress(progress_file)
            
            if saved_matrix is not None:
                print(f"Resuming from checkpoint: {len(processed_indices)} neurons already processed")
                real_connection_matrix = saved_matrix
            else:
                processed_indices = []

            # Throttle requests to avoid rate limiting
            request_interval = 0.5  # seconds
            
            for i, root_id in enumerate(tqdm(id_list, desc="Querying neuron connections")):
                # Skip completed indices
                if i in processed_indices:
                    continue
                    
                # Query partners via CAVE
                all_conn_ids = query_neuron_connections_with_retry(client, root_id)
                
                # Keep edges within sampled neuron set
                for conn_id in all_conn_ids:
                    if conn_id in id_to_idx:
                        j = id_to_idx[conn_id]
                        if j > i:  # Fill upper triangle only
                            real_connection_matrix[i, j] = 1
                
                # Mark index processed
                processed_indices.append(i)
                
                # Checkpoint every 100 neurons
                if len(processed_indices) % 100 == 0:
                    save_connection_progress(id_list, processed_indices, real_connection_matrix, progress_file)
                
                # Throttle requests
                time.sleep(request_interval)

            # Step 3: symmetrize adjacency
            print("Symmetrizing matrix...")
            real_connection_matrix = np.logical_or(real_connection_matrix, real_connection_matrix.T)

            # Step 4: save real adjacency and metadata
            print("Saving real connection matrix...")
            np.save(real_conn_file, real_connection_matrix)
            np.save(neuron_ids_file, np.array(id_list))
            np.save(id_to_coord_file, id_to_coord)
            print("Real connection matrix saved")

        # Real average degree (undirected)
        real_ave_deg = float(cp.sum(real_connection_matrix)) / n
        print(f"Real average degree: {real_ave_deg:.2f}")
        
        # Step 2: compute mean neuron coordinates
        print("\nStep 2: Loading coordinate data...")
        df_coords = pd.read_csv(csv_file, nrows=200000)
        
        # Accumulators for mean coordinates
        print("Computing mean neuron coordinates...")
        neuron_coords = np.zeros((n, 3), dtype=np.float32)
        neuron_coord_counts = np.zeros(n, dtype=np.int32)  # Avoid name clash with neuron_counts
        
        # Accumulate endpoint coordinates per neuron
        for i in tqdm(range(len(df_coords)), desc="Computing mean neuron coordinates"):
            pre_id = df_coords.iloc[i, 6]
            post_id = df_coords.iloc[i, 11]
            
            if pre_id in id_to_idx:
                idx = id_to_idx[pre_id]
                neuron_coords[idx] += df_coords.iloc[i, [2, 3, 4]].values.astype(np.float32)
                neuron_coord_counts[idx] += 1
            
            if post_id in id_to_idx:
                idx = id_to_idx[post_id]
                neuron_coords[idx] += df_coords.iloc[i, [7, 8, 9]].values.astype(np.float32)
                neuron_coord_counts[idx] += 1
        
        # Compute mean coordinate per neuron
        neuron_coords = neuron_coords / neuron_coord_counts[:, np.newaxis]
        
        # Move coordinates to GPU
        print("Transferring coordinates to GPU...")
        coordinates = cp.array(neuron_coords, dtype=cp.float32)
        
        # Pairwise distances on GPU
        distances = calculate_distances(coordinates)
        # Keep coordinates for downstream steps
        clear_gpu_memory()
        
        # Alpha sweep parameters
        alpha_values = np.round(np.arange(0, 2.1, 0.1), 2)  # 0.0–2.0 in steps of 0.1
        mean_distance = cp.mean(distances)  # Mean pairwise distance
        
        # Match generated network to real average degree
        print(f"\nComputing generated connectome for {target_neuron_count} neurons...")
        generate_ave_deg = real_ave_deg
        
        if target_neuron_count == 10000:
            energy_multiplier = 7.0
            prob_multiplier = 100
            bias = 0.02
        elif target_neuron_count == 20000:
            energy_multiplier = 5.0
            prob_multiplier = 80
            bias = -0.035
        elif target_neuron_count == 30000:
            energy_multiplier = 1.0
            prob_multiplier = 50
            bias = 0.6
        else:
            energy_multiplier = 1.0
            prob_multiplier = 50
            bias = 0.6
        
        w = mean_distance * generate_ave_deg * energy_multiplier  # Energy budget threshold
        
        # Reuse cached alpha sweep if available
        print(f"Checking for cached alpha results at ave_deg={int(generate_ave_deg)}...")
        existing_results = collect_all_results()
        
        if int(generate_ave_deg) in existing_results:
            print(f"Using cached alpha results for ave_deg={int(generate_ave_deg)}")
            alpha_results = existing_results[int(generate_ave_deg)]
            
            # Select best alpha by entropy
            best_alpha = max(alpha_results.items(), key=lambda x: x[1])[0]
            best_entropy = alpha_results[best_alpha]
            best_alpha = float(best_alpha)  # Cast to float
            print(f"Using cached result: best alpha={best_alpha:.2f}, entropy={best_entropy}")
        else:
            print(f"No cached alpha results for ave_deg={int(generate_ave_deg)}, computing...")
            
            # Sweep alpha values
            best_alpha = None
            best_entropy = -1
            
            for alpha in tqdm(alpha_values, desc="Finding best alpha", position=1, leave=False):
                print(f"\nProcessing alpha={alpha:.2f}...")
                
                # Deterministic sampling
                np.random.seed(RANDOM_SEED)
                cp.random.seed(RANDOM_SEED)
                
                # Compute probability matrix
                prob_matrix = calculate_connection_probabilities(distances, alpha)
                
                # Sample adjacency (prob_multiplier)
                connection_matrix = generate_connection_matrix(prob_matrix, generate_ave_deg, prob_multiplier)
                
                # Apply energy budget
                connection_matrix = get_energy_bound(connection_matrix, distances, w)
                
                # Compute entropy score
                entropy = calculate_entropy(connection_matrix)
                
                # Track best
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_alpha = alpha
                
                print(f"alpha={alpha:.2f} done, entropy={entropy}")
                
                # Free GPU buffers
                del connection_matrix, prob_matrix
                clear_gpu_memory()
            
            print(f"\nBest alpha for {target_neuron_count} neurons: {best_alpha:.2f}")
            print(f"Corresponding entropy: {best_entropy}")
            best_alpha = float(best_alpha)  # Cast to float
        
        # Generate final network at best alpha
        print(f"\nGenerating final connection matrix with best alpha={best_alpha:.2f}...")
        np.random.seed(RANDOM_SEED)
        cp.random.seed(RANDOM_SEED)
        
        best_alpha = best_alpha + bias
        final_prob_matrix = calculate_connection_probabilities(distances, best_alpha)
        final_connection_matrix = generate_connection_matrix(final_prob_matrix, generate_ave_deg, prob_multiplier)
        final_connection_matrix = get_energy_bound(final_connection_matrix, distances, w)
        
        # Metrics: generated network
        print("Computing topology metrics for generated matrix...")
        gen_topology_metrics = calculate_network_topology_metrics_gpu(final_connection_matrix)
        
        # Metrics: real network
        print("Computing topology metrics for real matrix...")
        real_topology_metrics = calculate_network_topology_metrics_gpu(real_connection_matrix)
        
        # Store results
        all_neuron_count_results[target_neuron_count] = {
            'real_metrics': real_topology_metrics,
            'gen_metrics': gen_topology_metrics,
            'best_alpha': best_alpha,
            'best_entropy': best_entropy,
            'real_ave_deg': real_ave_deg
        }
        
        print(f"Experiment for {target_neuron_count} neurons completed")
        
        # Release per-run buffers
        del distances, coordinates, final_connection_matrix, final_prob_matrix
        clear_gpu_memory()
    
    # Plot degree distributions across neuron counts
    print(f"\n{'='*60}")
    print("All neuron-count experiments done; creating split violin plot")
    print(f"{'='*60}")
    
    # Render split violin plot
    create_split_violin_plot(all_neuron_count_results, visual_cortex_result_dir)
    
    print("\nAll processing completed")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} s")
