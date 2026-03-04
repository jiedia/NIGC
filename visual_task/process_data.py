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
np.random.seed(RANDOM_SEED)
cp.random.seed(RANDOM_SEED)

# Get absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
coordinates_file = os.path.join(current_dir, "sampled_normalized_coordinates.npy")  # Use sampled normalized coordinates
region_path = os.path.join(os.path.dirname(__file__), 'sampled_region_labels.npy')
region_labels = np.load(region_path)

# Brain region name order
regions = [
    'VC',
    'MC',
    'PL',
    'IL',
    'OFC',
    'PPC',
    'SC',
    'LP',
    'TRN',
    'LGN',
    'OPN',
    'DS',
]

# Count neurons per brain region
dict_region_count = {region: np.sum(region_labels == region) for region in regions}

# Compute ave_deg for each brain region; set the largest region to 50 and linearly scale others
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
    """Clear GPU memory."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def clear_cpu_memory():
    """Clear CPU memory."""
    gc.collect()

def calculate_distances(coordinates):
    """Compute pairwise Euclidean distances between neurons."""
    print("Start computing pairwise neuron distances...")
    n = coordinates.shape[0]
    distances = cp.zeros((n, n), dtype=cp.float32)
    
    # Use GPU to accelerate distance computation
    for i in tqdm(range(n), desc="Computing distance matrix"):
        diff = coordinates - coordinates[i]
        distances[i] = cp.sqrt(cp.sum(diff**2, axis=1))
    
    print("Distance computation done")
    return distances

def calculate_connection_probabilities(distances, alpha):   # Compute connection probabilities (multi-objective balanced version)
    """Compute connection probabilities (multi-objective balanced version)."""
    print(f"Start computing connection probabilities for alpha={alpha:.2f}...")
    n = distances.shape[0]
    
    # Base distance-based probability
    prob_matrix = distances**(-alpha)
    cp.fill_diagonal(prob_matrix, 0)
    prob_matrix = prob_matrix/cp.max(prob_matrix)
    target_degree_weights = cp.zeros(n, dtype=cp.float16)

    node_indices = cp.arange(n, dtype=cp.float16)
    power_law_weights = 1.0 / (node_indices + 1)**0.8
    power_law_weights = power_law_weights / cp.sum(power_law_weights) * n
    
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
            
            # Clear current block
            del density_diff_block, similarity_block
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear temporary variables
    del neighbor_density_features
    cp.get_default_memory_pool().free_all_blocks()
    
    local_threshold = cp.percentile(distances, 10)
    global_threshold = cp.percentile(distances, 50)
    block_size = 1000
    dist_block = cp.empty((block_size, block_size), dtype=cp.float16)
    sim_block = cp.empty((block_size, block_size), dtype=cp.float16)
    prob_block = cp.empty((block_size, block_size), dtype=cp.float16)
    enhancement_block = cp.empty_like(prob_block, dtype=cp.float16)

    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(0, n, block_size):
            end_j = min(j + block_size, n)

            dist_block[:end_i-i, :end_j-j] = distances[i:end_i, j:end_j]
            sim_block[:end_i-i, :end_j-j] = neighbor_density_similarity[i:end_i, j:end_j]
            prob_block[:end_i-i, :end_j-j] = prob_matrix[i:end_i, j:end_j]

            local_mask = dist_block < local_threshold
            global_mask = (dist_block >= local_threshold) & (dist_block < global_threshold)
            distant_mask = dist_block >= global_threshold

            enhancement_block[:] = 1.0
            enhancement_block[local_mask] = 1.0 + sim_block[local_mask] * 0.8
            enhancement_block[global_mask] = 1.0 + sim_block[global_mask] * 0.3
            enhancement_block[distant_mask] = 1.0 + sim_block[distant_mask] * 0.1
            prob_matrix[i:end_i, j:end_j] = prob_block[:end_i-i, :end_j-j] * enhancement_block[:end_i-i, :end_j-j]

            # Clear current block
            cp.get_default_memory_pool().free_all_blocks()

    # Clear temporary variables
    del local_threshold, global_threshold, dist_block, sim_block, prob_block, enhancement_block
    cp.get_default_memory_pool().free_all_blocks()

    connection_tendency = cp.sum(normalized_neighbor_strength, axis=1)
    mean_tendency = cp.mean(connection_tendency)
    std_tendency = cp.std(connection_tendency)
    
    offset = -0.8 * std_tendency
    gaussian_weights = cp.exp(-0.5 * ((connection_tendency - (mean_tendency + offset)) / (std_tendency * 0.8 + 1e-6))**2)
    gaussian_weights = gaussian_weights.astype(cp.float16)
    
    # Clear temporary variables
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
            
            # Clear current block
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
    
    # Clear temporary variables
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
            
            # Clear current block
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
            
            # Clear current block
            del dist_block, prob_block, distance_penalty_block
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear all temporary variables
    del normalized_neighbor_strength, neighbor_strength
    cp.get_default_memory_pool().free_all_blocks()
    
    node_connection_density = cp.sum(prob_matrix, axis=1)
    
    mean_density = cp.mean(node_connection_density)
    std_density = cp.std(node_connection_density)
    skewness = cp.mean(((node_connection_density - mean_density) / (std_density + 1e-6))**3)
    
    if skewness > 2.0:
        balance_factor = 0.5
    elif skewness < 1.0:
        balance_factor = 0.8
    else:
        balance_factor = 0.65
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
            
            # Clear current block
            del prob_block, penalty_i, penalty_j
            cp.get_default_memory_pool().free_all_blocks()
    
    # Clear temporary variables
    del node_connection_density, density_threshold, high_density_mask, penalty_factor
    del mean_density, std_density, skewness
    cp.get_default_memory_pool().free_all_blocks()
    
    # Ensure probabilities are within a valid range
    prob_matrix = cp.clip(prob_matrix, 0, 1)

    print("Connection probability computation done")
    return prob_matrix

def generate_connection_matrix(prob_matrix, ave_deg):    # Generate binary connection matrix
    """Generate binary connection matrix."""
    print("Start generating connection matrix...")
    cp.random.seed(RANDOM_SEED)
    
    # Generate connection matrix
    # Generate a [0, 1) uniform random matrix with the same shape as the probability matrix (on GPU)
    random_matrix = cp.random.rand(*prob_matrix.shape)
    
    # Compare random numbers with the corresponding probability values
    result_matrix = random_matrix < prob_matrix
    print("Connection matrix generation done")
    
    # Cast to integer type (0 or 1)
    return result_matrix.astype(cp.int8)

def calculate_entropy(connection_matrix):    # Compute information entropy
    """Compute information entropy using batched processing to reduce memory usage, including first- and second-order neighbors."""
    print("Start computing information entropy...")
    n = connection_matrix.shape[0]
    batch_size = 500  # Process 500 nodes per batch
    total_entropy = 0
    total_nodes = 0
    
    # Compute second-order neighbor matrix (full matrix squared)
    print("Computing second-order neighbor matrix...")
    second_order_matrix = cp.dot(connection_matrix, connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Computing information entropy"):
        end_idx = min(i + batch_size, n)
        # Get connection matrix for current batch
        batch_matrix = connection_matrix[i:end_idx]
        batch_second_order = second_order_matrix[i:end_idx]
        
        # Merge first- and second-order neighbors
        combined_matrix = batch_matrix + batch_second_order
        
        # Set nonzero entries to 1
        combined_matrix = (combined_matrix > 0).astype(cp.int8)
        cp.fill_diagonal(combined_matrix, 0)
        
        # Compute connection count for current batch
        row_sums = cp.sum(combined_matrix, axis=1)
        row_sums = cp.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        
        # Compute information entropy for current batch
        q_j = combined_matrix / row_sums[:, cp.newaxis]
        q_j = cp.where(q_j == 0, 1, q_j)  # Avoid log(0)
        batch_entropy = -cp.sum(q_j * cp.log(q_j), axis=1)
        
        # Accumulate entropy for current batch
        total_entropy += float(cp.sum(batch_entropy))
        total_nodes += end_idx - i
    
    # Compute average entropy
    avg_entropy = total_entropy / total_nodes if total_nodes > 0 else 0
    print("Information entropy computation done")
    return avg_entropy

def get_energy_bound(connection_matrix, distances, w):    # Compute new connection matrix under energy constraint
    """Compute a new connection matrix under an energy constraint, using batched processing while keeping GPU parallelism."""
    print("Start applying energy constraint...")
    n = connection_matrix.shape[0]
    batch_size = 500  # Process 500 nodes per batch
    new_connection_matrix = cp.zeros_like(connection_matrix)
    
    for i in tqdm(range(0, n, batch_size), desc="Applying energy constraint"):
        end_idx = min(i + batch_size, n)
        # Get connections and distances for current batch
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
    
    print("Energy constraint applied")
    return new_connection_matrix

def plot_alpha_curve(alpha_values, entropy_values, save_path):
    """Plot alpha curve for a single ave_deg."""
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, entropy_values, 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('Information Entropy')
    plt.title(f'Information Entropy vs Alpha (ave_deg={ave_deg})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_weighted_connection_matrix(conn_bin, distances, mu=0, sigma=1, reciprocal_boost=2, single_ratio=0.5, seed=None):
    """
    Efficient GPU-parallel version: generate a floating-point weighted connection matrix from a binary connection matrix, with a distance modulation factor.
    """
    print("Start generating floating-point weighted connection matrix...")
    if seed is not None:
        cp.random.seed(seed)
    n = conn_bin.shape[0]
    W_raw = cp.zeros_like(conn_bin, dtype=cp.float32)
    mask = conn_bin == 1
    W_raw[mask] = cp.random.lognormal(mu, sigma, size=int(cp.sum(mask).item()))
    print(cp.max(W_raw[mask]), cp.min(W_raw[mask]), cp.mean(W_raw[mask]))

    # Batch processing for reciprocal pairs
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
    print("Reciprocal pair processing done; start scaling unidirectional connections in batch...")

    # Batch processing for unidirectional connections
    single_mask = (conn_bin == 1) & (conn_bin.T == 0)
    single_idx = cp.where(single_mask)
    if single_idx[0].size > 0:
        W_raw[single_idx] *= single_ratio
    print("Floating-point weighted connection matrix generation done!")

    # 1. Reciprocal-pair mask (use upper triangle only to avoid duplicates)
    reciprocal_pairs = cp.array(reciprocal_pairs).T  # shape (n_reci, 2)

    # 2. Unidirectional-connection mask
    single_pairs = cp.array(single_idx).T  # shape (n_single, 2)

    # 3. Randomly assign inhibition to reciprocal and unidirectional pairs
    inhib_ratio = 0.2
    n_reci = reciprocal_pairs.shape[0]
    n_single = single_pairs.shape[0]
    n_reci_inhib = int(n_reci * inhib_ratio)
    n_single_inhib = int(n_single * inhib_ratio)

    # Inhibitory reciprocal pairs
    if n_reci_inhib > 0:
        reci_inhib_idx = cp.random.choice(n_reci, n_reci_inhib, replace=False)
        reci_inhib_pairs = reciprocal_pairs[reci_inhib_idx]
        # Assign negative weights to both (i, j) and (j, i)
        W_raw[reci_inhib_pairs[:, 0], reci_inhib_pairs[:, 1]] *= -1
        W_raw[reci_inhib_pairs[:, 1], reci_inhib_pairs[:, 0]] *= -1

    # Inhibitory unidirectional pairs
    if n_single_inhib > 0:
        single_inhib_idx = cp.random.choice(n_single, n_single_inhib, replace=False)
        single_inhib_pairs = single_pairs[single_inhib_idx]
        W_raw[single_inhib_pairs[:, 0], single_inhib_pairs[:, 1]] *= -1

    # 2. Zero-mean normalization
    mean_val = cp.mean(W_raw[mask])
    W_raw[mask] -= mean_val

    return W_raw

def main():
    # Configure GPU
    print("Initializing GPU...")
    cp.cuda.Device(0).use()
    
    # Load coordinate data
    print("Loading coordinates...")
    print(f"Loading data from {coordinates_file}...")
    coordinates = np.load(coordinates_file)
    coordinates = cp.array(coordinates, dtype=cp.float32)
    
    # Compute distance matrix
    distances = calculate_distances(coordinates)
    
    # Experimental parameters
    alpha_values = np.round(np.arange(0, 2.05, 0.05), 2)  # From 0 to 2 with step 0.05
    mean_distance = cp.mean(distances) # Compute mean distance
    
    # Main loop over alpha
    entropy_results = []
    for alpha in tqdm(alpha_values, desc="Iterating over alpha"):
        print(f"\nProcessing alpha={alpha:.2f}...")
        
        # Compute connection probabilities
        prob_matrix = calculate_connection_probabilities(distances, alpha)
        
        # Generate connection matrix
        connection_matrix = generate_connection_matrix(prob_matrix, ave_deg)
        
        # Apply energy constraint
        connection_matrix = get_energy_bound(connection_matrix, distances, mean_distance * ave_deg_vec)
        
        # Compute information entropy
        entropy = calculate_entropy(connection_matrix)
        
        # Save results
        entropy_results.append(entropy)
        print(f"alpha={alpha:.2f} information entropy={entropy}")
        
        # Clear connection matrices
        del connection_matrix, prob_matrix
        clear_gpu_memory()
    
    # Plot figure
    # plot_alpha_curve(alpha_values, entropy_results, os.path.join(current_dir, 'alpha_entropy_curve.png'))
    
    # Select best alpha
    best_idx = int(np.argmax(entropy_results))
    best_alpha = float(alpha_values[best_idx])
    print(f"\nBest alpha: {best_alpha:.2f}, maximum information entropy: {entropy_results[best_idx]}")
    
    # Use best alpha to generate a final connection matrix once
    print("\nGenerating final connection matrix with best alpha...")
    final_prob_matrix = calculate_connection_probabilities(distances, best_alpha)
    final_connection_matrix = generate_connection_matrix(final_prob_matrix, ave_deg)
    final_connection_matrix = get_energy_bound(final_connection_matrix, distances, mean_distance * ave_deg_vec)
    
    # Obtain floating-point weighted connection matrix
    final_connection_matrix = generate_weighted_connection_matrix(final_connection_matrix, distances, mu=0, sigma=0.2, 
                                                                  reciprocal_boost=1.3, single_ratio=0.7)
    
    # Plot interactive 3D neuron distribution and partial connections for AuditoryAreas
    print("Plotting interactive 3D neuron distribution and partial connections for AuditoryAreas...")

    # Extract 3D coordinates
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

    target_region = "LGN"
    line_ratio = 0.5

    if isinstance(final_connection_matrix, cp.ndarray):
        conn_mat = cp.asnumpy(final_connection_matrix)
    else:
        conn_mat = final_connection_matrix

    auditory_idx = np.where(region_labels_np == target_region)[0]

    # Only consider nonzero absolute values
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

    # Add logarithmic colorbar
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
    print(f"Partial neuron connections for brain region {target_region} are displayed ({int(line_ratio*100)}%).")
    
    # Save final connection matrix
    print("Saving final connection matrix...")
    matrix_file = os.path.join(current_dir, 'final_connection_matrix.npy')
    np.save(matrix_file, cp.asnumpy(final_connection_matrix))
    
    print("All processing completed!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} s")