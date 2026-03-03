import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Define list of brain region subclasses
REGION_LIST = [
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
    'TRN',
]

# Sampling ratio
SAMPLING_RATIO = 0.001  # 0.1%

# Random seed
RANDOM_SEED = 42

# Data directory
DATA_DIR = Path(__file__).parent
MICRODATA_DIR = DATA_DIR / 'microdata'

# Output file names
COORDS_NPY = DATA_DIR / 'sampled_normalized_coordinates.npy'
SAMPLED_COORDS_NPY = DATA_DIR / 'sampled_coordinates.npy'  # Downsampled coordinates (unnormalized)
REGIONS_NPY = DATA_DIR / 'sampled_region_labels.npy'

def uniform_sampling(points, target_size):
    """Sample a fixed number of points from the set without replacement; return all indices if size does not exceed target.

    Parameters
    ----------
    points : array-like, shape (n_points, ...)
        Point set array; only its length n_points is used; typically 3D coordinate array.
    target_size : int
        Target sample size, must be >= 1; when n_points <= target_size returns n_points indices.

    Returns
    -------
    np.ndarray, shape (min(n_points, target_size),)
        Indices of selected points in points; integer dtype, used to index the downsampled subset.
    """
    n_points = len(points)
    if n_points <= target_size:
        return np.arange(n_points)
    return np.random.choice(n_points, size=target_size, replace=False)

def main():
    """Main workflow: load 3D neuron coordinates per region from REGION_LIST, uniform random downsampling by SAMPLING_RATIO,
    normalize coordinates by max possible distance, save npy files (normalized coords, unnormalized coords, region labels), print stats."""
    # Fix random seed
    np.random.seed(RANDOM_SEED)
    all_coords = []
    all_region_labels = []
    region_sampled_counts = {}
    total_points = 0
    sampled_points = 0

    print("\nProcessing neuron 3D coordinate data...\n")
    for region in REGION_LIST:
        file_path = MICRODATA_DIR / f"{region}.txt"
        if not file_path.exists():
            print(f"Warning: file not found: {file_path}")
            region_sampled_counts[region] = 0
            continue
        try:
            coords = np.loadtxt(file_path, delimiter=',', usecols=(0,1,2))
            n_points = len(coords)
            total_points += n_points
            target_size = max(1, int(n_points * SAMPLING_RATIO))
            indices = uniform_sampling(coords, target_size)
            sampled_coords = coords[indices]
            all_coords.append(sampled_coords)
            all_region_labels.extend([region] * len(sampled_coords))
            sampled_points += len(sampled_coords)
            region_sampled_counts[region] = len(sampled_coords)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            region_sampled_counts[region] = 0
            continue

    print(f"\nTotal neuron count before downsampling: {total_points}")
    print(f"Random seed fixed: {RANDOM_SEED} (reproducible across runs)")

    if not all_coords:
        print("No data read; exiting.")
        return

    # Merge all coordinates
    coords = np.vstack(all_coords)
    region_labels = np.array(all_region_labels)

    # Save downsampled coordinates (unnormalized)
    np.save(SAMPLED_COORDS_NPY, coords)
    print(f"Downsampled coordinates (unnormalized) saved to: {SAMPLED_COORDS_NPY}")

    # Normalize 3D coordinates (using max possible distance)
    print("\nNormalizing coordinates...")
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    max_possible_dist = np.sqrt(np.sum((max_xyz - min_xyz) ** 2))
    print(f"Max possible distance: {max_possible_dist:.6f}")
    print(f"Coordinate range before normalization: x=[{coords[:,0].min():.4f},{coords[:,0].max():.4f}], y=[{coords[:,1].min():.4f},{coords[:,1].max():.4f}], z=[{coords[:,2].min():.4f},{coords[:,2].max():.4f}]")
    norm_coords = coords / max_possible_dist
    print(f"Coordinate range after normalization: x=[{norm_coords[:,0].min():.4f},{norm_coords[:,0].max():.4f}], y=[{norm_coords[:,1].min():.4f},{norm_coords[:,1].max():.4f}], z=[{norm_coords[:,2].min():.4f},{norm_coords[:,2].max():.4f}]")

    # Save npy files
    np.save(COORDS_NPY, norm_coords)
    np.save(REGIONS_NPY, region_labels)
    print(f"\nTotal neuron count after downsampling: {len(norm_coords)}")
    print(f"3D coordinates normalized to [0,1], saved to: {COORDS_NPY}")
    print(f"Region labels saved to: {REGIONS_NPY}")

    print("\nNeuron count per region after downsampling:")
    for region in REGION_LIST:
        print(f"- {region}: {region_sampled_counts.get(region, 0)}")

if __name__ == '__main__':
    main()