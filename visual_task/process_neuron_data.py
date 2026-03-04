import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Define all brain subregions
REGION_LIST = [
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

# Sampling ratio
SAMPLING_RATIO = 0.001  # 0.1%

# Random seed
RANDOM_SEED = 42

# Data directory
DATA_DIR = Path(__file__).parent
MICRODATA_DIR = DATA_DIR / 'microdata'

# Output file names
COORDS_NPY = DATA_DIR / 'sampled_normalized_coordinates.npy'
REGIONS_NPY = DATA_DIR / 'sampled_region_labels.npy'

def uniform_sampling(points, target_size):
    """Randomly sample a specified number of points"""
    n_points = len(points)
    if n_points <= target_size:
        return np.arange(n_points)
    return np.random.choice(n_points, size=target_size, replace=False)

def main():
    np.random.seed(RANDOM_SEED)
    all_coords = []
    all_region_labels = []
    region_sampled_counts = {}
    total_points = 0
    sampled_points = 0

    print("\nStart processing neuronal 3D coordinate data...\n")
    for region in REGION_LIST:
        file_path = MICRODATA_DIR / f"{region}.txt"
        if not file_path.exists():
            print(f"Warning: file does not exist: {file_path}")
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
            print(f"Error while reading file {file_path}: {str(e)}")
            region_sampled_counts[region] = 0
            continue

    # Print the total number of neurons before downsampling
    print(f"\nTotal number of neurons before downsampling: {total_points}")

    if not all_coords:
        print("No data were loaded; exiting.")
        return

    # Merge all coordinates
    coords = np.vstack(all_coords)
    region_labels = np.array(all_region_labels)

    # Normalize 3D coordinates (using the maximum possible distance)
    print("\nStart coordinate normalization...")
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    max_possible_dist = np.sqrt(np.sum((max_xyz - min_xyz) ** 2))
    print(f"Maximum possible distance: {max_possible_dist:.6f}")
    print(f"Coordinate range before normalization: x=[{coords[:,0].min():.4f},{coords[:,0].max():.4f}], y=[{coords[:,1].min():.4f},{coords[:,1].max():.4f}], z=[{coords[:,2].min():.4f},{coords[:,2].max():.4f}]")
    norm_coords = coords / max_possible_dist
    print(f"Coordinate range after normalization: x=[{norm_coords[:,0].min():.4f},{norm_coords[:,0].max():.4f}], y=[{norm_coords[:,1].min():.4f},{norm_coords[:,1].max():.4f}], z=[{norm_coords[:,2].min():.4f},{norm_coords[:,2].max():.4f}]")

    # Save npy files
    np.save(COORDS_NPY, norm_coords)
    np.save(REGIONS_NPY, region_labels)
    print(f"\n✓ Total number of neurons after downsampling: {len(norm_coords)}")
    print(f"✓ 3D coordinates normalized to [0, 1] and saved as: {COORDS_NPY}")
    print(f"✓ Brain region labels saved as: {REGIONS_NPY}")

    print("\nNumber of neurons after downsampling for each brain region:")
    for region in REGION_LIST:
        print(f"- {region}: {region_sampled_counts.get(region, 0)}")

if __name__ == '__main__':
    main()