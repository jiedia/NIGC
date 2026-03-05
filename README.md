# Neuro-Informed Generative Connectome (NIGC)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Data](https://img.shields.io/badge/data-Figshare-orange.svg)
![Data](https://img.shields.io/badge/data-ScienceDB-purple.svg)

## 📖 Overview

The **Neuro-Informed Generative Connectome (NIGC)** framework is a biologically plausible computational approach demonstrating that minimal biophysical rules are sufficient for the emergence of computational intelligence at the neuronal scale. By integrating geometric scaling laws of distance-connection probability and spatial distributions of neurons, the NIGC framework generates realistic, brain-wide connectomes that exhibit both high **structural similarity** to biological neural networks and robust **functional consistency** in complex cognitive tasks (e.g., temporal and spatiotemporal processing via Echo State Networks).

This repository contains the official implementation of the NIGC framework, including modules for generating brain-like structural topologies, reproducing biological scaling laws, and performing functional validations using reservoir computing.

## ✨ Key Features

- **Structural Similarity Analysis**: Validates the geometric scaling law of distance-connection probability and compares generated connectomes against baseline biological networks (e.g., mouse visual cortex).
- **Ablation Studies**: Explores the impact of multiple parameters (e.g., varying $\alpha$ values and mean degrees) on network topology.
- **Functional Consistency Evaluation**: Embeds the generated connectomes into an Echo State Network (ESN) to execute functional tasks, analyzing spectral fingerprints and spatiotemporal trajectories.
- **Customizable Brain Regions**: Supports the integration of personalized 3D neuronal coordinates from external databases like the Blue Brain Project.

## 📁 Repository Structure

```text
NIGC/
├── Structural similarity/
│   ├── validation_of_the_mouse_connection_probability_model.py
│   ├── process_visual_cortex_data.py
│   └── neuron_count_experiment.py
├── Functional consistency/
│   ├── process_neuron_data.py
│   ├── process_data.py
│   └── arabic_digit_reservoir.py
└── visual_task/
    ├── process_neuron_data.py
    ├── process_data.py
    ├── Retina.py
    └── video_classification_reservoir.py
```

## ⚙️ Prerequisites & Data Setup

To run the scripts in this repository, you must download the appropriate datasets and place them in their respective directories. Please carefully read the instructions below to ensure a smooth setup.

### 1. Structural Similarity Module Data

Please navigate to the `Structural similarity/` directory and set up the following data:

- 🔴 **Strictly Required Raw Data:** You must download the public MICrONS consortium dataset `synapses_pni_2.csv` and place it directly in the `Structural similarity` folder.
  - **Download Link**: [synapses_pni_2.csv](https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv)
- 🟢 **Highly Recommended Cache Data:** Querying synapse and neuron counts online is extremely time-consuming. To save massive amounts of computational time, we provide a pre-computed cache.
  - **Download**: [NIGC_matrices.zip on Figshare](https://doi.org/10.6084/m9.figshare.31471300)
  - **Action**: Unzip this file and place the extracted `visual_cortex_result` (which contains `connection_progress.npy`) and `neuron_count_experiment_results` folders into the `Structural similarity` directory.
  - *Note: If you choose not to download this cache, the code will run from scratch, but it will take a significantly long time.*

### 2. Functional Consistency Module Data

Please navigate to the `Functional consistency/` directory and set up the following data:

- 🔴 **Strictly Required Microdata:** The 3D coordinates of neurons across various brain regions are mandatory for the pipeline. **Without this folder, the code will crash immediately.**
  - **Download**: [NIGC_microdata.zip on Figshare](https://doi.org/10.6084/m9.figshare.31471300)
  - **Action**: Unzip the file to extract the `microdata/` folder and place it directly in the `Functional consistency` directory.

#### 🧠 Advanced: Using Custom Brain Regions
If you wish to add or swap specific brain regions:
1. Visit the [Blue Brain Project Cell Atlas](https://bbp.epfl.ch/nexus/cell-atlas/).
2. Select your target region, check **"All cell positions"**, and download the `.txt` file.
3. Place the downloaded `.txt` file into the `microdata/` folder.
4. ⚠️ **CRITICAL**: The name of your new `.txt` file **must strictly match** the names defined in the `REGION_LIST` inside `process_neuron_data.py` and the `regions` list inside `process_data.py`.

### 3. Visual Task Module Data (Cross-modal generalization)

Please navigate to the `visual_task/` directory and set up the following data from [ScienceDB (DOI: 10.57760/sciencedb.28916)](https://doi.org/10.57760/sciencedb.28916):

- 🔴 **Strictly Required Microdata:** You must download `microdata.zip` (containing 3D coordinates for visual pathway regions from the Blue Brain Project).
  - **Action**: Unzip it to extract the `microdata/` folder and place it directly in the `visual_task` directory.
- 🟢 **Highly Recommended Cache Data:** Running the retina feature extraction from raw videos is **extremely slow**. We strongly advise using our pre-computed features to save time.
  - **Download**: `feature.zip` from ScienceDB.
  - **Action**: Create a new folder named `video_data` inside the `visual_task` directory. Unzip the downloaded features and place them into this `video_data` folder. *(If you do this, you can safely skip running `Retina.py`)*.
- 🔴 **Required Raw Data (If running from scratch):** If you choose NOT to use the feature cache and prefer to run `Retina.py` yourself:
  - **Download**: `video classification_raw_data_mp4.zip`.
  - **Action**: Unzip the raw videos and place them into the `visual_task/video_data` folder.

## 🚀 Usage

First, clone the repository and install the dependencies (see `requirements.txt` below). Note that a GPU environment configured with `cupy` is highly recommended for matrix operations.

### Running Structural Similarity Scripts
```bash
cd "Structural similarity"

# 1. Fit the geometric scaling law of distance-connection probability
python validation_of_the_mouse_connection_probability_model.py

# 2. Generate connectome and validate structural similarity
python process_visual_cortex_data.py

# 3. Reproduce the neuron count ablation experiment (from paper SI)
python neuron_count_experiment.py
```
*(Note: To run `process_visual_cortex_data.py` completely from scratch, delete the auto-generated `visual_cortex_alpha_results` folder before executing).*

### Running Functional Consistency Scripts
> ⚠️ **CRITICAL EXECUTION ORDER**: You **must** run the following three scripts in the strict sequential order listed below. Each script generates essential intermediate data files (e.g., downsampled coordinates, connection matrices) required by the subsequent step. Skipping a step or running them out of order will result in missing file errors.

```bash
cd "Functional consistency"

# 1. Process raw microdata to extract downsampled 3D neuronal coordinates
python process_neuron_data.py

# 2. Generate the corresponding connectome using the NIGC framework
python process_data.py

# 3. Embed connectome into ESN and perform functional analysis
python arabic_digit_reservoir.py
```

### Running Visual Task Scripts
> ⚠️ **CRITICAL EXECUTION ORDER**: You **must** run the following four scripts in the strict sequential order listed below. Each script relies on the intermediate files generated by the previous one. 

```bash
cd "visual_task"

# 1. Process raw microdata to extract 3D neuronal coordinates for the visual pathway
python process_neuron_data.py

# 2. Generate the corresponding connectome using the NIGC framework
python process_data.py

# 3. Extract retinal features from raw videos 
# (⚠️ EXTREMELY SLOW - You can SKIP this step if you downloaded and placed the feature.zip cache)
python Retina.py

# 4. Embed connectome into ESN and perform cross-modal video classification
python video_classification_reservoir.py
```

## ⚠️ Important Notes

- **Hardware Requirements**: Due to the heavy matrix computations involved in connectome generation, a CUDA-enabled GPU with adequate VRAM is strongly advised. 
- **Memory Management**: The scripts utilize explicit garbage collection (`gc.collect()` and `cp.get_default_memory_pool().free_all_blocks()`) to prevent OOM errors during large-scale network generation.

## 📝 Citation

If you find this repository or the NIGC framework useful in your research, please consider citing our paper:

```bibtex
@article{NIGC2026,
  title={Minimal biophysical rules are sufficient for the emergence of computational intelligence at the neuronal scale},
  author={Guanyu Wang, Liang Qi, Kunyang Li, Chenyu Tang, Xuhang Chen, Yingyan Mao, Luigi G. Occhipinti, Arokia Nathan, Ningli Wang, Yu Pan, Peter Smielewski, Ying Wang*, Hongbin Han*, Xiaoyu Guo*, Shuo Gao*},
  year={2026}
}
```
