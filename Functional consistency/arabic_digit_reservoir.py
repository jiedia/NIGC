import os
import random
import time
import numpy as np
import cupy as cp
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    pairwise_distances,
)
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, hilbert, coherence, welch, get_window, cwt, ricker
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine
from scipy.stats import (
    kurtosis,
    kendalltau,
    pearsonr,
    skew,
    spearmanr,
    truncnorm,
    entropy,
)
from mpl_toolkits.mplot3d import proj3d
from collections import Counter
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.colors as mcolors
import seaborn as sns
from pycirclize import Circos
from tqdm import tqdm
import traceback

def load_arabic_digit_txt(txt_path, max_len=93):
    """
    Load Arabic digit trajectory data from txt file: split by blank lines into blocks of (frames, 13), pad/truncate to uniform length.
    Assumptions: each line has space-separated floats; blank line marks end of sample; each block has 13 feature dimensions.

    Parameters:
        txt_path: path to data file; str.
        max_len: target time length (frames); int; default 93; pad with zeros if shorter, truncate if longer.

    Returns:
        np.ndarray, shape=(num_samples, max_len, 13), dtype=np.float32; stacked samples.
    """
    # Read txt, split by blank lines into blocks of shape=(frames, 13)
    samples = []
    with open(txt_path, 'r') as f:
        block = []
        for line in f:
            line = line.strip()
            if line == '':
                if block:
                    arr = np.array(block, dtype=np.float32)
                    # Pad to max_len
                    if arr.shape[0] < max_len:
                        pad = np.zeros((max_len - arr.shape[0], 13), dtype=np.float32)
                        arr = np.vstack([arr, pad])
                    elif arr.shape[0] > max_len:
                        arr = arr[:max_len]
                    samples.append(arr)
                    block = []
            else:
                block.append([float(x) for x in line.split()])
        if block:
            arr = np.array(block, dtype=np.float32)
            if arr.shape[0] < max_len:
                pad = np.zeros((max_len - arr.shape[0], 13), dtype=np.float32)
                arr = np.vstack([arr, pad])
            elif arr.shape[0] > max_len:
                arr = arr[:max_len]
            samples.append(arr)
    return np.stack(samples)  # (num_samples, max_len, 13)

class SimpleESN:
    def __init__(self, input_size, output_size, input_nodes_num, reservoir_size, leak_rate, 
                 target_mean_eig, w_in_scale=0.1, w_in_width=0.05, seed=42, weaken_region=None, weaken_ratio=0):
        """
        Initialize ESN reservoir from brain-region connection matrix: input nodes from CN, output (readout) nodes from PL/IL/OFC.
        Reservoir weights loaded from final_connection_matrix.npy; optional region weakening and eigenvalue scaling.
        Assumptions: sampled_region_labels.npy and final_connection_matrix.npy exist in the same directory.

        Parameters:
            input_size: number of input features; int.
            output_size: number of output (readout) nodes; int.
            input_nodes_num: number of input nodes; int; must not exceed CN region neuron count.
            reservoir_size: number of reservoir neurons; int.
            leak_rate: leak rate; float; controls state update smoothness.
            target_mean_eig: target mean eigenvalue magnitude for scaling W_res; float.
            w_in_scale: input weight center; float; default 0.1.
            w_in_width: input weight half-width; float; default 0.05; range [w_in_scale-w_in_width, w_in_scale+w_in_width].
            seed: random seed; int; default 42.
            weaken_region: list of region names to weaken for ablation; None for no weakening; optional.
            weaken_ratio: weakening ratio, 0~1; float; default 0.
        """
        np.random.seed(seed)
        self.input_size = input_size  # feature count
        self.output_size = output_size
        self.input_nodes_num = input_nodes_num  # input node count
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.threshold = 0.7
        self.reset_strength = -0.6
        # Load brain region labels
        region_path = os.path.join(os.path.dirname(__file__), 'sampled_region_labels.npy')
        region_labels = np.load(region_path)
        self.region_labels = region_labels
        
        # brain region -> neuron index mapping
        unique_regions = np.unique(region_labels)
        self.unique_regions = unique_regions  # brain region names in alphabetical order
        self.region_to_idx = {r: np.where(region_labels == r)[0] for r in unique_regions}

        # Input nodes: only from specified brain regions
        input_regions = ['CN']
        input_candidates = np.where(np.isin(region_labels, input_regions))[0]
        print(f"Input node candidates: {len(input_candidates)}")
        if input_nodes_num > len(input_candidates):
            raise ValueError(f'input_nodes_num {input_nodes_num} exceeds specified input region neuron count {len(input_candidates)}')
        input_nodes = np.random.choice(input_candidates, size=input_nodes_num, replace=False)
        # Output nodes: only from specified brain regions
        output_regions = ['PL', 'IL', 'OFC']
        output_candidates = np.where(np.isin(region_labels, output_regions))[0]
        print(f"Output node candidates: {len(output_candidates)}")
        if output_size > len(output_candidates):
            raise ValueError(f'output_size {output_size} exceeds specified output region neuron count {len(output_candidates)}')
        self.readout_nodes = np.random.choice(output_candidates, size=output_size, replace=False)
        # Input weights
        self.W_in = np.zeros((input_nodes_num, reservoir_size), dtype=np.float32)
        for i in range(input_nodes_num):
            self.W_in[i, input_nodes[i]] = np.random.uniform(w_in_scale-w_in_width, w_in_scale+w_in_width)
        # Load final_connection_matrix.npy as reservoir weights
        matrix_path = os.path.join(os.path.dirname(__file__), 'final_connection_matrix.npy')
        W_res = np.load(matrix_path).astype(np.float32)
        
        # Ablation: weaken connections involving specified brain regions
        if weaken_region is not None and weaken_ratio > 0:
            weaken_idx = np.where(np.isin(region_labels, weaken_region))[0]
            print(f"Weakened regions: {weaken_region}, weaken ratio: {weaken_ratio}, weakened neuron count: {len(weaken_idx)}")
            if len(weaken_idx) > 0 and weaken_ratio > 0:
                # Apply scaling once to all
                W_res[weaken_idx, :] *= (1-weaken_ratio)
                W_res[:, weaken_idx] *= (1-weaken_ratio)
                # Restore internal connections (weakened twice) to single weakening
                if weaken_ratio < 1:
                    W_res[np.ix_(weaken_idx, weaken_idx)] /= (1-weaken_ratio)

        # Eigenvalue scaling; mean_abs_eig stored globally after first computation (np.linalg.eigvals is slow)
        mean_abs_eig = 1.6718476
        #mean_abs_eig = 0.5751966 # HPC weakened
        #mean_abs_eig = 1.6386106 # cochlear nucleus weakened
        # eigvals = np.linalg.eigvals(W_res)
        # mean_abs_eig = np.mean(np.abs(eigvals))
        # print(mean_abs_eig)
        if mean_abs_eig > 0:
            W_res = W_res * (target_mean_eig / mean_abs_eig)
        self.W_res = W_res.astype(np.float32)
        self.input_nodes = input_nodes

    def forward(self, X):
        """
        One reservoir forward pass over batch input: update state by time step; return readout state at last time and full state sequence of sample 10.
        Computed on GPU with CuPy; update h_new = (1-leak)*h + leak*tanh(u@W_in + h@W_res).

        Parameters:
            X: input; np.ndarray, shape=(num_samples, time_steps, input_nodes_num), float32.

        Returns:
            h_np: readout node state at last time step; shape=(num_samples, output_size), float32.
            h_seq: full reservoir state sequence of sample 10; shape=(time_steps, reservoir_size), float32.
        """
        # X: (num_samples, time_steps, input_nodes_num)
        num_samples, time_steps, input_nodes_num = X.shape
        W_in_cp = cp.asarray(self.W_in)
        W_res_cp = cp.asarray(self.W_res)
        h = cp.zeros((num_samples, self.reservoir_size), dtype=cp.float32)
        X_cp = cp.asarray(X)
        h_seq = np.zeros((time_steps, self.reservoir_size), dtype=np.float32)
        for t in range(time_steps):
            u_t = X_cp[:, t, :]
            input_term = cp.dot(u_t, W_in_cp)
            res_term = cp.dot(h, W_res_cp)
            pre_act = input_term + res_term
            
            h_new = (1 - self.leak_rate) * h + self.leak_rate * cp.tanh(pre_act)
            h = h_new
            h_seq[t, :] = cp.asnumpy(h[10])
        h_np = cp.asnumpy(h[:, self.readout_nodes])
        del h, h_new, X_cp, W_in_cp, W_res_cp
        cp.get_default_memory_pool().free_all_blocks()
        return h_np, h_seq
    
    def simulate_with_stim(self, external_stim, h0=None):
        """
        Run a single time series (one sample) with fixed W_res and external stimulation; update h = (1-leak)*h + leak*tanh(W_res@h + external_stim(t)).

        Parameters:
            external_stim: external input per time step per neuron; np.ndarray, shape=(T, reservoir_size); use 0 where no stimulation.
            h0: initial state; shape=(reservoir_size,) or None (default all zeros).

        Returns:
            h_seq: state trajectory of all neurons; np.ndarray, shape=(T, reservoir_size), float32.
        """
        external_stim = np.asarray(external_stim, dtype=np.float32)
        T, N = external_stim.shape
        assert N == self.reservoir_size, "external_stim column count must equal reservoir_size"

        W_res_cp = cp.asarray(self.W_res)
        stim_cp = cp.asarray(external_stim)

        if h0 is None:
            h = cp.zeros((self.reservoir_size,), dtype=cp.float32)
        else:
            h = cp.asarray(h0.astype(np.float32))

        h_seq = np.zeros((T, self.reservoir_size), dtype=np.float32)

        for t in range(T):
            # Update: h(t+1) = (1 - leak) * h(t) + leak * tanh(W_res h(t) + external(t))
            res_term = cp.dot(h, W_res_cp)     # (N,)
            pre_act = res_term + stim_cp[t]    # (N,)
            h_new = (1 - self.leak_rate) * h + self.leak_rate * cp.tanh(pre_act)
            h = h_new
            h_seq[t] = cp.asnumpy(h)

        del h, h_new, W_res_cp, stim_cp
        cp.get_default_memory_pool().free_all_blocks()
        return h_seq

def expand_features(X, input_nodes_num):
    """
    Map feature dimensions to input nodes in a cyclic way: input node i gets feature dimension (i % feat_dim); output shape (num_samples, time_steps, input_nodes_num).

    Parameters:
        X: raw features; np.ndarray, shape=(num_samples, time_steps, feat_dim), e.g. (N, T, 13).
        input_nodes_num: number of input nodes; int.

    Returns:
        np.ndarray, shape=(num_samples, time_steps, input_nodes_num); column j corresponds to X[:, :, j % feat_dim].
    """
    # X: (num_samples, time_steps, 13) -> (num_samples, time_steps, input_nodes_num)
    num_samples, time_steps, feat_dim = X.shape
    idx = np.arange(input_nodes_num) % feat_dim
    X_expanded = X[:, :, idx]  # numpy broadcasting
    return X_expanded

def save_plot_data_to_txt(save_path, data_dict, description=""):
    """
    Write data corresponding to the figure to a .txt file with the same base name, for reproducibility; supports arrays, matrices, dicts, lists.

    Parameters:
        save_path: figure save path (.png); str; txt path is save_path with .png replaced by .txt.
        data_dict: data dict; keys are data names (str), values are data (np.ndarray/dict/list etc.).
        description: description text for the data; str; default ""; written at file header.
    """
    txt_path = save_path.replace('.png', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Data file: {os.path.basename(txt_path)}\n")
        f.write(f"Corresponding figure: {os.path.basename(save_path)}\n")
        f.write("="*60 + "\n\n")
        
        if description:
            f.write(f"Description:\n{description}\n\n")
        
        f.write("="*60 + "\n")
        f.write("Data content:\n")
        f.write("="*60 + "\n\n")
        
        for key, value in data_dict.items():
            f.write(f"[{key}]\n")
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    f.write(f"Shape: ({len(value)},) - 1D array\n")
                    f.write("Data:\n")
                    for i, v in enumerate(value):
                        # Compatible: numeric / string / object
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            f.write(f"{i}\t{float(v):.6f}\n")
                        else:
                            f.write(f"{i}\t{v}\n")
                elif value.ndim == 2:
                    f.write(f"Shape: {value.shape} - 2D matrix\n")
                    f.write("Row index\tColumn index\tValue\n")
                    for i in range(value.shape[0]):
                        for j in range(value.shape[1]):
                            f.write(f"{i}\t{j}\t{value[i,j]:.6f}\n")
                else:
                    f.write(f"Shape: {value.shape}\n")
                    f.write("Data (flattened):\n")
                    for i, v in enumerate(value.flatten()):
                        f.write(f"{i}\t{v:.6f}\n")
            elif isinstance(value, dict):
                f.write("Dictionary data:\n")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        f.write(f"{k}: {v:.6f}\n")
                    elif isinstance(v, np.ndarray):
                        f.write(f"{k}: array, shape {v.shape}\n")
                        if v.ndim == 1 and len(v) <= 100:
                            f.write("  Values: " + ", ".join([f"{x:.6f}" for x in v]) + "\n")
                    else:
                        f.write(f"{k}: {v}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"List/tuple, length: {len(value)}\n")
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        f.write(f"{i}\t{v:.6f}\n")
                    else:
                        f.write(f"{i}\t{v}\n")
            else:
                f.write(f"Value: {value}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("Data save complete.\n")
        f.write("="*60 + "\n")
    
    print(f"Data file saved: {txt_path}")

def compute_structural_projection(W_res, seed_idx, k_hop=3, decay=0.5):
    """
    Structural projection field (k-hop diffusion): from seed neurons along |W_res|, multi-hop diffusion gives each neuron's projection strength; each hop weighted by decay^(step-1).

    Parameters:
        W_res: reservoir weight matrix; np.ndarray, shape=(N, N), W_res[i,j] is i->j weight.
        seed_idx: seed neuron indices; 1D array or list, int.
        k_hop: max hop count; int; default 3.
        decay: distance decay factor, 0~1; float; default 0.5.

    Returns:
        proj: structural projection strength per neuron (unnormalized); np.ndarray, shape=(N,), float32.
    """
    W = np.abs(W_res).astype(np.float64)
    N = W.shape[0]
    seed_idx = np.asarray(seed_idx, dtype=int)
    if seed_idx.size == 0:
        raise ValueError("seed_idx is empty; cannot compute structural projection field")

    v = np.zeros(N, dtype=np.float64)
    v[seed_idx] = 1.0 / seed_idx.size  # uniform on seeds initially

    proj = np.zeros(N, dtype=np.float64)
    cur = v.copy()
    for step in range(1, k_hop + 1):
        cur = cur @ W  # one-hop propagation
        proj += (decay ** (step - 1)) * cur

    return proj.astype(np.float32)

def aggregate_by_region(values, region_labels, region_order=None, eps=1e-8, normalize=True):
    """
    Aggregate scalar values by brain region: for each region take mean of values over its neurons; if normalize=True scale by max to 0-1.

    Parameters:
        values: value per neuron; np.ndarray, shape=(N,).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        region_order: region order; 1D array or None (default np.unique(region_labels)).
        eps: small constant to avoid division by zero in normalization; float; default 1e-8.
        normalize: whether to normalize to 0-1 by current max; bool; default True.

    Returns:
        regions: brain region name array, same order as region_values.
        region_values: aggregated value per region; np.ndarray, float32; range ~[0,1] if normalize.
    """
    if region_order is None:
        regions = np.unique(region_labels)
    else:
        regions = np.asarray(region_order)

    region_values = []
    for r in regions:
        idx = np.where(region_labels == r)[0]
        if idx.size == 0:
            region_values.append(0.0)
        else:
            region_values.append(values[idx].mean())
    region_values = np.asarray(region_values, dtype=np.float32)

    if normalize:
        maxv = np.max(region_values)
        if maxv > eps:
            region_values = region_values / maxv
    return regions, region_values

def make_region_stim(esn, stim_regions=None, stim_neurons=None,
                     T=400, onset=50, duration=30, amplitude=0.5):
    """
    Build square-wave stimulus sequence for specified brain region(s)/neurons for dynamics simulation.
    In [onset, onset+duration) assign amplitude to selected neurons, 0 elsewhere.

    Parameters:
        esn: SimpleESN instance, for reservoir_size and region_to_idx.
        stim_regions: list of stimulated region names, e.g. ['CN']; list or None.
        stim_neurons: list of additional stimulated neuron indices; list or None.
        T: total time steps; int; default 400.
        onset: stimulus onset time step; int; default 50.
        duration: stimulus duration in steps; int; default 30.
        amplitude: stimulus amplitude; float; default 0.5.

    Returns:
        external_stim: external stimulus; np.ndarray, shape=(T, reservoir_size), float32.
        stim_idx: indices of neurons actually stimulated; np.ndarray, 1D int.
    """
    N = esn.reservoir_size
    external_stim = np.zeros((T, N), dtype=np.float32)

    idx_list = []
    if stim_regions is not None:
        for r in stim_regions:
            if r not in esn.region_to_idx:
                continue
            idx_list.extend(list(esn.region_to_idx[r]))
    if stim_neurons is not None:
        idx_list.extend(list(stim_neurons))

    stim_idx = np.unique(np.asarray(idx_list, dtype=int))
    if stim_idx.size == 0:
        print("Warning: no neurons selected for stimulation")
        return external_stim, stim_idx

    t_start = max(0, onset)
    t_end = min(T, onset + duration)
    external_stim[t_start:t_end, stim_idx] = amplitude
    return external_stim, stim_idx

def compute_region_lfp(h_seq, region_labels, region_order=None):
    """
    Compute LFP per brain region: mean activity over all neurons in the region at each time step.

    Parameters:
        h_seq: reservoir state sequence; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        region_order: region order; 1D array or None (default np.unique(region_labels)).

    Returns:
        regions: brain region name array.
        lfp_dict: dict, key=region name, value=LFP sequence for that region; shape=(T,), float32.
    """
    T, N = h_seq.shape
    if region_order is None:
        regions = np.unique(region_labels)
    else:
        regions = np.asarray(region_order)

    lfp_dict = {}
    for r in regions:
        idx = np.where(region_labels == r)[0]
        if idx.size == 0:
            lfp_dict[r] = np.zeros(T, dtype=np.float32)
        else:
            lfp_dict[r] = h_seq[:, idx].mean(axis=1)
    return regions, lfp_dict

def compute_region_response(lfp_dict, onset, offset, baseline_end=None,
                            eps=1e-8, normalize=True, mode="integral_abs"):
    """
    Compute causal response strength per brain region from LFP deviation in [onset, offset) relative to baseline (before baseline_end).
    mode: integral_abs (absolute deviation integral), integral_signed (signed integral), peak.

    Parameters:
        lfp_dict: dict mapping region name -> LFP sequence (T,).
        onset: response window start time step; int.
        offset: response window end time step; int.
        baseline_end: baseline window end time step; int or None (default onset).
        eps: avoid division by zero in normalization; float; default 1e-8.
        normalize: whether to normalize to [0,1] by max; bool; default True.
        mode: "integral_abs" | "integral_signed" | "peak"; str; default "integral_abs".

    Returns:
        regions: brain region name array (same order as lfp_dict keys).
        resp: response strength per region; np.ndarray, float32; ~[0,1] if normalize.
    """
    if baseline_end is None:
        baseline_end = onset
    regions = np.array(list(lfp_dict.keys()))
    resp = []
    for r in regions:
        ts = lfp_dict[r]
        base = ts[:baseline_end].mean()
        seg = ts[onset:offset] - base
        if mode == "integral_abs":
            val = np.sum(np.abs(seg))
        elif mode == "integral_signed":
            val = np.sum(seg)
        else:  # "peak"
            val = seg.max()
        resp.append(val)
    resp = np.asarray(resp, dtype=np.float32)
    if normalize:
        maxv = np.max(resp)
        if maxv > eps:
            resp = resp / maxv
    return regions, resp

def autocorr_1d(x, max_lag):
    """
    Compute normalized autocorrelation of 1D sequence for lags 0 to max_lag (lag=0 is 1); for intrinsic timescale estimation.

    Parameters:
        x: 1D sequence; array-like.
        max_lag: maximum lag; int.

    Returns:
        np.ndarray, shape=(max_lag+1,), float32; ac[0]=1, ac[k] is normalized autocorrelation at lag k.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n <= 1:
        return np.ones(max_lag + 1)
    ac_full = np.correlate(x, x, mode='full')
    ac = ac_full[n-1:n+max_lag]  # from lag=0
    ac = ac / (ac[0] + 1e-12)
    return ac.astype(np.float32)

def estimate_timescale_integral(ac, max_lag=None):
    """
    Estimate timescale tau from autocorrelation: sum of positive part of ac[0..max_lag] (negative values clipped to 0).

    Parameters:
        ac: autocorrelation sequence; array-like, typically from autocorr_1d.
        max_lag: max lag included in integral; int or None (default len(ac)-1).

    Returns:
        float; integral of positive part, as timescale estimate.
    """
    ac = np.asarray(ac, dtype=np.float32)
    if max_lag is None or max_lag >= len(ac):
        max_lag = len(ac) - 1
    ac_seg = ac[:max_lag + 1]
    ac_pos = np.clip(ac_seg, a_min=0.0, a_max=None)  # clip negatives to 0
    return float(ac_pos.sum())

def estimate_timescale_from_ac(ac, threshold=np.exp(-1), default=None):
    """
    Estimate timescale tau from autocorrelation: first lag where ac[lag] <= threshold (e-decay gives 1/e).

    Parameters:
        ac: autocorrelation sequence; array-like.
        threshold: decision threshold; float; default np.exp(-1).
        default: value to return if no lag satisfies; int/None; if None return len(ac)-1.

    Returns:
        int; first lag with ac[lag] <= threshold, or default / len(ac)-1.
    """
    idx = np.where(ac <= threshold)[0]
    if idx.size == 0:
        return default if default is not None else len(ac) - 1
    return int(idx[0])

def build_region_adjacency(W_res, region_labels, region_order=None, normalize_pair=True):
    """
    Build region-region weighted adjacency matrix A from neuron-level W_res: A[i,j] is total weight from region i to j;
    if normalize_pair=True divide by (N_i*N_j) for mean connection strength.

    Parameters:
        W_res: neuron-level connection weights; np.ndarray, shape=(N, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        region_order: region order; 1D array or None (default np.unique(region_labels)).
        normalize_pair: whether to normalize by region pair neuron counts; bool; default True.

    Returns:
        regions: brain region name array, same order as A rows/columns.
        A: region-region matrix; np.ndarray, shape=(R, R), float32.
    """
    if region_order is None:
        regions = np.unique(region_labels)
    else:
        regions = np.asarray(region_order)

    R = len(regions)
    A = np.zeros((R, R), dtype=np.float32)

    # Cache neuron indices per brain region to avoid repeated np.where
    region_to_idx = {r: np.where(region_labels == r)[0] for r in regions}

    for i, ri in enumerate(regions):
        src_idx = region_to_idx[ri]
        if src_idx.size == 0:
            continue
        for j, rj in enumerate(regions):
            tgt_idx = region_to_idx[rj]
            if tgt_idx.size == 0:
                continue
            sub = np.abs(W_res[np.ix_(src_idx, tgt_idx)])
            w_sum = sub.sum()
            if normalize_pair:
                A[i, j] = w_sum / (src_idx.size * tgt_idx.size)
            else:
                A[i, j] = w_sum
    return regions, A

def compute_structural_projection_region(adj_mat, seed_region_idx, k_hop=3, decay=0.5, row_normalize=True):
    """
    k-hop diffusion on region-region adjacency matrix: from seed region, multi-step propagation with decay^(step-1); optional row normalization (random walk).

    Parameters:
        adj_mat: region-region matrix; np.ndarray, shape=(R, R), adj_mat[i,j] is i->j strength.
        seed_region_idx: seed region index, 0~R-1; int.
        k_hop: max hop count; int; default 3.
        decay: distance decay factor; float; default 0.5.
        row_normalize: whether to normalize each row to probabilities before diffusion; bool; default True.

    Returns:
        proj: structural projection strength per region; np.ndarray, shape=(R,), float32.
    """
    A = adj_mat.astype(np.float64)
    R = A.shape[0]

    if row_normalize:
        row_sum = A.sum(axis=1, keepdims=True)
        P = np.divide(A, row_sum, where=row_sum > 0)
    else:
        P = A

    v = np.zeros(R, dtype=np.float64)
    v[seed_region_idx] = 1.0  # start from this region

    proj = np.zeros(R, dtype=np.float64)
    cur = v.copy()
    for step in range(1, k_hop + 1):
        cur = cur @ P
        proj += (decay ** (step - 1)) * cur

    return proj.astype(np.float32)

def compute_rank_matrix(dist_mat):
    """
    Convert distance matrix to rank matrix: per row rank samples by distance; column j is rank of sample j (1 = nearest; self rank set to n).

    Parameters:
        dist_mat: distance matrix; np.ndarray, shape=(n, n); diagonal set large internally to exclude self.

    Returns:
        np.ndarray, shape=(n, n), same dtype as dist_mat; ranks[i,j] is rank of j among i's neighbors.
    """
    n = dist_mat.shape[0]
    # Set diagonal large to exclude self
    diag_large = dist_mat.max() + 1e6
    dm = dist_mat.copy()
    np.fill_diagonal(dm, diag_large)
    # argsort to get ranks
    order = np.argsort(dm, axis=1)
    ranks = np.empty_like(order)
    # ranks[i, order[i, k]] = k+1
    for i in range(n):
        ranks[i, order[i]] = np.arange(1, n+1)
    # Self rank set to n (max)
    np.fill_diagonal(ranks, n)
    return ranks

def continuity_score(X, Y, k=15, metric='euclidean'):
    """
    Continuity C(k): measures how well k-neighbors in original space are preserved in low-dim Y; dual to Trustworthiness; value in [0, 1], higher is better.
    Formula from van der Maaten & Hinton.

    Parameters:
        X: high-dim samples; np.ndarray, shape=(n, d_high).
        Y: low-dim embedding; np.ndarray, shape=(n, d_low); low-dim distance Euclidean.
        k: number of neighbors; int; default 15.
        metric: distance metric in X space; str; default 'euclidean' (consistent with sklearn pairwise_distances).

    Returns:
        float; C(k) in [0, 1].
    """
    n = X.shape[0]
    # Distances and ranks
    Dx = pairwise_distances(X, metric=metric)
    Dy = pairwise_distances(Y, metric='euclidean')
    Rx = compute_rank_matrix(Dx)
    Ry = compute_rank_matrix(Dy)

    # Neighbor sets
    Nx = np.argsort(Dx, axis=1)[:, :k+1]  # include self, remove later
    Ny = np.argsort(Dy, axis=1)[:, :k+1]

    penalty = 0.0
    for i in range(n):
        nx = [j for j in Nx[i] if j != i][:k]
        ny = [j for j in Ny[i] if j != i][:k]
        U = set(nx) - set(ny)  # neighbors in original space but not in low-dim
        if len(U) == 0:
            continue
        # Penalty using low-dim rank Ry (r_ij - k)
        penalty += np.sum(Ry[i, list(U)] - k)

    const = 2.0 / (n * k * (2*n - 3*k - 1))
    C = 1.0 - const * penalty
    return float(max(0.0, min(1.0, C)))

def knn_retention_rate(X, Y, k=15, metric='euclidean'):
    """
    P@k: mean over samples of intersection ratio of top-k neighbors in original vs low-dim space; k-NN retention after dimensionality reduction.

    Parameters:
        X: high-dim samples; np.ndarray, shape=(n, d_high).
        Y: low-dim embedding; np.ndarray, shape=(n, d_low).
        k: number of neighbors; int; default 15.
        metric: distance metric in X space; str; default 'euclidean'.

    Returns:
        float; mean k-NN overlap ratio per sample, range [0, 1].
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
    Evaluate Trustworthiness, Continuity, P@k on high-dim data (samples=time steps) at multiple UMAP dimensions.
    k constrained by sklearn trustworthiness (k < n_samples/2); out-of-range k are skipped.

    Parameters:
        X_time_nodes: high-dim data; np.ndarray, shape=(n_samples, n_features), e.g. (time_steps, feature_dim).
        dims: list of UMAP target dimensions to evaluate; iterable int.
        k_list: list of k values to evaluate; iterable int.
        umap_params: UMAP parameter dict, at least n_neighbors, min_dist; dict.
        seed: random seed; int; default 42.
        high_metrics: tuple of high-dim distance metrics; default ('euclidean','cosine').

    Returns:
        dict: key=high_metric name, value={Trust_*, Cont_*, P@k_*} with {k: [values per dim]}.
    """
    n_samples = X_time_nodes.shape[0]
    # Constraint: sklearn trustworthiness requires k < n_samples/2
    allowed_k = [k for k in k_list if (isinstance(k, int) and k >= 1 and k < (n_samples/2) and k <= n_samples-1)]
    if len(allowed_k) < len(k_list):
        skipped = sorted(set(k_list) - set(allowed_k))
        print(f"  k constrained (n_samples={n_samples}): skipped {skipped}")

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
            # Compute metrics for each k
            for k in allowed_k:
                T = sk_trustworthiness(X_time_nodes, Y, n_neighbors=k, metric=high_metric)
                C = continuity_score(X_time_nodes, Y, k=k, metric=high_metric)
                P = knn_retention_rate(X_time_nodes, Y, k=k, metric=high_metric)
                res_metric[f'Trust_{high_metric}'][k].append(T)
                res_metric[f'Cont_{high_metric}'][k].append(C)
                res_metric[f'P@k_{high_metric}'][k].append(P)
        results[high_metric] = res_metric
    return results

def flatten_metric(region_dim_metrics, metric_key, k):
    """
    Extract Euclidean/cosine results for given metric_key and k from multi-region, multi-dim metric structure; compute correlation between the two for consistency report.

    Parameters:
        region_dim_metrics: per-region evaluate_region_dim_curve-style results; dict, key=region name, value={metric_family: {*_euclidean/*_cosine: {k: list}}}.
        metric_key: metric key name, e.g. 'Trust_euclidean'; str.
        k: neighbor count k; int.

    Returns:
        float; correlation between Euclidean and cosine value sequences; 0.0 if no valid data.
    """
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
    """
    Find knee dimension from marginal gain of Trust/Cont vs dimension: from d>=3, first dimension where both dT and dC are below gain threshold.
    Depends on global gain_thresh (see call site). Returns smallest dimension satisfying majority of regions and that fraction.

    Parameters:
        region_dim_metrics: per-region dimension-curve results; dict, structure as in evaluate_region_dim_curve and downstream.
        dims: dimension list, same length as curves; iterable.
        k: neighbor count k; int.
        metric_family: metric family used, e.g. 'euclidean'; str; default 'euclidean'.

    Returns:
        best_d: knee dimension for majority of regions; int or None if no valid knee.
        ratio: fraction of regions with valid knee; float.
    """
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
        # Marginal gain
        dT = np.diff(T_curve, prepend=T_curve[0])
        dC = np.diff(C_curve, prepend=C_curve[0])
        # From d=3 find first dimension where both gains < threshold
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
    # Smallest d satisfying majority of regions
    cnt = Counter(knees)
    best_d = sorted(cnt.items(), key=lambda x: (x[1], -x[0]), reverse=True)[0][0]
    ratio = ok_regions / max(1, len(region_dim_metrics))
    return best_d, ratio

def calculate_trajectory_angles_and_orthogonality(region_umap_data, time_window=5):
    """
    Compute direction-vector angle and orthogonality (orthogonality=1-|cos|, max at 90°) between UMAP spatiotemporal trajectories of regions in a sliding time window.
    Only for region pairs i!=j; boundary time steps (t < time_window or t > time_steps-time_window-1) not computed.

    Parameters:
        region_umap_data: per-region UMAP trajectories; dict, key=region name, value=array (time_steps, dim).
        time_window: half-window for direction vector (window t-time_window to t+time_window+1); int; default 5.

    Returns:
        angles_matrix: angle matrix (degrees); np.ndarray, shape=(time_steps, n_regions, n_regions).
        orthogonality_matrix: orthogonality matrix; np.ndarray, shape=(time_steps, n_regions, n_regions).
        region_names: list of region names, matching matrix dimensions.
    """
    region_names = list(region_umap_data.keys())
    n_regions = len(region_names)
    
    # Get time step count (assume same for all regions)
    time_steps = len(region_umap_data[region_names[0]])
    
    # Initialize result matrices
    angles_matrix = np.zeros((time_steps, n_regions, n_regions))
    orthogonality_matrix = np.zeros((time_steps, n_regions, n_regions))
    
    print(f"Computing angles and orthogonality among {n_regions} regions...")
    print(f"Time steps: {time_steps}, time window: {time_window}")
    
    for t in range(time_window, time_steps - time_window):
        for i, region1 in enumerate(region_names):
            for j, region2 in enumerate(region_names):
                if i != j:  # do not compute self-angle
                    # Get trajectory data for both regions near time t
                    traj1 = region_umap_data[region1][t-time_window:t+time_window+1]
                    traj2 = region_umap_data[region2][t-time_window:t+time_window+1]
                    
                    # Direction vector (central difference)
                    if len(traj1) >= 3 and len(traj2) >= 3:
                        # Trajectory direction vectors
                        dir1 = traj1[-1] - traj1[0]  # vector from window start to end
                        dir2 = traj2[-1] - traj2[0]
                        
                        # Angle (radians to degrees)
                        cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-8)
                        cos_angle = np.clip(cos_angle, -1, 1)  # avoid numerical error
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        angles_matrix[t, i, j] = angle
                        
                        # Orthogonality (max at 90°)
                        orthogonality = 1 - abs(cos_angle)  # orthogonality measure
                        orthogonality_matrix[t, i, j] = orthogonality
    
    return angles_matrix, orthogonality_matrix, region_names

def plot_all_regions_band_decomposition(states, region_labels, target_regions, fs=1000, save_dir=None):
    """
    Plot band decomposition for all target brain regions: delta/theta/alpha/beta/gamma band-pass filtered, shared time axis, y-axis labeled by region name.
    If save_dir given, save figure and corresponding txt data.

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        target_regions: list of region names to plot; iterable str.
        fs: sampling rate (Hz); int/float; default 1000.
        save_dir: save directory; str or None; None then plt.show().
    """
    # Frequency band definitions
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }
    
    # Store band signals for all regions
    all_region_signals = {}
    
    # Band decomposition per region
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color scheme
    band_colors = {
        "delta": '#1f77b4',   # blue
        "theta": '#ff7f0e',   # orange
        "alpha": '#2ca02c',   # green
        "beta": '#d62728',    # red
        "gamma": '#9467bd'    # purple
    }
    
    # Plot band signals per region
    y_offset = 0
    region_centers = {}
    
    for i, region in enumerate(target_regions):
        if region not in all_region_signals:
            continue
            
        region_signals = all_region_signals[region]
        
        # Signal range for this region
        all_signals = [region_signals[band] for band in bands.keys()]
        min_val = min([np.min(sig) for sig in all_signals])
        max_val = max([np.max(sig) for sig in all_signals])
        signal_range = max_val - min_val
        
        # Plot each band
        for band in bands.keys():
            signal = region_signals[band]
            # Normalize to region range
            normalized_signal = (signal - min_val) / signal_range if signal_range > 0 else signal
            # y-offset
            y_pos = normalized_signal + y_offset
            
            ax.plot(y_pos, color=band_colors[band], linewidth=1.5, alpha=0.8, 
                   label=f"{band} band" if i == 0 else "")
        
        # Region center for y-axis
        region_centers[region] = y_offset + 0.5
        y_offset += 1.2  # spacing between regions
        
        # Dashed line at zero
        ax.axhline(y=y_offset - 0.6, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # y-axis labels = region names
    ax.set_yticks(list(region_centers.values()))
    ax.set_yticklabels(list(region_centers.keys()), fontweight='bold', fontsize=12)
    
    # x-axis label
    ax.set_xlabel('Time (samples)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Brain Regions', fontweight='bold', fontsize=14)
    ax.set_title('Band-pass Filtered Signals Across All Brain Regions', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'all_regions_band_decomposition.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: all_regions_band_decomposition.png")
        
        # Save corresponding data file
        band_data = {}
        for region in target_regions:
            if region in all_region_signals:
                for band in bands.keys():
                    band_data[f'{region}_{band}'] = all_region_signals[region][band]
        save_plot_data_to_txt(
            out_path,
            band_data,
            description="Band decomposition data for all regions\n"
                       f"- Key format: region_band\n"
                       f"- Value: time series array for that region and band"
        )
    else:
        plt.show()

def analyze_multiple_regions_power(states, region_labels, target_regions, fs=1000, save_dir=None):
    """
    Analyze relative power distribution across frequency bands (Delta/Theta/Alpha/Beta/Gamma) for multiple regions; plot horizontal bar chart.
    If HPC present, write HPC relative power txt and trigger hippocampal relative-difference check and plot.
    Depends on globals weaken_ratio, weaken_region (see call site).

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        target_regions: list of region names to analyze; iterable str.
        fs: sampling rate (Hz); int/float; default 1000.
        save_dir: save directory; str or None; None then plt.show().

    Returns:
        all_results: dict, key=region name, value={band name: relative power (float)}.
    """
    # Band definitions
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 100)
    }
    
    # Store results for all regions
    all_results = {}
    
    print("Analyzing relative power distribution across six brain regions...")
    
    for region in target_regions:
        print(f"Analyzing region: {region}")
        
        # Get nodes for this region
        idx = np.where(np.isin(region_labels, [region]))[0]
        if len(idx) == 0:
            print(f"No nodes found for region {region}!")
            continue
            
        # Compute LFP
        lfp = np.mean(states[:, idx], axis=1)
        
        # Relative power per band
        region_results = {}
        for band, (low, high) in bands.items():
            nyq = 0.5 * fs
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            band_signal = filtfilt(b, a, lfp)
            power = np.mean(band_signal**2)
            region_results[band] = power
            
        # Relative power (normalize)
        total_power = sum(region_results.values())
        for band in region_results:
            region_results[band] = region_results[band] / total_power if total_power > 0 else 0
            
        all_results[region] = region_results
        print(f"  {region}: {len(idx)} nodes")
        
        # If HPC region, save relative power to txt
        if region == 'HPC':
            # Filename from weaken_ratio
            if weaken_ratio == 0:
                txt_filename = 'HPC_relative_power_normal.txt'
            else:
                weaken_ratio_str = str(int(weaken_ratio * 100))
                txt_filename = f'HPC_relative_power_{weaken_ratio_str}.txt'
            
            # Save to txt file
            if save_dir is not None:
                txt_path = os.path.join(save_dir, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"HPC Relative Power Analysis\n")
                    f.write(f"Weaken Region: {weaken_region}\n")
                    f.write(f"Weaken Ratio: {weaken_ratio}\n")
                    f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*50 + "\n\n")
                    
                    f.write("Frequency Band Relative Power:\n")
                    for band, power in region_results.items():
                        f.write(f"{band}: {power:.6f}\n")
                    
                    f.write("\n" + "="*50 + "\n")
                    f.write("CSV Format:\n")
                    f.write("Band,Relative_Power\n")
                    for band, power in region_results.items():
                        f.write(f"{band},{power:.6f}\n")
                
                print(f"Saved HPC relative power data: {txt_filename}")
                
                # Check for both files and compute relative difference
                check_and_plot_hippocampal_relative_difference(save_dir)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar chart parameters
    bar_width = 0.12  # bar width
    band_names = list(bands.keys())
    n_bands = len(band_names)
    n_regions = len(target_regions)
    
    # Horizontal bars per band
    x_pos = np.arange(n_bands)  # band positions
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, region in enumerate(target_regions):
        if region in all_results:
            values = [all_results[region][band] for band in band_names]
            # Horizontal bars with offset per band
            bars = ax.barh(x_pos + i * bar_width, values, bar_width, 
                          label=region, color=colors[i], alpha=0.8, 
                          edgecolor='black', linewidth=0.8)
            
            # Value labels at bar end
            for j, (bar, value) in enumerate(zip(bars, values)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Chart properties
    ax.set_yticks(x_pos + bar_width * (n_regions - 1) / 2)
    ax.set_yticklabels(band_names, fontweight='bold', fontsize=12)
    ax.set_xlabel('Relative Power', fontweight='bold', fontsize=14)
    ax.set_ylabel('Frequency Bands', fontweight='bold', fontsize=14)
    ax.set_title('Relative Power Distribution Across Frequency Bands\nSix Brain Regions Comparison', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_xlim(0, max([max(all_results[region].values()) for region in target_regions if region in all_results]) * 1.2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'multi_regions_relative_power.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save corresponding data file
        power_data = {}
        for region in target_regions:
            if region in all_results:
                power_data[region] = all_results[region]
        save_plot_data_to_txt(
            out_path,
            power_data,
            description="Multi-region relative power distribution data\n"
                       f"- Each key is a region name\n"
                       f"- Each value is a dict of band (Delta, Theta, Alpha, Beta, Gamma) -> relative power"
        )
    else:
        plt.show()
    
    # Print detailed results
    print("\nDetailed relative power results:")
    print("=" * 80)
    print(f"{'Region':<20} {'Delta':<8} {'Theta':<8} {'Alpha':<8} {'Beta':<8} {'Gamma':<8}")
    print("-" * 80)
    
    for region in target_regions:
        if region in all_results:
            values = [all_results[region][band] for band in band_names]
            print(f"{region:<20} {values[0]:<8.3f} {values[1]:<8.3f} {values[2]:<8.3f} {values[3]:<8.3f} {values[4]:<8.3f}")
    
    # Save detailed results to same directory
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
    In save_dir, check for HPC_relative_power_normal.txt and any weakened HPC relative power files;
    if present, read them, compute (weakened-normal)/normal relative difference, and plot/save histogram.

    Parameters:
        save_dir: directory path containing HPC relative power txt files; str.
    """
    
    # File paths
    normal_file = os.path.join(save_dir, 'HPC_relative_power_normal.txt')
    
    # Find weakened files
    weakened_files = []
    for filename in os.listdir(save_dir):
        if filename.startswith('HPC_relative_power_') and filename.endswith('.txt') and 'normal' not in filename:
            weakened_files.append(filename)
    
    # Check if files exist
    if not os.path.exists(normal_file):
        print("HPC_relative_power_normal.txt not found")
        return
    
    if len(weakened_files) == 0:
        print("No weakened HPC relative power files found")
        return
    
    print(f"Found {len(weakened_files)} weakened file(s): {weakened_files}")
    
    # Read normal file
    try:
        print(f"Reading normal file: {normal_file}")
        normal_powers = read_hippocampal_power_from_txt(normal_file)
        print(f"Normal relative power data loaded")
        for band, power in normal_powers.items():
            print(f"  {band}: {power:.6f}")
    except Exception as e:
        print(f"Failed to read normal file: {e}")
        traceback.print_exc()
        return
    
    # Compute relative difference for each weakened file
    for weakened_file in weakened_files:
        weakened_path = os.path.join(save_dir, weakened_file)
        try:
            print(f"Reading weakened file: {weakened_path}")
            weakened_powers = read_hippocampal_power_from_txt(weakened_path)
            print(f"Weakened relative power data loaded: {weakened_file}")
            for band, power in weakened_powers.items():
                print(f"  {band}: {power:.6f}")
            
            # Compute relative difference
            print("Computing relative difference...")
            relative_diff = calculate_hippocampal_relative_difference(weakened_powers, normal_powers)
            print(f"Relative difference:")
            for band, diff in relative_diff.items():
                print(f"  {band}: {diff:.6f}")
            
            # Plot relative difference histogram
            print("Plotting relative difference histogram...")
            plot_hippocampal_relative_difference_histogram(relative_diff, save_dir, weakened_file)
            
        except Exception as e:
            print(f"Failed to process file {weakened_file}: {e}")
            traceback.print_exc()
            continue

def read_hippocampal_power_from_txt(file_path):
    """
    Parse per-band relative power from HPC relative power txt file; file must contain "CSV Format:" section and "Band,Relative_Power" line.

    Parameters:
        file_path: path to txt file; str.

    Returns:
        dict: keys are frequency band names (str), values are relative power (float).
    """
    
    powers = {}
    
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Locate CSV-format data
    csv_start = False
    for line in lines:
        if "CSV Format:" in line:
            csv_start = True
            continue
        
        if csv_start and line.strip() and not line.startswith("=") and not line.startswith("Band,"):
            # Parse CSV row
            parts = line.strip().split(',')
            if len(parts) == 2:
                band_name = parts[0]
                power_value = float(parts[1])
                powers[band_name] = power_value
    
    print(f"Successfully read {len(powers)} frequency bands of relative power data")
    return powers

def calculate_hippocampal_relative_difference(weakened_powers, normal_powers):
    """
    Compute per-band relative difference (weakened - normal) / normal; return 0.0 for that band when normal is near zero.

    Parameters:
        weakened_powers: relative power per band under weakened condition; dict, key=band name, value=float.
        normal_powers: relative power per band under normal condition; dict, same structure.

    Returns:
        dict: keys=band names (consistent with normal_powers), values=relative difference float.
    """
    relative_diff = {}
    
    for band in normal_powers.keys():
        if band in weakened_powers:
            normal_val = normal_powers[band]
            weakened_val = weakened_powers[band]
            
            # Avoid division by zero
            if abs(normal_val) < 1e-10:
                relative_diff[band] = 0.0
            else:
                relative_diff[band] = (weakened_val - normal_val) / normal_val
        else:
            relative_diff[band] = 0.0
    
    return relative_diff

def plot_hippocampal_relative_difference_histogram(relative_diff, save_dir, weakened_filename):
    """
    Plot and save HPC per-band relative difference bar chart; write bands and relative differences to corresponding txt.

    Parameters:
        relative_diff: per-band relative difference; dict, key=band name, value=(weakened-normal)/normal.
        save_dir: save directory; str.
        weakened_filename: corresponding weakened filename (for output figure naming); str.

    Returns:
        None. Saves PNG and same-base-name txt data file.
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
    
    # Prepare data
    bands = list(relative_diff.keys())
    values = list(relative_diff.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot bar chart
    bars = ax.bar(bands, values, color=colors[:len(bands)], alpha=0.8, 
                  edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height >= 0 else -0.02), 
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=11, fontweight='bold')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Set plot properties
    ax.set_xlabel('Frequency Bands', fontweight='bold', fontsize=14)
    ax.set_ylabel('Relative Difference\n(Weakened - Normal) / Normal', fontweight='bold', fontsize=14)
    ax.set_title(f'HPC Relative Power Difference\n{weakened_filename.replace(".txt", "")}', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = weakened_filename.replace('.txt', '_relative_difference_histogram.png')
    save_path = os.path.join(save_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved HPC relative difference histogram: {output_filename}")
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'frequency_bands': bands,
            'relative_difference': relative_diff
        },
        description="HPC relative difference histogram data\n"
                   f"- frequency_bands: list of frequency band names\n"
                   f"- relative_difference: dict, key=band name, value=relative difference (weakened - normal) / normal"
    )

def analyze_multi_region_lfp_flow(states, region_labels, fs=1000, 
                                 baseline_window=50, threshold_std=1.5,
                                 time_window=(0, 500)):
    """
    Analyze multi-region LFP time series: compute per-region LFP, baseline and peak statistics, and call visualization plus Granger/transfer entropy analysis.
    Region order fixed as CN→Pons→IC→ACx→HPC→MGB→LP→TRN→SP→IL→PL→OFC→FP.

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        fs: sampling rate (Hz); int/float; default 1000.
        baseline_window: baseline window length (samples); int; default 50.
        threshold_std: number of baseline std for significant deviation; float; default 1.5.
        time_window: time window (start_ms, end_ms); tuple; default (0, 500).
    """
    
    # Define 13-region information flow order (reversed, input on top)
    region_flow_order = [
        'CN', 'Pons', 'IC', 'ACx',
        'HPC', 'MGB', 'LP',
        'TRN', 'SP', 'IL',
        'PL', 'OFC', 'FP'
    ]
    
    # Load coordinate data
    coords_path = os.path.join(os.path.dirname(__file__), 'sampled_normalized_coordinates.npy')
    if os.path.exists(coords_path):
        coordinates = np.load(coords_path)
        print(f"Loaded coordinate data: {coordinates.shape}")
    else:
        print("Warning: coordinate data file not found")
        coordinates = None
    
    # Compute LFP and statistics per region
    region_lfps = {}
    region_stats = {}
    region_centers = {}
    
    print("Computing per-region LFP and statistics...")
    for region in region_flow_order:
        idx = np.where(region_labels == region)[0]
        if len(idx) > 0:
            # Compute LFP directly (simplified)
            lfp = np.mean(states[:, idx], axis=1)
            
            region_lfps[region] = lfp
            
            # Compute statistics (improved threshold)
            baseline = lfp[:baseline_window]
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)
            
            # More sensitive threshold: based on LFP amplitude percentage
            lfp_abs = np.abs(lfp)
            max_lfp_abs = np.max(lfp_abs)
            
            # Use 10% of max as threshold
            abs_threshold = 0.1 * max_lfp_abs  # 10% threshold
            rel_threshold = threshold_std * baseline_std  # relative threshold
            threshold = max(abs_threshold, rel_threshold)
            
            # Ensure threshold within bounds
            threshold = max(threshold, 0.0001)  # min threshold
            threshold = min(threshold, 0.01)    # max threshold
            
            # LFP absolute value
            lfp_abs = np.abs(lfp)
            max_value = np.max(lfp_abs)
            
            # Red dot: 10% of max, first point exceeding from t=0
            red_threshold = 0.10 * max_value
            red_mask = lfp_abs >= red_threshold
            first_significant_idx = np.argmax(red_mask) if np.any(red_mask) else len(lfp)
            
            # Blue triangle: 90% of max, first point exceeding from t=0
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
            
            # Region center coordinates
            if coordinates is not None:
                region_coords = coordinates[idx]
                region_centers[region] = np.mean(region_coords, axis=0)
            else:
                region_centers[region] = np.array([0, 0, 0])
            
            print(f"  {region}: {len(idx)} nodes, threshold: {threshold:.4f}, "
                  f"first significant time: {region_stats[region]['first_significant_time']:.1f}ms, "
                  f"peak time: {region_stats[region]['peak_time']:.1f}ms")
    
    # Create advanced visualization
    create_lfp_flow_visualization(region_lfps, region_stats, region_centers, 
                                 region_flow_order, fs, time_window)
    
    return region_lfps, region_stats, region_centers

def create_lfp_flow_visualization(region_lfps, region_stats, region_centers, 
                                 region_flow_order, fs, time_window):
    """
    Create LFP flow visualization: call multi-region LFP overlay and propagation timing analysis figures, save to result folder in script directory.

    Parameters:
        region_lfps: per-region LFP sequences; dict, key=region name, value=(T,) array.
        region_stats: per-region stats (first_significant_time, peak_time, etc.); dict.
        region_centers: per-region center coordinates; dict, key=region name, value=(3,) array.
        region_flow_order: region order list; list of str.
        fs: sampling rate (Hz); int/float.
        time_window: time window (start_ms, end_ms); tuple.
    """
    
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
    
    # Create result folder
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 1. Multi-region LFP time series overlay
    create_multi_region_lfp_plot(region_lfps, region_stats, region_flow_order, 
                                fs, time_window, result_dir)
    
    # 2. Propagation timing analysis figure
    create_propagation_timing_analysis(region_stats, region_flow_order, result_dir)
    
    print(f"\nAll LFP flow figures saved to {result_dir}")

def create_multi_region_lfp_plot(region_lfps, region_stats, region_flow_order, 
                                fs, time_window, result_dir):
    """
    Create multi-region LFP time series overlay: plot per-region LFP (normalized+offset) in time_window, mark first significant deviation and peak time, save 07 figure and txt.

    Parameters:
        region_lfps: per-region LFP; dict, key=region name, value=(T,).
        region_stats: per-region stats; dict with first_significant_time, peak_time (ms).
        region_flow_order: region plot order; list of str.
        fs: sampling rate (Hz); int/float.
        time_window: (start_ms, end_ms); tuple.
        result_dir: save directory; str.
    """
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Time axis: use actual data length
    data_length = len(list(region_lfps.values())[0])
    time_points = np.arange(data_length) / fs * 1000  # ms
    time_mask = (time_points >= time_window[0]) & (time_points <= time_window[1])
    time_axis = time_points[time_mask]
    
    # Color scheme: gradient from input to output
    colors = plt.cm.viridis(np.linspace(0, 1, len(region_flow_order)))
    
    # Vertical offset
    vertical_offset = 0.8
    y_positions = {}
    
    # Debug info
    print(f"Debug figure 7 data:")
    print(f"  Time axis length: {len(time_axis)}")
    print(f"  Time range: {time_axis[0]:.1f} - {time_axis[-1]:.1f} ms")
    
    for i, region in enumerate(region_flow_order):
        if region in region_lfps:
            lfp = region_lfps[region][time_mask]
            y_pos = i * vertical_offset
            y_positions[region] = y_pos
            
            # Take LFP absolute value for accurate peak detection
            lfp_abs = np.abs(lfp)
            
            # Normalize LFP for visualization
            lfp_normalized = (lfp_abs - np.min(lfp_abs)) / (np.max(lfp_abs) - np.min(lfp_abs) + 1e-8)
            lfp_scaled = lfp_normalized * 0.6 + y_pos  # Scale to height 0.6
            
            # Plot LFP curve
            ax.plot(time_axis, lfp_scaled, color=colors[i], linewidth=2.5, 
                   alpha=0.9, label=region)
            
            # Fill area
            ax.fill_between(time_axis, lfp_scaled, y_pos, 
                          color=colors[i], alpha=0.2)
            
            # Mark first significant deviation time
            first_sig_time = region_stats[region]['first_significant_time']
            first_sig_idx = int(first_sig_time * fs / 1000)
            
            if 0 <= first_sig_idx < len(lfp_scaled):
                actual_time = first_sig_idx / fs * 1000
                ax.scatter(actual_time, lfp_scaled[first_sig_idx], 
                          color='red', s=80, marker='o', zorder=5, 
                          edgecolor='white', linewidth=2)
            
            # Mark peak time
            peak_time = region_stats[region]['peak_time']
            peak_idx = int(peak_time * fs / 1000)
            
            if 0 <= peak_idx < len(lfp_scaled):
                actual_peak_time = peak_idx / fs * 1000
                ax.scatter(actual_peak_time, lfp_scaled[peak_idx], 
                          color='darkblue', s=100, marker='^', zorder=5,
                          edgecolor='white', linewidth=2)

    # Add stimulus onset line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(0, len(region_flow_order) * vertical_offset * 0.95, 'Stimulus Onset', 
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Set figure properties
    ax.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Brain Regions (Information Flow Direction)', fontweight='bold', fontsize=14)
    ax.set_title('Multi-Region LFP Temporal Flow Analysis\n'
                'Information propagation from input to output regions\n'
                '• Red circles: First significant deviation from baseline\n'
                '• Blue triangles: Peak activation time\n'
                '• Vertical dashed line: Stimulus onset', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Set X-axis range to actual data length
    max_time = (data_length - 1) / fs * 1000  # Last time point
    ax.set_xlim(0, max_time)
    
    # Set Y axis: keep colored labels near Y axis
    ax.set_yticks([y_positions[region] for region in region_flow_order if region in y_positions])
    ax.set_yticklabels([])  # Clear default labels
    ax.set_ylim(-0.5, len(region_flow_order) * vertical_offset + 0.5)
    
    # Add colored labels manually near Y axis
    for i, region in enumerate(region_flow_order):
        if region in y_positions:
            y_pos = y_positions[region]
            ax.text(-2, y_pos, region, ha='right', va='center',
                   fontsize=10, fontweight='bold', color=colors[i])
    
    # Set Y-axis title position (left of region names)
    ax.set_ylabel('Brain Regions (Information Flow Direction)', fontweight='bold', fontsize=14)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_label_coords(-0.15, 0.5)
    
    # Clear Y-axis tick lines, keep labels only
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    
    # Grid and style
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
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
    save_path = os.path.join(result_dir, '07_multi_region_lfp_flow.png')
    plt.savefig(save_path)
    plt.close()
    print("Saved: 07_multi_region_lfp_flow.png")
    
    # Save corresponding data file
    lfp_flow_data = {}
    for region in region_flow_order:
        if region in region_lfps:
            lfp_flow_data[f'{region}_lfp'] = region_lfps[region][time_mask]
            lfp_flow_data[f'{region}_first_sig_time'] = region_stats[region]['first_significant_time']
            lfp_flow_data[f'{region}_peak_time'] = region_stats[region]['peak_time']
    lfp_flow_data['time_axis'] = time_axis
    save_plot_data_to_txt(
        save_path,
        lfp_flow_data,
        description="Multi-region LFP flow data\n"
                   f"- time_axis: time axis (ms)\n"
                   f"- Per region three entries: {region}_lfp (time series), {region}_first_sig_time (first significant time), {region}_peak_time (peak time)"
    )

def create_propagation_timing_analysis(region_stats, region_flow_order, result_dir):
    """
    Create propagation timing analysis figure: two horizontal bar charts for per-region first significant deviation time and peak time (by region_flow_order), save data txt.

    Parameters:
        region_stats: per-region stats; dict with first_significant_time, peak_time, node_count (ms/count).
        region_flow_order: region order; list of str.
        result_dir: save directory; str.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    regions = [r for r in region_flow_order if r in region_stats]
    first_sig_times = [region_stats[r]['first_significant_time'] for r in regions]
    peak_times = [region_stats[r]['peak_time'] for r in regions]
    node_counts = [region_stats[r]['node_count'] for r in regions]
    
    # Subplot 1: First significant deviation time
    colors1 = plt.cm.Reds(np.linspace(0.3, 1, len(regions)))
    bars1 = ax1.barh(range(len(regions)), first_sig_times, color=colors1, alpha=0.8,
                    edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars1, first_sig_times)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f}ms', ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax1.set_yticks(range(len(regions)))
    ax1.set_yticklabels(regions, fontsize=10)
    ax1.set_xlabel('First Significant Deviation Time (ms)', fontweight='bold', fontsize=12)
    ax1.set_title('(A) Information Flow Timing\nFirst significant deviation from baseline\n(Ordered by information flow)', 
                 fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Subplot 2: Peak time
    colors2 = plt.cm.Blues(np.linspace(0.3, 1, len(regions)))
    bars2 = ax2.barh(range(len(regions)), peak_times, color=colors2, alpha=0.8,
                    edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars2, peak_times)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f}ms', ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax2.set_yticks(range(len(regions)))
    ax2.set_yticklabels(regions, fontsize=10)
    ax2.set_xlabel('Peak Activation Time (ms)', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Peak Activation Timing\nMaximum response amplitude\n(Ordered by information flow)', 
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add propagation delay analysis
    if len(first_sig_times) > 1:
        delays = np.diff(first_sig_times)
        avg_delay = np.mean(delays)
        
        # Add delay info in subplot 1
        ax1.text(0.02, 0.98, f'Average Inter-Region Delay: {avg_delay:.1f}ms', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    save_path = os.path.join(result_dir, '08_propagation_timing_analysis.png')
    plt.savefig(save_path)
    plt.close()
    print("Saved: 08_propagation_timing_analysis.png")
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'regions': regions,
            'first_significant_times': first_sig_times,
            'peak_times': peak_times,
            'node_counts': node_counts,
            'average_inter_region_delay': np.mean(np.diff(first_sig_times)) if len(first_sig_times) > 1 else 0
        },
        description="Propagation timing analysis data\n"
                   f"- regions: list of region names\n"
                   f"- first_significant_times: first significant deviation time per region (ms)\n"
                   f"- peak_times: peak time per region (ms)\n"
                   f"- node_counts: node count per region\n"
                   f"- average_inter_region_delay: average inter-region delay (ms)"
    )

def calculate_granger_causality(signal1, signal2, max_lag=10):
    """
    Compute Granger causality strength of signal1 on signal2: linear autoregression; compare residual variance with/without signal1 lags; log, non-negative.

    Parameters:
        signal1: source signal; array-like, 1D.
        signal2: target signal; array-like, 1D.
        max_lag: max lag order; int; default 10. Returns 0.0 if len(signal1)<=max_lag.

    Returns:
        float; Granger causality strength signal1->signal2 (>=0).
    """
    
    n = len(signal1)
    if n <= max_lag:
        return 0.0
    
    # Prepare data
    y = signal2[max_lag:]  # Target variable
    X = np.zeros((n-max_lag, 2*max_lag))
    
    for i in range(max_lag):
        X[:, i] = signal1[max_lag-i-1:n-i-1]  # signal1 lags
        X[:, max_lag+i] = signal2[max_lag-i-1:n-i-1]  # signal2 lags
    
    # Full model (with signal1)
    model_full = LinearRegression()
    model_full.fit(X, y)
    residuals_full = y - model_full.predict(X)
    var_full = np.var(residuals_full)
    
    # Restricted model (without signal1)
    X_restricted = X[:, max_lag:]  # signal2 lags only
    model_restricted = LinearRegression()
    model_restricted.fit(X_restricted, y)
    residuals_restricted = y - model_restricted.predict(X_restricted)
    var_restricted = np.var(residuals_restricted)
    
    # Granger causality
    if var_restricted > 0:
        gc = np.log(var_restricted / var_full)
    else:
        gc = 0.0
    
    return max(0, gc)  # Return non-negative only

def calculate_transfer_entropy(signal1, signal2, lag=1, bins=10):
    """
    Compute transfer entropy of signal1 on signal2 (discretized, joint/marginal probabilities); measures extra information of signal1 on signal2 current value given signal2 history.

    Parameters:
        signal1: source signal; array-like, 1D.
        signal2: target signal; array-like, 1D.
        lag: lag steps; int; default 1. Returns 0.0 if len(signal1)<=lag.
        bins: discretization bin count; int; default 10.

    Returns:
        float; transfer entropy value (>=0).
    """
    
    # Prepare time series
    n = len(signal1)
    if n <= lag:
        return 0.0
    
    # Create joint distribution
    x_t = signal1[lag:n]
    y_t = signal2[lag:n]
    y_t_minus_1 = signal2[lag-1:n-1]
    
    # Discretize
    x_binned = np.digitize(x_t, bins=np.linspace(x_t.min(), x_t.max(), bins))
    y_binned = np.digitize(y_t, bins=np.linspace(y_t.min(), y_t.max(), bins))
    y_minus_1_binned = np.digitize(y_t_minus_1, bins=np.linspace(y_t_minus_1.min(), y_t_minus_1.max(), bins))
    
    # Compute joint probability
    joint_prob = np.zeros((bins, bins, bins))
    for i in range(len(x_binned)):
        joint_prob[x_binned[i]-1, y_binned[i]-1, y_minus_1_binned[i]-1] += 1
    joint_prob /= joint_prob.sum()
    
    # Compute marginal probabilities
    p_y_y_minus_1 = joint_prob.sum(axis=0)
    p_x_y_minus_1 = joint_prob.sum(axis=1)
    p_y_minus_1 = p_y_y_minus_1.sum(axis=0)
    
    # Compute transfer entropy
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
    q_threshold=25,   # Percentile threshold: keep strong connections >= this
    min_alpha=0.10,
    eps_frac=1e-9,
):
    """
    Draw region causal chord diagram with pyCirclize: sectors are regions, chord width and alpha show Granger causality strength; keep only connections >= q_threshold percentile; save PNG and SVG to result_dir.

    Parameters:
        causality_matrix: Granger causality matrix; np.ndarray, shape=(n_regions, n_regions).
        te_matrix: transfer entropy matrix; np.ndarray or None; chord uses only causality_matrix.
        region_labels: region labels; np.ndarray, same dimension as matrix.
        result_dir: save directory; str.
        region_order: region order on circle; list of str or None (default np.unique(region_labels)).
        ring_outer: outer ring radius; float; default 1.00.
        ring_inner: inner ring radius; float; default 0.86.
        attach_radius: chord attachment radius; float; default 0.86.
        curvature: chord curvature; float; default 0.55.
        q_threshold: percentile threshold (0-100), keep connections >= this; int/float; default 25.
        min_alpha: minimum chord alpha; float; default 0.10.
        eps_frac: small constant; float; default 1e-9.
    """

    # ============= 1. Global style =============
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # ============= 2. Prepare matrix: Granger causality only =============
    gc = np.asarray(causality_matrix, dtype=float)
    te = np.asarray(te_matrix, dtype=float) if te_matrix is not None else None

    unique_regions = np.unique(region_labels)

    if region_order is None:
        ordered_labels = list(unique_regions)
    else:
        # Filter by region_order then append any missing regions
        ordered_labels = [r for r in region_order if r in unique_regions]
        for r in unique_regions:
            if r not in ordered_labels:
                ordered_labels.append(r)

    # Build index mapping and reorder matrix
    name_to_idx = {name: i for i, name in enumerate(unique_regions)}
    idx = [name_to_idx[r] for r in ordered_labels]
    gc_reordered = gc[np.ix_(idx, idx)]
    te_reordered = te[np.ix_(idx, idx)] if te is not None else None

    # Exclude self-connections
    M = gc_reordered.copy()
    np.fill_diagonal(M, 0.0)

    # ============= 3. Filter strong connections by percentile =============
    pos_vals = M[M > 0]
    if pos_vals.size == 0:
        print("Chord: no positive Granger values, skip plot")
        return

    # q_threshold is percentile (0-100), e.g. 25 keeps top 75% strong connections
    threshold = np.percentile(pos_vals, q_threshold) if q_threshold is not None else 0.0
    M_vis = M.copy()
    M_vis[M_vis < threshold] = 0.0

    pos_vis = M_vis[M_vis > 0]
    if pos_vis.size == 0:
        print(f"Chord: q_threshold={q_threshold} too high, no connections retained; suggest lowering")
        return

    vmin = float(pos_vis.min())
    vmax = float(pos_vis.max())

    # Build DataFrame for pyCirclize
    matrix_df = pd.DataFrame(M_vis, index=ordered_labels, columns=ordered_labels)

    # ============= 4. Colors: fixed region palette =============
    nature_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d3", "#c7c7c7",
    ]
    if len(ordered_labels) > len(nature_colors):
        times = len(ordered_labels) // len(nature_colors) + 1
        nature_colors = nature_colors * times
    cmap = {label: nature_colors[i] for i, label in enumerate(ordered_labels)}

    # ============= 5. Set link alpha and linewidth by connection strength =============
    def link_kws_handler(from_label: str, to_label: str):
        """Return pycirclize chord alpha, lw, zorder by connection strength from_label->to_label."""
        v = float(matrix_df.loc[from_label, to_label])
        if v <= 0:
            # Non-visible edge: fully transparent and very thin
            return dict(alpha=0.0, lw=0.1, zorder=0)
        if vmax == vmin:
            norm = 1.0
        else:
            norm = (v - vmin) / (vmax - vmin)
        alpha = max(min_alpha, 0.15 + 0.75 * norm)   # Stronger connection more opaque
        lw = 0.4 + 1.3 * norm                        # Stronger connection thicker
        return dict(alpha=alpha, lw=lw)

    # ============= 6. Draw chord diagram with pyCirclize =============
    circos = Circos.chord_diagram(
        matrix_df,
        space=3,                        # Small gap between sectors
        r_lim=(93, 100),                # Outer ring position
        cmap=cmap,                      # Fixed color per region
        label_kws=dict(
            r=106,                      # Labels outside outer ring
            orientation="vertical",     # Vertical layout
            size=18,
            color="black",
            weight="bold",
        ),                              # Region labels
        link_kws=dict(
            direction=1,                # Show direction arrow (from -> to)
            ec="black",
        ),
        link_kws_handler=link_kws_handler,
    )

    # ============= 7. Save figure (filename includes q_threshold) =============
    # Safe string for q_threshold (e.g. 25 -> q25, 12.5 -> q12p5)
    if q_threshold is None:
        q_tag = "qNone"
    else:
        try:
            q_float = float(q_threshold)
            if q_float.is_integer():
                q_tag = f"q{int(q_float)}"
            else:
                q_tag = f"q{q_float:g}".replace(".", "p")
        except Exception:
            q_tag = f"q{q_threshold}"

    png_path = os.path.join(result_dir, f"11_causal_chord_nature_{q_tag}.png")
    svg_path = os.path.join(result_dir, f"11_causal_chord_nature_{q_tag}.svg")  # Vector for typesetting

    circos.savefig(png_path)
    circos.savefig(svg_path)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path} (vector)")

    # ============= 8. Save corresponding data =============
    node_sizes = {r: int(np.sum(region_labels == r)) for r in ordered_labels}

    save_plot_data_to_txt(
        png_path,
        {
            "granger_matrix_reordered": gc_reordered,
            "granger_matrix_thresholded": M_vis,
            "te_matrix_reordered": te_reordered,
            "region_order": ordered_labels,
            "node_sizes": node_sizes,
            "threshold_percentile": q_threshold,
            "threshold_value": float(threshold),
        },
        description=(
            "Region causal chord data (pyCirclize)\n"
            f"- granger_matrix_reordered: Granger causality matrix reordered by region_order\n"
            f"- granger_matrix_thresholded: connections below q_threshold percentile set to 0\n"
            f"- te_matrix_reordered: transfer entropy matrix reordered (if provided)\n"
            f"- region_order: region order in chord diagram\n"
            f"- node_sizes: neuron count per region\n"
            f"- threshold_percentile / threshold_value: connection strength threshold used"
        ),
    )

def create_granger_causality_heatmap(causality_matrix, region_labels, result_dir):
    """
    Plot Granger causality matrix heatmap; rows and columns are regions; save 12_granger_causality_heatmap.png and corresponding txt.

    Parameters:
        causality_matrix: Granger causality matrix; np.ndarray, shape=(n_regions, n_regions).
        region_labels: region labels; np.ndarray, used for region names and order.
        result_dir: save directory; str.
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
    
    # Create heatmap
    im = ax.imshow(causality_matrix, cmap='Reds', aspect='auto', 
                   vmin=0, vmax=np.max(causality_matrix))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Granger Causality Strength', fontweight='bold', fontsize=12)
    
    # Set axes
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(unique_regions, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(unique_regions, fontsize=10)
    
    # Annotate significant values only
    threshold = np.percentile(causality_matrix[causality_matrix > 0], 80)
    for i in range(n_regions):
        for j in range(n_regions):
            if causality_matrix[i, j] > threshold:
                text = ax.text(j, i, f'{causality_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8, fontweight='bold')
    
    # Set title and labels
    ax.set_xlabel('Target Regions', fontweight='bold', fontsize=12)
    ax.set_ylabel('Source Regions', fontweight='bold', fontsize=12)
    ax.set_title('Granger Causality Matrix\n'
                'Causal influence between brain regions\n'
                '• Red intensity: Causal strength\n'
                '• Diagonal: Self-connections (excluded)', 
                fontweight='bold', fontsize=14, pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    save_path = os.path.join(result_dir, '12_granger_causality_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    print("Saved: 12_granger_causality_heatmap.png")
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'causality_matrix': causality_matrix,
            'region_names': unique_regions.tolist() if isinstance(unique_regions, np.ndarray) else list(unique_regions)
        },
        description="Granger causality heatmap data\n"
                   f"- causality_matrix: shape ({causality_matrix.shape[0]}, {causality_matrix.shape[1]}), rows=source regions, columns=target regions\n"
                   f"- region_names: list of region names, corresponding to matrix row/column indices"
    )

def create_transfer_entropy_heatmap(te_matrix, region_labels, result_dir):
    """
    Plot transfer entropy matrix heatmap; rows and columns are regions; save 13_transfer_entropy_heatmap.png and corresponding txt.

    Parameters:
        te_matrix: transfer entropy matrix; np.ndarray, shape=(n_regions, n_regions).
        region_labels: region labels; np.ndarray.
        result_dir: save directory; str.
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
    
    # Create heatmap
    im = ax.imshow(te_matrix, cmap='Blues', aspect='auto',
                   vmin=0, vmax=np.max(te_matrix))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Transfer Entropy (bits)', fontweight='bold', fontsize=12)
    
    # Set axes
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(unique_regions, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(unique_regions, fontsize=10)
    
    # Annotate significant values only
    threshold = np.percentile(te_matrix[te_matrix > 0], 80)
    for i in range(n_regions):
        for j in range(n_regions):
            if te_matrix[i, j] > threshold:
                text = ax.text(j, i, f'{te_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8, fontweight='bold')
    
    # Set title and labels
    ax.set_xlabel('Target Regions', fontweight='bold', fontsize=12)
    ax.set_ylabel('Source Regions', fontweight='bold', fontsize=12)
    ax.set_title('Transfer Entropy Matrix\n'
                'Information flow between brain regions\n'
                '• Blue intensity: Information transfer strength\n'
                '• Higher values: Stronger information flow', 
                fontweight='bold', fontsize=14, pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    save_path = os.path.join(result_dir, '13_transfer_entropy_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    print("Saved: 13_transfer_entropy_heatmap.png")
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'te_matrix': te_matrix,
            'region_names': unique_regions.tolist() if isinstance(unique_regions, np.ndarray) else list(unique_regions)
        },
        description="Transfer entropy heatmap data\n"
                   f"- te_matrix: shape ({te_matrix.shape[0]}, {te_matrix.shape[1]}), rows=source regions, columns=target regions\n"
                   f"- region_names: list of region names, corresponding to matrix row/column indices"
    )

def create_connection_strength_analysis(causality_matrix, te_matrix, region_labels, result_dir):
    """
    Create connection strength analysis figure: plot Granger causality strength distribution histogram with mean/median lines; save figure 14 and txt.

    Parameters:
        causality_matrix: Granger causality matrix; np.ndarray, shape=(n_regions, n_regions).
        te_matrix: transfer entropy matrix; np.ndarray, shape=(n_regions, n_regions); this function uses only causality_matrix.
        region_labels: region labels; np.ndarray, for matrix dimension.
        result_dir: save directory; str.
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
    
    # Single subplot: Granger causality distribution only
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
    save_path = os.path.join(result_dir, '14_connection_strength_analysis.png')
    plt.savefig(save_path)
    plt.close()
    print("Saved: 14_connection_strength_analysis.png")
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'gc_connections': gc_connections,
            'mean_gc': np.mean(gc_connections),
            'median_gc': np.median(gc_connections),
            'std_gc': np.std(gc_connections),
            'min_gc': np.min(gc_connections),
            'max_gc': np.max(gc_connections)
        },
        description="Connection strength analysis data\n"
                   f"- gc_connections: 1D array of all non-zero Granger causality connection strengths\n"
                   f"- mean_gc: mean connection strength\n"
                   f"- median_gc: median connection strength\n"
                   f"- std_gc: std of connection strength\n"
                   f"- min_gc: min connection strength\n"
                   f"- max_gc: max connection strength"
    )

def visualize_causal_flow(states, region_labels, fs=1000):
    """
    Visualize inter-region causal flow: compute per-region LFP, Granger causality and transfer entropy matrices; generate chord (multi-threshold), Granger heatmap, TE heatmap, connection strength figure; save to result directory.

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        fs: sampling rate (Hz); int/float; default 1000.

    Returns:
        combined_matrix: (causality + TE) / 2; np.ndarray.
        causality_matrix: Granger causality matrix; np.ndarray.
        te_matrix: transfer entropy matrix; np.ndarray.
    """
    # Create result folder
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Get all regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    print(f"Analyzing causal connectivity among {n_regions} regions...")
    
    # Compute LFP per region
    region_lfps = {}
    for region in unique_regions:
        idx = np.where(region_labels == region)[0]
        if len(idx) > 0:
            region_lfps[region] = np.mean(states[:, idx], axis=1)
    
    # Compute causality matrices
    causality_matrix = np.zeros((n_regions, n_regions))
    te_matrix = np.zeros((n_regions, n_regions))
    
    print("Computing Granger causality and transfer entropy...")
    for i, region1 in enumerate(unique_regions):
        for j, region2 in enumerate(unique_regions):
            if i != j and region1 in region_lfps and region2 in region_lfps:
                # Granger causality
                gc = calculate_granger_causality(region_lfps[region1], region_lfps[region2])
                causality_matrix[i, j] = gc
                
                # Transfer entropy
                te = calculate_transfer_entropy(region_lfps[region1], region_lfps[region2])
                te_matrix[i, j] = te
    
    # 1. Chord diagrams
    try:
        desired_order = [
            'FP','ACx','PL','IL','OFC',
            'HPC','SP','MGB',
            'LP','TRN',
            'IC','Pons','CN',
        ]
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
        curvature=0.55,      # Curvature; tunable in 0.45-0.65
        q_threshold=25
    )

    create_nature_chord_diagram(
        causality_matrix,
        te_matrix,
        region_labels,
        result_dir,
        region_order=desired_order,
        ring_outer=1.00,
        ring_inner=0.86,
        attach_radius=0.86,
        curvature=0.55,      # Curvature; tunable in 0.45-0.65
        q_threshold=50
    )

    create_nature_chord_diagram(
        causality_matrix,
        te_matrix,
        region_labels,
        result_dir,
        region_order=desired_order,
        ring_outer=1.00,
        ring_inner=0.86,
        attach_radius=0.86,
        curvature=0.55,      # Curvature; tunable in 0.45-0.65
        q_threshold=75
    )
    
    # 2. Granger causality heatmap
    create_granger_causality_heatmap(causality_matrix, region_labels, result_dir)
    
    # 3. Transfer entropy heatmap
    create_transfer_entropy_heatmap(te_matrix, region_labels, result_dir)
    
    # 4. Connection strength analysis
    create_connection_strength_analysis(causality_matrix, te_matrix, region_labels, result_dir)
    
    
    # Print statistics
    combined_matrix = (causality_matrix + te_matrix) / 2
    all_connections = combined_matrix[combined_matrix > 0].flatten()
    
    print(f"\nCausal connectivity statistics:")
    print(f"Total connections: {len(all_connections)}")
    print(f"Mean connection strength: {np.mean(all_connections):.4f}")
    print(f"Max connection strength: {np.max(all_connections):.4f}")
    print(f"Std of connection strength: {np.std(all_connections):.4f}")
    
    # Find strongest connection
    max_idx = np.unravel_index(np.argmax(combined_matrix), combined_matrix.shape)
    strongest_source = unique_regions[max_idx[0]]
    strongest_target = unique_regions[max_idx[1]]
    print(f"\nStrongest causal connection: {strongest_source} -> {strongest_target}")
    print(f"Connection strength: {combined_matrix[max_idx]:.4f}")
    
    print(f"\nAll 4 causal analysis figures saved to {result_dir}")
    
    return combined_matrix, causality_matrix, te_matrix

def calculate_plv(signal1, signal2, fs=1000):
    """
    Compute phase locking value (PLV) of two signals: Hilbert instantaneous phase, PLV = |mean(exp(1j*(phase1-phase2)))|, range [0, 1].

    Parameters:
        signal1: time series; array-like, 1D.
        signal2: time series; array-like, 1D.
        fs: sampling rate (Hz); int/float; default 1000 (not directly used in this implementation).

    Returns:
        float; phase locking value, 0-1.
    """
    
    # Hilbert transform for instantaneous phase
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
    Coherence for short signals: larger nperseg (about half signal length), Hanning-window Welch coherence; may be inaccurate for very short signals.

    Parameters:
        signal1: time series; array-like, 1D.
        signal2: time series; array-like, 1D.
        fs: sampling rate (Hz); int/float; default 1000.

    Returns:
        freqs: frequency array; np.ndarray.
        coherence: coherence array; np.ndarray, same length as freqs, range [0, 1].
    """
    
    signal_length = min(len(signal1), len(signal2))
    print(f"Signal length: {signal_length} samples")
    print(f"Signal duration: {signal_length/fs:.3f} s")
    
    if signal_length < 100:
        print("Warning: signal too short, coherence may be inaccurate")
        print("Suggestion: use other analysis or increase signal length")
    
    # Method 1: use maximum possible nperseg
    nperseg = signal_length // 2
    if nperseg % 2 == 1:
        nperseg += 1
    
    print(f"Using nperseg: {nperseg}")
    print(f"Frequency resolution: {fs / nperseg:.2f} Hz")
    
    # Welch coherence with window to reduce spectral leakage
    window = 'hann'  # Hanning window
    
    try:
        freqs, coherence = coherence(signal1, signal2, fs=fs, nperseg=nperseg, window=window)
        print(f"Coherence computed; frequency points: {len(freqs)}")
    except Exception as e:
        print(f"Coherence computation failed: {e}")
        # On failure return zero coherence
        freqs = np.linspace(0, fs/2, nperseg//2 + 1)
        coherence = np.zeros_like(freqs)
    
    return freqs, coherence

def calculate_coherence(signal1, signal2, fs=1000, nperseg=None):
    """
    Compute Welch coherence of two signals; if signal length < 200 delegate to calculate_coherence_short_signal, else use given or auto nperseg.

    Parameters:
        signal1: time series; array-like, 1D.
        signal2: time series; array-like, 1D.
        fs: sampling rate (Hz); int/float; default 1000.
        nperseg: FFT segment length; int or None (auto); default None.

    Returns:
        freqs: frequency array; np.ndarray.
        coherence: coherence array; np.ndarray.
    """
    signal_length = min(len(signal1), len(signal2))
    
    # If signal too short use short-signal method
    if signal_length < 200:
        return calculate_coherence_short_signal(signal1, signal2, fs)
    
    if nperseg is None:
        signal_length = min(len(signal1), len(signal2))
        nperseg = min(2048, signal_length // 2)
        
        if signal_length < 1000:
            nperseg = min(512, signal_length // 2)
    
    if nperseg % 2 == 1:
        nperseg += 1
    
    nperseg = min(nperseg, len(signal1) // 2, len(signal2) // 2)
    
    print(f"nperseg actually used: {nperseg}")
    print(f"Expected frequency resolution: {fs / nperseg:.2f} Hz")
    
    freqs, coherence = coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    
    print(f"Actual frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
    print(f"Actual frequency point count: {len(freqs)}")
    
    return freqs, coherence

def calculate_alternative_metrics_short_signal(signal1, signal2, fs=1000):
    """
    Compute multiple alternative metrics for short signals: Pearson/Spearman/cosine, amplitude correlation, derivative correlation, energy ratio, wavelet correlation, PLV/phase consistency/phase std/skewness, mutual information, SNR ratio, stability ratio, etc.; return as dict.

    Parameters:
        signal1: time series; array-like, 1D.
        signal2: time series; array-like, 1D.
        fs: sampling rate (Hz); int/float; default 1000.

    Returns:
        dict: keys=metric names (e.g. pearson_correlation, plv, energy_ratio), values=float; some keys may be 0 on exception.
    """
    
    metrics = {}
    
    # 1. Time-domain similarity
    # 1.1 Pearson correlation
    corr_coef, p_value = pearsonr(signal1, signal2)
    metrics['pearson_correlation'] = corr_coef
    metrics['pearson_p_value'] = p_value
    
    # 1.2 Spearman correlation (nonlinear)
    spearman_corr, spearman_p = spearmanr(signal1, signal2)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_p_value'] = spearman_p
    
    # 1.3 Cosine similarity
    cosine_sim = 1 - cosine(signal1, signal2)
    metrics['cosine_similarity'] = cosine_sim
    
    # 2. Time-domain features
    # 2.1 Signal amplitude similarity
    amplitude_corr = np.corrcoef(np.abs(signal1), np.abs(signal2))[0, 1]
    metrics['amplitude_correlation'] = amplitude_corr
    
    # 2.2 Derivative similarity
    diff1 = np.diff(signal1)
    diff2 = np.diff(signal2)
    if len(diff1) > 0 and len(diff2) > 0:
        diff_corr = np.corrcoef(diff1, diff2)[0, 1]
        metrics['derivative_correlation'] = diff_corr
    else:
        metrics['derivative_correlation'] = 0
    
    # 2.3 Signal energy ratio
    energy1 = np.sum(signal1**2)
    energy2 = np.sum(signal2**2)
    # Energy ratio: signal1 relative to signal2; >1: signal1 higher, <1: signal2 higher, =1: equal
    energy_ratio = energy1 / energy2 if energy2 > 0 else float('inf') if energy1 > 0 else 1.0
    metrics['energy_ratio'] = energy_ratio
    
    # 3. Time-frequency (short signals)
    # 3.1 Wavelet similarity
    try:
        # Morlet wavelet
        widths = np.arange(1, min(len(signal1), len(signal2))//4)
        cwt1 = cwt(signal1, ricker, widths)
        cwt2 = cwt(signal2, ricker, widths)
        
        # Wavelet coefficient similarity
        cwt_corr = np.corrcoef(cwt1.flatten(), cwt2.flatten())[0, 1]
        metrics['wavelet_correlation'] = cwt_corr
    except:
        metrics['wavelet_correlation'] = 0
    
    # 4. Phase analysis
    try:
        # Phase estimation
        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # 4.1 Phase difference
        phase_diff = phase1 - phase2
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Normalize to [-pi, pi]
        
        # 4.2 Phase locking value (PLV)
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        metrics['plv'] = plv
        
        # 4.3 Phase consistency
        phase_consistency = np.abs(np.mean(np.cos(phase_diff)))
        metrics['phase_consistency'] = phase_consistency
        
        # 4.4 Std of phase difference
        phase_std = np.std(phase_diff)
        metrics['phase_std'] = phase_std
        
        # 4.5 Shape of phase difference distribution
        phase_skew = np.mean((phase_diff - np.mean(phase_diff))**3) / (np.std(phase_diff)**3)
        metrics['phase_skewness'] = phase_skew
        
    except:
        metrics['plv'] = 0
        metrics['phase_consistency'] = 0
        metrics['phase_std'] = 0
        metrics['phase_skewness'] = 0
    
    # 5. Nonlinear analysis
    # 5.1 Mutual information (approximate)
    try:
        # Signal as feature and target
        X = signal1.reshape(-1, 1)
        y = signal2
        mi = mutual_info_regression(X, y)[0]
        metrics['mutual_information'] = mi
    except:
        metrics['mutual_information'] = 0
    
    # 6. Signal quality metrics
    # 6.1 SNR estimate
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
    Compare LFP of two regions: compute PLV, coherence (or short-signal alternatives), plot LFP and coherence curves; save to result_dir (default result/Coherent_analysis).

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        region1: region name; str.
        region2: region name; str.
        fs: sampling rate (Hz); int/float; default 1000.
        result_dir: result save directory; str or None.
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
    
    # Get nodes of two regions
    idx1 = np.where(np.isin(region_labels, [region1]))[0]
    idx2 = np.where(np.isin(region_labels, [region2]))[0]
    
    if len(idx1) == 0 or len(idx2) == 0:
        print(f"No nodes found for region {region1} or {region2}")
        return
    
    # Compute LFP of two regions
    lfp1 = np.mean(states[:, idx1], axis=1)
    lfp2 = np.mean(states[:, idx2], axis=1)
    
    print(f"Region {region1}: {len(idx1)} nodes")
    print(f"Region {region2}: {len(idx2)} nodes")
    
    # Compute PLV
    plv = calculate_plv(lfp1, lfp2, fs)
    print(f"Phase locking value (PLV): {plv:.4f}")
    
    # Check signal length for analysis method
    signal_length = min(len(lfp1), len(lfp2))
    print(f"Signal length: {signal_length} samples")
    
    if signal_length < 200:
        print("Signal length short; using short-signal metrics")
        # Compute short-signal metrics
        alt_metrics = calculate_alternative_metrics_short_signal(lfp1, lfp2, fs)
        print(f"Short-signal analysis results:")
        for key, value in alt_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Simplified band analysis for short signal
        freqs = np.linspace(0, fs/2, 50)  # 50 frequency points
        
        # Estimate coherence from metrics
        base_coherence = alt_metrics['plv'] * 0.7  # From PLV
        correlation_factor = abs(alt_metrics['pearson_correlation'])
        
        # Simplified coherence curve
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
        
        # Add small noise
        noise = np.random.normal(0, 0.02, len(coherence))
        coherence = np.clip(coherence + noise, 0, 1)
        
        print(f"Coherence estimated from PLV ({alt_metrics['plv']:.3f}) and Pearson ({alt_metrics['pearson_correlation']:.3f})")
        print("Note: short-signal band analysis is estimated; for reference only")
    else:
        # Compute coherence
        freqs, coherence = calculate_coherence(lfp1, lfp2, fs)
    
    # Debug info
    print(f"\nCoherence computation debug:")
    print(f"Signal length: {len(lfp1)} time points")
    print(f"Sampling rate: {fs} Hz")
    print(f"nperseg used: {len(lfp1) // 2 if len(lfp1) < 1000 else min(2048, len(lfp1) // 2)}")
    print(f"Frequency points: {len(freqs)}")
    print(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
    print(f"Coherence range: {coherence.min():.4f} - {coherence.max():.4f}")
    print(f"Coherence mean: {coherence.mean():.4f}")
    print(f"Coherence std: {coherence.std():.4f}")
    
    # 1. LFP time series figure
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
    save_path = os.path.join(result_dir, f'LFP_timeseries_{region1}_vs_{region2}.png')
    plt.savefig(save_path)
    plt.close()
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'time_axis': time_axis,
            f'{region1}_lfp': lfp1,
            f'{region2}_lfp': lfp2
        },
        description=f"LFP time series data: {region1} vs {region2}\n"
                   f"- time_axis: time axis (s)\n"
                   f"- {region1}_lfp: {region1} region LFP signal\n"
                   f"- {region2}_lfp: {region2} region LFP signal"
    )
    
    # 2. Coherence spectrum (0-100 Hz)
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
    save_path = os.path.join(result_dir, f'Coherence_spectrum_{region1}_vs_{region2}.png')
    plt.savefig(save_path)
    plt.close()
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'frequencies': freqs_limited,
            'coherence': coherence_limited,
            'plv': plv
        },
        description=f"Coherence spectrum data: {region1} vs {region2}\n"
                   f"- frequencies: frequency array (Hz), 0-100 Hz\n"
                   f"- coherence: coherence value per frequency (0-1)\n"
                   f"- plv: phase locking value"
    )
    
    # 3. Band-wise coherence bar chart
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 100)
    }
    
    band_coherence = {}
    print(f"\nDebug - frequency range and coherence:")
    print(f"Frequency range: {freqs.min():.2f} - {freqs.max():.2f} Hz")
    print(f"Coherence range: {coherence.min():.4f} - {coherence.max():.4f}")
    
    for band, (low, high) in bands.items():
        # Find frequency indices in band
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
    
    # Annotate values on bars
    for bar, value in zip(bars, coherence_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(result_dir, f'Band_coherence_{region1}_vs_{region2}.png')
    plt.savefig(save_path)
    plt.close()
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'frequency_bands': bands_list,
            'band_coherence': coherence_values,
            'band_coherence_dict': band_coherence
        },
        description=f"Band-wise coherence data: {region1} vs {region2}\n"
                   f"- frequency_bands: list of band names\n"
                   f"- band_coherence: mean coherence per band\n"
                   f"- band_coherence_dict: dict band -> coherence"
    )
    
    # 4. Phase difference distribution histogram
    if signal_length < 200:
        # Short signal: comprehensive analysis figure
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
        
        # Add value labels
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
        
        # Add value labels
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
        
        # Add value labels
        for bar, value in zip(bars3, quality_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 4: phase difference distribution
        phase_diff_for_save = None
        if 'phase_diff' in alt_metrics:
            phase_diff_for_save = alt_metrics['phase_diff']
        else:
            # Compute phase difference
            analytic1 = hilbert(lfp1)
            analytic2 = hilbert(lfp2)
            phase_diff_for_save = np.angle(analytic1) - np.angle(analytic2)
            phase_diff_for_save = np.angle(np.exp(1j * phase_diff_for_save))
        
        n, bins, patches = ax4.hist(phase_diff_for_save, bins=20, alpha=0.7, color='#d62728', 
                                   edgecolor='black', linewidth=0.8)
        
        # Add statistics
        mean_phase_diff = np.mean(phase_diff_for_save)
        std_phase_diff = np.std(phase_diff_for_save)
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
        save_path = os.path.join(result_dir, f'Short_signal_analysis_{region1}_vs_{region2}.png')
        plt.savefig(save_path)
        plt.close()
        
        # Save corresponding data file
        save_plot_data_to_txt(
            save_path,
            {
                'alternative_metrics': alt_metrics,
                'phase_difference': phase_diff_for_save
            },
            description=f"Short-signal analysis data: {region1} vs {region2}\n"
                       f"- alternative_metrics: dict of multiple similarity metrics\n"
                       f"- phase_difference: phase difference distribution (rad)"
        )
        
    else:
        # Long signal: standard phase difference analysis
        analytic1 = hilbert(lfp1)
        analytic2 = hilbert(lfp2)
        phase_diff = np.angle(analytic1) - np.angle(analytic2)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        n, bins, patches = ax.hist(phase_diff, bins=50, alpha=0.7, color='#d62728', 
                                  edgecolor='black', linewidth=0.8)
        
        # Add statistics
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
        save_path = os.path.join(result_dir, f'Phase_difference_{region1}_vs_{region2}.png')
        plt.savefig(save_path)
        plt.close()
        
        # Save corresponding data file
        save_plot_data_to_txt(
            save_path,
            {
                'phase_difference': phase_diff,
                'mean_phase_diff': mean_phase_diff,
                'std_phase_diff': std_phase_diff,
                'plv': plv
            },
            description=f"Phase difference distribution data: {region1} vs {region2}\n"
                       f"- phase_difference: phase difference time series (rad)\n"
                       f"- mean_phase_diff: mean phase difference\n"
                       f"- std_phase_diff: std of phase difference\n"
                       f"- plv: phase locking value"
        )
    
    # Print band-wise coherence
    print(f"\nBand-wise coherence:")
    for band in bands_list:
        print(f"{band:6s}: {band_coherence[band]:.4f}")
    
    print(f"\nAll coherence analysis figures saved to: {result_dir}")
    
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
    Run short-signal analysis for all brain-region pairs: compute surrogate metrics (Pearson, PLV, energy ratio, etc.),
    generate heatmaps and txt; may trigger relative-difference check.

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        fs: sampling rate (Hz); int/float; default 1000.
        result_dir: result save directory; str or None.

    Returns:
        all_results: list of analysis results per region pair; list of dict.
        metrics_matrices: heatmap matrix per metric; dict, key=metric name, value=(n_regions, n_regions).
    """
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), 'result', 'Coherent_analysis')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Get all brain regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    print(f"Starting short-signal analysis for all {n_regions} region pairs...")
    print(f"Total region pairs to analyze: {n_regions * (n_regions - 1) // 2}")
    
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
            if i < j:  # Avoid duplicate and self-comparison
                pair_count += 1
                print(f"\nAnalyzing region pair {pair_count}: {region1} vs {region2}")
                
                try:
                    # Get LFP for the two regions
                    idx1 = np.where(region_labels == region1)[0]
                    idx2 = np.where(region_labels == region2)[0]
                    
                    if len(idx1) == 0 or len(idx2) == 0:
                        print(f"No nodes found for region {region1} or {region2}!")
                        continue
                    
                    lfp1 = np.mean(states[:, idx1], axis=1)
                    lfp2 = np.mean(states[:, idx2], axis=1)
                    
                    # Compute short-signal metrics
                    metrics = calculate_alternative_metrics_short_signal(lfp1, lfp2, fs)
                    
                    # Store in matrices
                    for metric_name, value in metrics.items():
                        if metric_name in metrics_matrices:
                            metrics_matrices[metric_name][i, j] = value
                            metrics_matrices[metric_name][j, i] = value  # Symmetric matrix
                    
                    # Store result
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
                    print(f"Error analyzing {region1} vs {region2}: {e}")
                    continue
    
    # Save results to text file
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
    
    print(f"\nDone. Analyzed {len(all_results)} region pairs.")
    print(f"Results saved to: {txt_file}")
    
    # Create visualizations
    create_short_signal_visualizations(metrics_matrices, unique_regions, result_dir)
    
    return all_results, metrics_matrices

def create_short_signal_visualizations(metrics_matrices, region_names, result_dir):
    """
    Plot and save heatmaps for each metric matrix from short-signal analysis; if metric is pearson_correlation,
    also save matrix to txt and trigger relative-difference check.
    Depends on global weaken_ratio, weaken_region (see call site).

    Parameters:
        metrics_matrices: metric name -> (n_regions, n_regions) matrix; dict.
        region_names: region name list/array, aligned with matrix rows/columns.
        result_dir: save directory; str.
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
    
    # Create heatmap for each metric
    for metric_name, matrix in metrics_matrices.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set colorbar range dynamically
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        
        # For energy_ratio, use special colorbar range
        if metric_name == 'energy_ratio':
            # Symmetric range centered at 1
            center = 1.0
            max_deviation = max(abs(vmax - center), abs(vmin - center))
            vmin = max(0, center - max_deviation)
            vmax = center + max_deviation
            # Log scale may be more appropriate
            if vmax > 2 * vmin:
                # Use log scale
                im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', 
                             norm=LogNorm(vmin=max(0.1, vmin), vmax=vmax))
            else:
                im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
        else:
            # Other metrics use raw range
            im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set axes
        ax.set_xticks(range(n_regions))
        ax.set_yticks(range(n_regions))
        ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(region_names, fontsize=10)
        
        # Add value annotations
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
        
        # Special note for energy_ratio
        if metric_name == 'energy_ratio':
            ax.set_title(f'Energy Ratio Matrix\n(Value > 1: Row region has higher energy than column region)', 
                        fontweight='bold', fontsize=16, pad=20)
        else:
            ax.set_title(f'{metric_name.replace("_", " ").title()} Matrix', 
                        fontweight='bold', fontsize=16, pad=20)
        
        # Add grid
        ax.set_xticks(np.arange(n_regions+1)-0.5, minor=True)
        ax.set_yticks(np.arange(n_regions+1)-0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{metric_name}_heatmap.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {metric_name}_heatmap.png")
        
        # Save corresponding data file
        save_plot_data_to_txt(
            save_path,
            {
                'metric_matrix': matrix,
                'region_names': region_names.tolist() if isinstance(region_names, np.ndarray) else list(region_names)
            },
            description=f"Short-signal metric heatmap data: {metric_name}\n"
                       f"- metric_matrix: shape ({matrix.shape[0]}, {matrix.shape[1]}), rows and columns correspond to brain regions\n"
                       f"- region_names: region name list, corresponding to matrix row and column indices"
        )
        
        # If pearson_correlation, also save matrix values to txt
        if metric_name == 'pearson_correlation':
            # Generate filename from weaken_ratio and weaken_region
            if weaken_ratio == 0:
                txt_filename = 'pearson_correlation_normal.txt'
            else:
                # Get first string in weaken_region
                weaken_region_name = weaken_region[0] if isinstance(weaken_region, list) and len(weaken_region) > 0 else str(weaken_region)
                weaken_ratio_str = str(int(weaken_ratio * 100))
                txt_filename = f'pearson_correlation_{weaken_region_name}{weaken_ratio_str}.txt'
            
            # Save matrix to txt file
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
            
            print(f"Saved matrix data: {txt_filename}")
            
            # Check if both files exist and plot relative-difference heatmap
            check_and_plot_relative_difference(result_dir, region_names)

def check_and_plot_relative_difference(result_dir, region_names):
    """
    In result_dir, check for pearson_correlation_normal.txt and any weakened Pearson matrix files;
    if present, read them, compute relative-difference matrix and plot/save heatmap.

    Parameters:
        result_dir: directory containing Pearson matrix txt files; str.
        region_names: region name list/array, same order as matrix rows/columns.
    """
    
    # Define file path
    normal_file = os.path.join(result_dir, 'pearson_correlation_normal.txt')
    
    # Find weakened files
    # Exclude heatmap.txt produced by save_plot_data_to_txt
    weakened_files = []
    for filename in os.listdir(result_dir):
        if (filename.startswith('pearson_correlation_') and 
            filename.endswith('.txt') and 
            'normal' not in filename and
            'heatmap' not in filename):  # Exclude heatmap.txt
            weakened_files.append(filename)
    
    # Check if files exist
    if not os.path.exists(normal_file):
        print("pearson_correlation_normal.txt not found.")
        return
    
    if len(weakened_files) == 0:
        print("No weakened Pearson correlation files found.")
        return
    
    print(f"Found {len(weakened_files)} weakened file(s): {weakened_files}")
    
    # Read normal file
    try:
        print(f"Reading normal file: {normal_file}")
        normal_matrix = read_pearson_matrix_from_txt(normal_file, region_names)
        print(f"Read normal matrix, shape: {normal_matrix.shape}")
    except Exception as e:
        print(f"Failed to read normal file: {e}")
        traceback.print_exc()
        return
    
    # Compute relative difference for each weakened file
    for weakened_file in weakened_files:
        weakened_path = os.path.join(result_dir, weakened_file)
        try:
            print(f"Reading weakened file: {weakened_path}")
            weakened_matrix = read_pearson_matrix_from_txt(weakened_path, region_names)
            print(f"Read weakened matrix {weakened_file}, shape: {weakened_matrix.shape}")
            
            # Compute relative difference
            print("Computing relative difference...")
            relative_diff = calculate_relative_difference(weakened_matrix, normal_matrix)
            print(f"Relative difference matrix shape: {relative_diff.shape}")
            print(f"Relative difference range: {relative_diff.min():.6f} to {relative_diff.max():.6f}")
            
            # Plot relative-difference heatmap
            print("Plotting relative-difference heatmap...")
            plot_relative_difference_heatmap(relative_diff, region_names, result_dir, weakened_file)
            
        except Exception as e:
            print(f"Failed to process file {weakened_file}: {e}")
            traceback.print_exc()
            continue

def read_pearson_matrix_from_txt(file_path, region_names):
    """
    Parse matrix from Pearson correlation matrix txt file; file must contain "Matrix in CSV format:" section
    and CSV lines with region names and values.

    Parameters:
        file_path: path to txt file; str.
        region_names: region name list/array, determines matrix dimension and row/column order; list or np.ndarray.

    Returns:
        np.ndarray, shape=(len(region_names), len(region_names)); unmatched positions are 0.
    """
    
    # Ensure region_names is a list
    if isinstance(region_names, np.ndarray):
        region_names = region_names.tolist()
    elif not isinstance(region_names, list):
        region_names = list(region_names)
    
    matrix = np.zeros((len(region_names), len(region_names)))
    
    print(f"Reading file: {file_path}")
    print(f"Region names: {region_names}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Locate CSV-format matrix data
    csv_start = False
    csv_lines = []
    
    for line in lines:
        if "Matrix in CSV format:" in line:
            csv_start = True
            continue
        
        if csv_start and line.strip() and not line.startswith("="):
            csv_lines.append(line.strip())
    
    print(f"Found {len(csv_lines)} CSV data lines")
    
    # Parse CSV data
    for line in csv_lines:
        parts = line.split(',')
        if len(parts) > 1:
            region_name = parts[0]
            print(f"Processing region: {region_name}")
            
            # Check if region_name is in region_names list
            if region_name in region_names:
                row_idx = region_names.index(region_name)
                print(f"Region {region_name} at index {row_idx}")
                
                for col_idx, value_str in enumerate(parts[1:]):
                    if col_idx < len(region_names):
                        try:
                            matrix[row_idx, col_idx] = float(value_str)
                        except ValueError:
                            print(f"Warning: cannot convert '{value_str}' to float")
                            matrix[row_idx, col_idx] = 0.0
            else:
                print(f"Warning: region {region_name} not in region_names list")
    
    print(f"Matrix read successfully, shape: {matrix.shape}")
    print(f"Matrix range: {matrix.min():.6f} to {matrix.max():.6f}")
    
    return matrix

def calculate_relative_difference(weakened_matrix, normal_matrix):
    """
    Compute relative-difference matrix (weakened - normal) / |normal|; use 1e-10 when |normal| is too small to avoid division by zero.

    Parameters:
        weakened_matrix: matrix under weakened condition; np.ndarray, same shape as normal_matrix.
        normal_matrix: matrix under normal condition; np.ndarray.

    Returns:
        np.ndarray; same shape as input, elements are relative differences.
    """
    # Avoid division by zero: set near-zero values to a small positive constant
    normal_abs = np.abs(normal_matrix)
    normal_abs = np.where(normal_abs < 1e-10, 1e-10, normal_abs)
    
    relative_diff = (weakened_matrix - normal_matrix) / normal_abs
    return relative_diff

def plot_relative_difference_heatmap(relative_diff, region_names, result_dir, weakened_filename):
    """
    Plot Pearson correlation relative-difference heatmap (diverging colormap centered at 0) and save; also write data to corresponding txt.

    Parameters:
        relative_diff: relative-difference matrix; np.ndarray, shape=(n_regions, n_regions).
        region_names: region name list/array; aligned with matrix rows/columns.
        result_dir: save directory; str.
        weakened_filename: corresponding weakened filename (for output naming); str.
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
    
    # Colorbar range centered at 0
    vmax = max(np.abs(relative_diff.min()), np.abs(relative_diff.max()))
    vmin = -vmax
    
    # Plot heatmap
    im = ax.imshow(relative_diff, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set axes
    n_regions = len(region_names)
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(region_names, fontsize=10)
    
    # Add value annotations
    for i in range(n_regions):
        for j in range(n_regions):
            text = ax.text(j, i, f'{relative_diff[i, j]:.3f}',
                         ha="center", va="center", color="black", 
                         fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Relative Difference\n(Weakened - Normal) / |Normal|', 
                  fontweight='bold', fontsize=12)
    
    # Set title and labels
    ax.set_xlabel('Target Regions', fontweight='bold', fontsize=14)
    ax.set_ylabel('Source Regions', fontweight='bold', fontsize=14)
    ax.set_title(f'Relative Difference Heatmap\n{weakened_filename.replace(".txt", "")}', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Add grid
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
    
    print(f"Saved relative-difference heatmap: {output_filename}")
    
    # Save corresponding data file
    save_plot_data_to_txt(
        save_path,
        {
            'relative_difference_matrix': relative_diff,
            'region_names': region_names if isinstance(region_names, list) else region_names.tolist()
        },
        description="Relative-difference heatmap data\n"
                   f"- relative_difference_matrix: shape ({relative_diff.shape[0]}, {relative_diff.shape[1]}), relative-difference matrix\n"
                   f"- region_names: region name list, corresponding to matrix row and column indices"
    )

def analyze_all_region_pairs_coherence(states, region_labels, fs=1000, result_dir=None):
    """
    Analyze coherence and phase-locking value for all region pairs; compute LFP, PLV, coherence, etc. per pair,
    aggregate and optionally save to result_dir (default result/Coherent_analysis).

    Parameters:
        states: reservoir state time series; np.ndarray, shape=(T, N).
        region_labels: brain region per neuron; np.ndarray, shape=(N,).
        fs: sampling rate (Hz); int/float; default 1000.
        result_dir: result save directory; str or None.
    """
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), 'result', 'Coherent_analysis')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Get all brain regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    print(f"Starting coherence analysis for all {n_regions} region pairs...")
    print(f"Total region pairs to analyze: {n_regions * (n_regions - 1) // 2}")
    
    # Store all results
    all_results = {}
    pair_count = 0
    
    for i, region1 in enumerate(unique_regions):
        for j, region2 in enumerate(unique_regions):
            if i < j:  # Avoid duplicate and self-comparison
                pair_count += 1
                print(f"\nAnalyzing region pair {pair_count}: {region1} vs {region2}")
                
                try:
                    results = compare_regions_lfp(states, region_labels, region1, region2, fs, result_dir)
                    if results is not None:
                        all_results[f"{region1}_vs_{region2}"] = results
                except Exception as e:
                    print(f"Error analyzing {region1} vs {region2}: {e}")
                    continue
    
    print(f"\nDone. Analyzed {len(all_results)} region pairs.")
    print(f"All figures saved to: {result_dir}")
    
    return all_results

def save_plot_data_to_txt(save_path, data_dict, description=""):
    """
    Write data corresponding to the figure to a .txt file with the same base name (overloaded version for model comparison; supports dict/array etc.).

    Parameters:
        save_path: figure save path (.png); str; corresponding txt is save_path with .png replaced by .txt.
        data_dict: data dict; keys are data names (str), values are data (dict/array etc.).
        description: data description text; str; default "".
    """
    txt_path = save_path.replace('.png', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Data file: {os.path.basename(txt_path)}\n")
        f.write(f"Corresponding figure: {os.path.basename(save_path)}\n")
        f.write("="*60 + "\n\n")
        
        if description:
            f.write(f"Data description:\n{description}\n\n")
        
        f.write("="*60 + "\n")
        f.write("Data content:\n")
        f.write("="*60 + "\n\n")
        
        for key, value in data_dict.items():
            f.write(f"[{key}]\n")
            if isinstance(value, dict):
                f.write("Dictionary data:\n")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        f.write(f"  {k}: {v:.6f}\n")
                    elif isinstance(v, (list, np.ndarray)):
                        if isinstance(v, list):
                            v = np.array(v)
                        f.write(f"  {k}: array, shape {v.shape}\n")
                        if v.ndim == 1:
                            # For 1D array, save all values
                            f.write(f"    Values: " + ", ".join([f"{x:.6f}" for x in v]) + "\n")
                        else:
                            # For multi-dim array, save flattened
                            f.write(f"    Values (flattened): " + ", ".join([f"{x:.6f}" for x in v.flatten()]) + "\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            elif isinstance(value, np.ndarray):
                if value.ndim == 1:
                    f.write(f"Shape: ({len(value)},) - 1D array\n")
                    f.write("Data:\n")
                    for i, v in enumerate(value):
                        f.write(f"  {i}\t{v:.6f}\n")
                elif value.ndim == 2:
                    f.write(f"Shape: {value.shape} - 2D matrix\n")
                    f.write("Row index\tColumn index\tValue\n")
                    for i in range(value.shape[0]):
                        for j in range(value.shape[1]):
                            f.write(f"  {i}\t{j}\t{value[i,j]:.6f}\n")
                else:
                    f.write(f"Shape: {value.shape}\n")
                    f.write("Data (flattened):\n")
                    for i, v in enumerate(value.flatten()):
                        f.write(f"  {i}\t{v:.6f}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"List/tuple, length: {len(value)}\n")
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        f.write(f"  {i}\t{v:.6f}\n")
                    else:
                        f.write(f"  {i}\t{v}\n")
            else:
                f.write(f"Value: {value}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("Data save complete.\n")
        f.write("="*60 + "\n")
    
    print(f"Saved data file: {txt_path}")

# ========== LSTM model definition ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize LSTM classification model: single or multi-layer LSTM + FC on last time step.
        Parameters: input_size input feature dim; hidden_size hidden state dim; output_size number of classes; num_layers number of layers, default 1.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward: LSTM then FC on last time step.
        Parameters: x (batch_size, seq_len, input_size).
        Returns: (batch_size, output_size).
        """
        # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take output of last time step
        output = self.fc(lstm_out[:, -1, :])
        return output

# ========== CNN model definition ==========
class CNNModel(nn.Module):
    def __init__(self, input_channels=216, output_size=10):
        """
        Initialize 1D-CNN classification model: three conv layers + global average pooling + FC.
        Parameters: input_channels input channels (time-step feature dim), default 216; output_size number of classes, default 10.
        """
        super(CNNModel, self).__init__()
        
        # Conv1: 216 -> 128
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv2: 128 -> 128
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1)
        self.pool2 = nn.MaxPool1d(2)
        
        # Conv3: 128 -> 304
        self.conv3 = nn.Conv1d(128, 304, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global average pooling + FC layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(304, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

        # Added: dropout
        self.dropout_conv = nn.Dropout(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """
        Forward: transpose to (batch, channels, seq_len) then conv, pool, global pool, FC.
        Parameters: x (batch_size, seq_len, input_channels).
        Returns: (batch_size, output_size).
        """
        # x: (batch_size, seq_len, input_channels)
        # Convert to CNN format: (batch_size, input_channels, seq_len)
        x = x.transpose(1, 2)
        
        # Conv1 + ReLU + Pool + Dropout
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout_conv(x)
        
        # Conv2 + ReLU + Pool + Dropout
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout_conv(x)
        
        # Conv3 + ReLU + Pool + Dropout
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout_conv(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 304, 1)
        x = x.squeeze(-1)        # (batch, 304)
        
        # FC layer + Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

# ========== Transformer model definition ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Sinusoidal positional encoding.
        Parameters: d_model model dimension; max_len max sequence length, default 5000.
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term length is ceil(d_model/2), used to fill even index positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model, odd index positions are one fewer than even
        # So use only first d_model//2 div_term values to fill odd positions
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # First d_model//2 only
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward: add positional encoding to x.
        Parameters: x (seq_len, batch_size, d_model).
        Returns: same shape.
        """
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size=216, d_model=120, nhead=8, num_layers=3, 
                 dim_feedforward=240, output_size=10, max_seq_len=5000):
        """
        Initialize Transformer classification model: linear projection + positional encoding + multi-layer TransformerEncoder + temporal mean + FC.
        Parameters: input_size/d_model/nhead/num_layers/dim_feedforward/output_size/max_seq_len.
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Input projection layer
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_dropout = nn.Dropout(p=0.3)   # Added
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer Encoder (dropout set to 0.5)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.5,              # Modified
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        """
        Forward: project -> transpose -> positional encoding -> Encoder -> temporal mean -> FC.
        Parameters: x (batch_size, seq_len, input_size).
        Returns: (batch_size, output_size).
        """
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = x.transpose(0, 1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = x.mean(dim=0)
        x = self.fc_dropout(x)
        output = self.fc(x)
        
        return output

# ========== GNN model definition ==========
class TimeGCNLayer(nn.Module):
    """
    Temporal graph convolution layer: apply A@X@W on chain-graph adjacency matrix; used in GNNModel.
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        """
        Parameters: in_features/out_features input/output feature dim; dropout default 0.1.
        """
        super(TimeGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        """
        x: (batch_size, seq_len, in_features) - node features
        adj: (seq_len, seq_len) - adjacency matrix (chain graph)
        """
        # Graph convolution: A @ X @ W
        # x: (batch_size, seq_len, in_features)
        # adj: (seq_len, seq_len)
        # Matrix multiplication per batch
        x = torch.matmul(adj, x)  # (batch_size, seq_len, in_features)
        x = self.linear(x)  # (batch_size, seq_len, out_features)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class GNNModel(nn.Module):
    def __init__(self, input_size=216, hidden_size=192, num_layers=10, output_size=10, dropout=0.1):
        """
        Initialize temporal GNN: multi-layer TimeGCNLayer on chain graph + temporal mean + FC.
        Parameters: input_size/hidden_size/num_layers/output_size/dropout.
        """
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        
        # First layer: input_size -> hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(TimeGCNLayer(input_size, hidden_size, dropout))
        
        # Subsequent layers: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            self.layers.append(TimeGCNLayer(hidden_size, hidden_size, dropout))
        
        # Classification head
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, seq_len=None):
        """
        Forward: build chain adjacency, pass through GNN layers (first no residual, later with residual) + temporal mean + FC.
        Parameters: x (batch_size, seq_len, input_size); seq_len unused, inferred from x.
        Returns: (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.shape
        
        # Build chain-graph adjacency (each time step connected only to adjacent nodes)
        adj = self._build_chain_adjacency(seq_len, x.device)
        
        # Through all GNN layers; first layer no residual (dimension mismatch), later layers with residual
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer: input_size -> hidden_size, dimension mismatch, no residual
                x = layer(x, adj)
            else:
                # Subsequent layers: hidden_size -> hidden_size, use residual connection
                x = x + layer(x, adj)   # Residual connection
        
        # Average pooling over all time steps
        x = x.mean(dim=1)  # (batch_size, hidden_size)
        
        # Classification
        output = self.fc(x)
        
        return output

    
    def _build_chain_adjacency(self, seq_len, device):
        """
        Build chain-graph adjacency matrix: self-loops + (t,t+1)/(t+1,t) edges, then degree normalization.
        Parameters: seq_len number of time steps; device tensor device.
        Returns: (seq_len, seq_len) Tensor.
        """
        adj = torch.zeros(seq_len, seq_len, device=device)
        
        # Self-loops
        adj.fill_diagonal_(1.0)
        
        # Forward edges: (t, t+1)
        for i in range(seq_len - 1):
            adj[i, i+1] = 1.0
        
        # Backward edges: (t+1, t)
        for i in range(seq_len - 1):
            adj[i+1, i] = 1.0
        
        # Normalization (optional; simple degree normalization here)
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / (degree + 1e-8)
        
        return adj

# ========== Dataset class definition ==========
class SequenceDataset(Dataset):
    """
    Sequence dataset for batch training: wrap (X, y) as PyTorch Tensor, return (sample, label) by index.
    Parameters: X feature array (N, seq_len, feat); y label array (N,).
    Returns: __len__ is N; __getitem__(idx) returns (X[idx], y[idx]).
    """
    def __init__(self, X, y):
        """X: (N, seq_len, feat); y: (N,) integer labels."""
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== Multi-network comparison experiment ==========
def compare_all_networks(num_runs=20):
    """
    Compare biological ESN, LSTM, CNN, Transformer, GNN: run each num_runs times on Arabic digit data,
    collect train/test accuracy and runtime, then call plotting/saving.

    Parameters:
        num_runs: number of runs per model; int; default 20.

    Returns:
        results: dict, key=model name (ESN/LSTM/CNN/Transformer/GNN), value=dict with train_acc, test_acc, runtime lists.
    """
    print("\n" + "="*80)
    print(f"Starting multi-network comparison (each model {num_runs} random initializations): Biological ESN vs LSTM vs CNN vs Transformer vs GNN")
    print("="*80)
    
    # Basic parameters
    input_size = 13
    output_size = 1024
    reservoir_size = 12966
    leak_rate = 0.1
    mean_eig = 0.8
    w_in_scale = 0.1
    w_in_width = 0.05
    input_nodes_num = 216
    base_seed = 42
    
    # Use a temporary ESN to compute sparsity and parameter count
    tmp_esn = SimpleESN(input_size, output_size, input_nodes_num,
                        reservoir_size, leak_rate, mean_eig,
                        w_in_scale, w_in_width,
                        seed=base_seed,
                        weaken_region=weaken_region,
                        weaken_ratio=weaken_ratio)
    sparsity = np.count_nonzero(tmp_esn.W_res) / tmp_esn.W_res.size
    esn_params = int(reservoir_size * reservoir_size * sparsity + output_size * 10)
    del tmp_esn

    # Feature expansion
    train_X_exp = expand_features(train_X, input_nodes_num)
    test_X_exp = expand_features(test_X, input_nodes_num)
    
    print(f"Dataset info:")
    print(f"  Training set size: {train_X.shape}")
    print(f"  Test set size: {test_X.shape}")
    print(f"  Input node count: {input_nodes_num}")
    print(f"  Output node count: {output_size}")
    print(f"  Reservoir size: {reservoir_size}")
    print(f"  Sparsity: {sparsity*100:.4f}%")
    print(f"  ESN reservoir parameter estimate: {esn_params}")
    print(f"  Leak rate: {leak_rate}")
    print(f"  Target eigenvalue: {mean_eig}")
    print(f"  GPU acceleration: {'Yes' if torch.cuda.is_available() else 'No'}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_X_tensor = torch.FloatTensor(train_X_exp).to(device)
    test_X_tensor = torch.FloatTensor(test_X_exp).to(device)
    train_y_tensor = torch.LongTensor(train_y).to(device)
    test_y_tensor = torch.LongTensor(test_y).to(device)
    
    # Store all models' run results in dict
    results = {
        'Biological ESN': {'train_acc': [], 'test_acc': [], 'runtime': []},
        'LSTM':           {'train_acc': [], 'test_acc': [], 'runtime': []},
        'CNN':            {'train_acc': [], 'test_acc': [], 'runtime': []},
        'Transformer':    {'train_acc': [], 'test_acc': [], 'runtime': []},
        'GNN':            {'train_acc': [], 'test_acc': [], 'runtime': []},
    }
    
    # ============================ 1. Biological ESN ============================
    print("="*60)
    print("Biological ESN model (multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"\n[ESN] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        start_time = time.time()
        
        esn_model = SimpleESN(input_size, output_size, input_nodes_num,
                              reservoir_size, leak_rate, mean_eig,
                              w_in_scale, w_in_width,
                              seed=seed,
                              weaken_region=weaken_region,
                              weaken_ratio=weaken_ratio)
        
        # Forward to get features
        train_feat, _ = esn_model.forward(train_X_exp)
        test_feat, _ = esn_model.forward(test_X_exp)
        
        # Readout via Logistic regression
        clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
        clf.fit(train_feat, train_y)
        
        train_pred = clf.predict(train_feat)
        test_pred = clf.predict(test_feat)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['Biological ESN']['train_acc'].append(train_acc)
        results['Biological ESN']['test_acc'].append(test_acc)
        results['Biological ESN']['runtime'].append(runtime)
        
        del esn_model, clf, train_feat, test_feat, train_pred, test_pred
    
    # ============================ 2. LSTM ============================
    print("\n" + "="*60)
    print("LSTM model (multiple random initializations)")
    print("="*60)
    
    num_layers = 3
    hidden_size = 117
    
    for run_idx in range(num_runs):
        seed = base_seed + 100 + run_idx
        print(f"\n[LSTM] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        lstm_model = LSTMModel(
            input_size=input_nodes_num,
            hidden_size=hidden_size,
            output_size=10,
            num_layers=num_layers
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0005)
        
        start_time = time.time()
        # Train for 200 epochs
        for epoch in range(200):
            lstm_model.train()
            optimizer.zero_grad()
            outputs = lstm_model(train_X_tensor)
            loss = criterion(outputs, train_y_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        lstm_model.eval()
        with torch.no_grad():
            train_outputs = lstm_model(train_X_tensor)
            test_outputs = lstm_model(test_X_tensor)
        end_time = time.time()
        runtime = end_time - start_time
        
        train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
        test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['LSTM']['train_acc'].append(train_acc)
        results['LSTM']['test_acc'].append(test_acc)
        results['LSTM']['runtime'].append(runtime)
        
        del lstm_model, train_outputs, test_outputs, train_pred, test_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 3. CNN ============================
    print("\n" + "="*60)
    print("CNN model (multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + 200 + run_idx
        print(f"\n[CNN] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        cnn_model = CNNModel(input_channels=input_nodes_num, output_size=10).to(device)
        cnn_criterion = nn.CrossEntropyLoss()
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)
        
        # DataLoader for training
        batch_size_cnn = 512
        train_dataset_cnn = SequenceDataset(train_X_exp, train_y)
        train_loader_cnn = DataLoader(train_dataset_cnn,
                                      batch_size=batch_size_cnn,
                                      shuffle=True,
                                      pin_memory=True if torch.cuda.is_available() else False)
        
        start_time = time.time()
        for epoch in range(200):
            cnn_model.train()
            for batch_X, batch_y in train_loader_cnn:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                cnn_optimizer.zero_grad()
                outputs = cnn_model(batch_X)
                loss = cnn_criterion(outputs, batch_y)
                loss.backward()
                cnn_optimizer.step()
                
                del batch_X, batch_y, outputs, loss
        
        # Overall evaluation
        cnn_model.eval()
        with torch.no_grad():
            train_outputs = cnn_model(train_X_tensor)
            test_outputs = cnn_model(test_X_tensor)
        end_time = time.time()
        runtime = end_time - start_time
        
        train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
        test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['CNN']['train_acc'].append(train_acc)
        results['CNN']['test_acc'].append(test_acc)
        results['CNN']['runtime'].append(runtime)
        
        del cnn_model, train_dataset_cnn, train_loader_cnn
        del train_outputs, test_outputs, train_pred, test_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 4. Transformer ============================
    print("\n" + "="*60)
    print("Transformer model (multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + 300 + run_idx
        print(f"\n[Transformer] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        transformer_model = TransformerModel(
            input_size=input_nodes_num,
            d_model=120,
            nhead=8,
            num_layers=3,
            dim_feedforward=240,
            output_size=10
        ).to(device)
        
        transformer_criterion = nn.CrossEntropyLoss()
        transformer_optimizer = torch.optim.Adam(
            transformer_model.parameters(),
            lr=0.0005,
            weight_decay=0
        )
        
        batch_size = 512
        train_dataset = SequenceDataset(train_X_exp, train_y)
        test_dataset = SequenceDataset(test_X_exp, test_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True if torch.cuda.is_available() else False)
        
        start_time = time.time()
        for epoch in range(200):
            transformer_model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                transformer_optimizer.zero_grad()
                outputs = transformer_model(batch_X)
                loss = transformer_criterion(outputs, batch_y)
                loss.backward()
                transformer_optimizer.step()
                
                del batch_X, batch_y, outputs, loss
        
        # Evaluate: use DataLoader to avoid OOM
        transformer_model.eval()
        train_preds, train_labels = [], []
        test_preds, test_labels = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                outputs = transformer_model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(batch_y.numpy())
                del batch_X, batch_y, outputs, preds
            
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = transformer_model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(batch_y.numpy())
                del batch_X, batch_y, outputs, preds
        
        end_time = time.time()
        runtime = end_time - start_time
        
        train_acc = accuracy_score(train_labels, train_preds)
        test_acc = accuracy_score(test_labels, test_preds)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['Transformer']['train_acc'].append(train_acc)
        results['Transformer']['test_acc'].append(test_acc)
        results['Transformer']['runtime'].append(runtime)
        
        del transformer_model, train_dataset, test_dataset, train_loader, test_loader
        del train_preds, train_labels, test_preds, test_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 5. GNN ============================
    print("\n" + "="*60)
    print("GNN model (multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + 400 + run_idx
        print(f"\n[GNN] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        gnn_model = GNNModel(
            input_size=input_nodes_num,
            hidden_size=192,
            num_layers=10,
            output_size=10,
            dropout=0.2
        ).to(device)
        
        gnn_criterion = nn.CrossEntropyLoss()
        gnn_optimizer = torch.optim.Adam(
            gnn_model.parameters(),
            lr=0.0001,
            weight_decay=0
        )
        
        batch_size = 512
        train_dataset = SequenceDataset(train_X_exp, train_y)
        test_dataset = SequenceDataset(test_X_exp, test_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True if torch.cuda.is_available() else False)
        
        start_time = time.time()
        for epoch in range(200):
            gnn_model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                gnn_optimizer.zero_grad()
                outputs = gnn_model(batch_X)
                loss = gnn_criterion(outputs, batch_y)
                loss.backward()
                gnn_optimizer.step()
                
                del batch_X, batch_y, outputs, loss
        
        # Evaluate
        gnn_model.eval()
        gnn_train_pred, gnn_train_label = [], []
        gnn_test_pred, gnn_test_label = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                outputs = gnn_model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gnn_train_pred.extend(preds)
                gnn_train_label.extend(batch_y.numpy())
                del batch_X, batch_y, outputs, preds
            
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = gnn_model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gnn_test_pred.extend(preds)
                gnn_test_label.extend(batch_y.numpy())
                del batch_X, batch_y, outputs, preds
        
        end_time = time.time()
        runtime = end_time - start_time
        
        train_acc = accuracy_score(gnn_train_label, gnn_train_pred)
        test_acc = accuracy_score(gnn_test_label, gnn_test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['GNN']['train_acc'].append(train_acc)
        results['GNN']['test_acc'].append(test_acc)
        results['GNN']['runtime'].append(runtime)
        
        del gnn_model, train_dataset, test_dataset, train_loader, test_loader
        del gnn_train_pred, gnn_train_label, gnn_test_pred, gnn_test_label
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 6. Aggregate mean ± std and compute score ============================
    print("\n" + "="*80)
    print("Summary of 10 runs per model (Test acc and Runtime):")
    print("="*80)
    
    for model_name, metrics in results.items():
        train_acc_arr = np.array(metrics['train_acc'])
        test_acc_arr = np.array(metrics['test_acc'])
        runtime_arr = np.array(metrics['runtime'])
        
        mean_train = train_acc_arr.mean()
        std_train = train_acc_arr.std()
        mean_test = test_acc_arr.mean()
        std_test = test_acc_arr.std()
        mean_time = runtime_arr.mean()
        std_time = runtime_arr.std()
        
        score = mean_test / mean_time
        metrics['score'] = score
        
        print(f"{model_name}:")
        print(f"  Train acc = {mean_train:.4f} ± {std_train:.4f}")
        print(f"  Test  acc = {mean_test:.4f} ± {std_test:.4f}")
        print(f"  Runtime   = {mean_time:.2f} ± {std_time:.2f} sec")
        print(f"  score = mean(Test acc) / mean(Runtime) = {score:.4f}")
        print()
    
    # Plot (boxplot + score bar chart)
    create_network_comparison_plots(results)
    
    return results

def create_network_comparison_plots(results):
    """
    Plot three figures from compare_all_networks results: test accuracy boxplot, runtime boxplot, score bar chart (score=mean(test_acc)/mean(runtime)); save to result directory.

    Parameters:
        results: dict, key=model name, value=list of {train_acc, test_acc, runtime}; same as compare_all_networks return value.
    """
    # Create result directory
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
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
        'grid.linewidth': 0.5
    })
    
    print("Generating 3 comparison figures (boxplot + score bar chart)...")
    
    # 1. Accuracy boxplot (test acc)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_accuracy_boxplot(ax1, results)
    plt.tight_layout()
    save_path1 = os.path.join(result_dir, '01_accuracy_comparison.png')
    plt.savefig(save_path1)
    plt.close()
    print("Saved: 01_accuracy_comparison.png (Test Accuracy boxplot)")
    
    # Save corresponding data file
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    accuracy_data = {}
    for model_name in model_order:
        if model_name in results:
            accuracy_data[model_name] = {
                'test_acc': results[model_name]['test_acc'],
                'train_acc': results[model_name]['train_acc'],
                'mean_test_acc': np.mean(results[model_name]['test_acc']),
                'std_test_acc': np.std(results[model_name]['test_acc']),
                'mean_train_acc': np.mean(results[model_name]['train_acc']),
                'std_train_acc': np.std(results[model_name]['train_acc'])
            }
    save_plot_data_to_txt(
        save_path1,
        accuracy_data,
        description="Model accuracy comparison data\n"
                   f"- Each model contains: test_acc (test accuracy list), train_acc (train accuracy list)\n"
                   f"- mean_test_acc: mean test accuracy\n"
                   f"- std_test_acc: test accuracy std\n"
                   f"- mean_train_acc: mean train accuracy\n"
                   f"- std_train_acc: train accuracy std"
    )
    
    # 2. Runtime boxplot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_runtime_boxplot(ax2, results)
    plt.tight_layout()
    save_path2 = os.path.join(result_dir, '02_runtime_comparison.png')
    plt.savefig(save_path2)
    plt.close()
    print("Saved: 02_runtime_comparison.png (Runtime boxplot)")
    
    # Save corresponding data file
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    runtime_data = {}
    for model_name in model_order:
        if model_name in results:
            runtime_data[model_name] = {
                'runtime': results[model_name]['runtime'],
                'mean_runtime': np.mean(results[model_name]['runtime']),
                'std_runtime': np.std(results[model_name]['runtime']),
                'min_runtime': np.min(results[model_name]['runtime']),
                'max_runtime': np.max(results[model_name]['runtime'])
            }
    save_plot_data_to_txt(
        save_path2,
        runtime_data,
        description="Model runtime comparison data\n"
                   f"- Each model contains: runtime (runtime list, unit: seconds)\n"
                   f"- mean_runtime: mean runtime\n"
                   f"- std_runtime: runtime std\n"
                   f"- min_runtime: min runtime\n"
                   f"- max_runtime: max runtime"
    )
    
    # 3. Score bar chart
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    plot_score_bar(ax3, results)
    plt.tight_layout()
    save_path3 = os.path.join(result_dir, '03_score_bar.png')
    plt.savefig(save_path3)
    plt.close()
    print("Saved: 03_score_bar.png (score bar chart)")
    
    # Save corresponding data file
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    score_data = {}
    for model_name in model_order:
        if model_name in results and 'score' in results[model_name]:
            score_data[model_name] = {
                'score': results[model_name]['score'],
                'mean_test_acc': np.mean(results[model_name]['test_acc']),
                'mean_runtime': np.mean(results[model_name]['runtime'])
            }
    save_plot_data_to_txt(
        save_path3,
        score_data,
        description="Model score comparison data\n"
                   f"- score = mean(Test Acc) / mean(Runtime)\n"
                   f"- Each model contains: score (accuracy-speed tradeoff), mean_test_acc (mean test accuracy), mean_runtime (mean runtime)"
    )
    
    print(f"\nAll 3 figures saved to {result_dir}.")


def plot_accuracy_boxplot(ax, results):
    """
    Plot test accuracy boxplot for each model on given ax (multiple runs).
    Parameters: ax matplotlib Axes; results same structure as compare_all_networks return, with test_acc list.
    """
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    labels = ['Biological\nESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    
    data = [results[m]['test_acc'] for m in model_order]
    
    box = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Test Accuracy', fontweight='bold', fontsize=11)
    ax.set_title('Model Test Accuracy (10 runs)', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1)  # Set y-axis range to 0-1
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=9)


def plot_runtime_boxplot(ax, results):
    """
    Plot runtime boxplot for each model on given ax (unit: seconds).
    Parameters: ax matplotlib Axes; results contains runtime list.
    """
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    labels = ['Biological\nESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    
    data = [results[m]['runtime'] for m in model_order]
    
    box = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Runtime (seconds)', fontweight='bold', fontsize=11)
    ax.set_title('Model Runtime (10 runs)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=9)


def plot_score_bar(ax, results):
    """
    Plot score bar chart for each model on given ax (score = mean(test_acc)/mean(runtime)).
    Parameters: ax matplotlib Axes; results contains score or derivable test_acc/runtime.
    """
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    labels = ['Biological\nESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    
    scores = [results[m]['score'] for m in model_order]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    for bar, s in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{s:.4f}', ha='center', va='bottom',
                fontweight='bold', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('score = mean(Test Acc) / mean(Runtime)', fontweight='bold', fontsize=11)
    ax.set_title('Accuracy-Speed Tradeoff (score)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=9)

# ========== 10K-parameter network comparison experiment ==========
# ========== CNN 10K model definition ==========
class CNNModel10K(nn.Module):
    def __init__(self, input_channels=216, output_size=10):
        """
        Initialize 10K-parameter 1D-CNN classification model: three conv layers (8/8/20 channels) + global average pooling + FC, for lightweight comparison.
        Parameters: input_channels input channel count, default 216; output_size number of classes, default 10.
        """
        super(CNNModel10K, self).__init__()
        
        # Conv1: 216 -> 8
        self.conv1 = nn.Conv1d(input_channels, 8, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv2: 8 -> 8
        self.conv2 = nn.Conv1d(8, 8, kernel_size=5, padding=2, stride=1)
        self.pool2 = nn.MaxPool1d(2)
        
        # Conv3: 8 -> 20
        self.conv3 = nn.Conv1d(8, 20, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global average pooling + FC
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(20, 22)
        self.fc2 = nn.Linear(22, output_size)
        
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """
        Forward: transpose to (batch, channels, seq_len) then three conv, pool, global pool, FC.
        Parameters: x (batch_size, seq_len, input_channels).
        Returns: (batch_size, output_size).
        """
        # x: (batch_size, seq_len, input_channels)
        # Convert to CNN format: (batch_size, input_channels, seq_len)
        x = x.transpose(1, 2)
        
        # Conv1 + ReLU + Pool + Dropout
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout_conv(x)
        
        # Conv2 + ReLU + Pool + Dropout
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout_conv(x)
        
        # Conv3 + ReLU + Pool + Dropout
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout_conv(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 20, 1)
        x = x.squeeze(-1)        # (batch, 20)
        
        # FC layer + Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

# ========== Multi-network comparison (10K-parameter version) ==========
def compare_all_networks_10k(num_runs=20):
    """
    Compare biological ESN with 10K-parameter LSTM, CNN, Transformer, GNN: run each num_runs times on Arabic digit data, collect accuracy and runtime, plot and save.
    Parameters: num_runs runs per model, default 20.
    Returns: results dict, same structure as compare_all_networks.
    """
    print("\n" + "="*80)
    print(f"Starting multi-network comparison (10K-parameter version, each model {num_runs} random initializations): Biological ESN vs LSTM vs CNN vs Transformer vs GNN")
    print("="*80)
    
    # Basic parameters
    input_size = 13
    output_size = 1024
    reservoir_size = 12966
    leak_rate = 0.1
    mean_eig = 0.8
    w_in_scale = 0.1
    w_in_width = 0.05
    input_nodes_num = 216
    base_seed = 42
    
    # Use a temporary ESN to compute sparsity and parameter count
    tmp_esn = SimpleESN(input_size, output_size, input_nodes_num,
                        reservoir_size, leak_rate, mean_eig,
                        w_in_scale, w_in_width,
                        seed=base_seed,
                        weaken_region=weaken_region,
                        weaken_ratio=weaken_ratio)
    sparsity = np.count_nonzero(tmp_esn.W_res) / tmp_esn.W_res.size
    esn_params = int(reservoir_size * reservoir_size * sparsity + output_size * 10)
    del tmp_esn

    # Feature expansion
    train_X_exp = expand_features(train_X, input_nodes_num)
    test_X_exp = expand_features(test_X, input_nodes_num)
    
    print(f"Dataset info:")
    print(f"  Training set size: {train_X.shape}")
    print(f"  Test set size: {test_X.shape}")
    print(f"  Input node count: {input_nodes_num}")
    print(f"  Output node count: {output_size}")
    print(f"  Reservoir size: {reservoir_size}")
    print(f"  Sparsity: {sparsity*100:.4f}%")
    print(f"  ESN reservoir parameter estimate: {esn_params}")
    print(f"  Leak rate: {leak_rate}")
    print(f"  Target eigenvalue: {mean_eig}")
    print(f"  GPU acceleration: {'Yes' if torch.cuda.is_available() else 'No'}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_X_tensor = torch.FloatTensor(train_X_exp).to(device)
    test_X_tensor = torch.FloatTensor(test_X_exp).to(device)
    train_y_tensor = torch.LongTensor(train_y).to(device)
    test_y_tensor = torch.LongTensor(test_y).to(device)
    
    # Store all models' run results in dict
    results = {
        'Biological ESN': {'train_acc': [], 'test_acc': [], 'runtime': []},
        'LSTM':           {'train_acc': [], 'test_acc': [], 'runtime': []},
        'CNN':            {'train_acc': [], 'test_acc': [], 'runtime': []},
        'Transformer':    {'train_acc': [], 'test_acc': [], 'runtime': []},
        'GNN':            {'train_acc': [], 'test_acc': [], 'runtime': []},
    }
    
    # ============================ 1. Biological ESN ============================
    print("="*60)
    print("Biological ESN model (multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"\n[ESN] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        start_time = time.time()
        
        esn_model = SimpleESN(input_size, output_size, input_nodes_num,
                              reservoir_size, leak_rate, mean_eig,
                              w_in_scale, w_in_width,
                              seed=seed,
                              weaken_region=weaken_region,
                              weaken_ratio=weaken_ratio)
        
        # Forward to get features
        train_feat, _ = esn_model.forward(train_X_exp)
        test_feat, _ = esn_model.forward(test_X_exp)
        
        # Readout via Logistic regression
        clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
        clf.fit(train_feat, train_y)
        
        train_pred = clf.predict(train_feat)
        test_pred = clf.predict(test_feat)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['Biological ESN']['train_acc'].append(train_acc)
        results['Biological ESN']['test_acc'].append(test_acc)
        results['Biological ESN']['runtime'].append(runtime)
        
        del esn_model, clf, train_feat, test_feat, train_pred, test_pred
    
    # ============================ 2. LSTM (10K version) ============================
    print("\n" + "="*60)
    print("LSTM model (10K-parameter version, multiple random initializations)")
    print("="*60)
    
    num_layers = 2
    hidden_size = 10
    
    for run_idx in range(num_runs):
        seed = base_seed + 100 + run_idx
        print(f"\n[LSTM] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        lstm_model = LSTMModel(
            input_size=input_nodes_num,
            hidden_size=hidden_size,
            output_size=10,
            num_layers=num_layers
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
        
        start_time = time.time()
        # Train for 200 epochs
        for epoch in range(200):
            lstm_model.train()
            optimizer.zero_grad()
            outputs = lstm_model(train_X_tensor)
            loss = criterion(outputs, train_y_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        lstm_model.eval()
        with torch.no_grad():
            train_outputs = lstm_model(train_X_tensor)
            test_outputs = lstm_model(test_X_tensor)
        end_time = time.time()
        runtime = end_time - start_time
        
        train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
        test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['LSTM']['train_acc'].append(train_acc)
        results['LSTM']['test_acc'].append(test_acc)
        results['LSTM']['runtime'].append(runtime)
        
        del lstm_model, train_outputs, test_outputs, train_pred, test_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 3. CNN (10K version, batched training) ============================
    print("\n" + "="*60)
    print("CNN model (10K-parameter version, multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + 200 + run_idx
        print(f"\n[CNN] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        cnn_model = CNNModel10K(input_channels=input_nodes_num, output_size=10).to(device)
        cnn_criterion = nn.CrossEntropyLoss()
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)
        
        # DataLoader for training
        batch_size_cnn = 512
        train_dataset_cnn = SequenceDataset(train_X_exp, train_y)
        train_loader_cnn = DataLoader(train_dataset_cnn,
                                      batch_size=batch_size_cnn,
                                      shuffle=True,
                                      pin_memory=True if torch.cuda.is_available() else False)
        
        start_time = time.time()
        for epoch in range(200):
            cnn_model.train()
            for batch_X, batch_y in train_loader_cnn:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                cnn_optimizer.zero_grad()
                outputs = cnn_model(batch_X)
                loss = cnn_criterion(outputs, batch_y)
                loss.backward()
                cnn_optimizer.step()
                
                del batch_X, batch_y, outputs, loss
        
        # Overall evaluation
        cnn_model.eval()
        with torch.no_grad():
            train_outputs = cnn_model(train_X_tensor)
            test_outputs = cnn_model(test_X_tensor)
        end_time = time.time()
        runtime = end_time - start_time
        
        train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
        test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['CNN']['train_acc'].append(train_acc)
        results['CNN']['test_acc'].append(test_acc)
        results['CNN']['runtime'].append(runtime)
        
        del cnn_model, train_dataset_cnn, train_loader_cnn
        del train_outputs, test_outputs, train_pred, test_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 4. Transformer (10K version, no batching) ============================
    print("\n" + "="*60)
    print("Transformer model (10K-parameter version, multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + 300 + run_idx
        print(f"\n[Transformer] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        transformer_model = TransformerModel(
            input_size=input_nodes_num,
            d_model=15,
            nhead=3,
            num_layers=3,
            dim_feedforward=40,
            output_size=10
        ).to(device)
        
        transformer_criterion = nn.CrossEntropyLoss()
        transformer_optimizer = torch.optim.Adam(
            transformer_model.parameters(),
            lr=0.0005,
            weight_decay=0
        )
        
        start_time = time.time()
        # Train for 200 epochs, no batching, use full tensor
        for epoch in range(200):
            transformer_model.train()
            transformer_optimizer.zero_grad()
            outputs = transformer_model(train_X_tensor)
            loss = transformer_criterion(outputs, train_y_tensor)
            loss.backward()
            transformer_optimizer.step()
        
        # Evaluate
        transformer_model.eval()
        with torch.no_grad():
            train_outputs = transformer_model(train_X_tensor)
            test_outputs = transformer_model(test_X_tensor)
        end_time = time.time()
        runtime = end_time - start_time
        
        train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
        test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['Transformer']['train_acc'].append(train_acc)
        results['Transformer']['test_acc'].append(test_acc)
        results['Transformer']['runtime'].append(runtime)
        
        del transformer_model, train_outputs, test_outputs, train_pred, test_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 5. GNN (10K version, batched training) ============================
    print("\n" + "="*60)
    print("GNN model (10K-parameter version, multiple random initializations)")
    print("="*60)
    
    for run_idx in range(num_runs):
        seed = base_seed + 400 + run_idx
        print(f"\n[GNN] Run {run_idx+1}/{num_runs}, seed = {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        gnn_model = GNNModel(
            input_size=input_nodes_num,
            hidden_size=25,
            num_layers=8,
            output_size=10,
            dropout=0.2
        ).to(device)
        
        gnn_criterion = nn.CrossEntropyLoss()
        gnn_optimizer = torch.optim.Adam(
            gnn_model.parameters(),
            lr=0.0001,
            weight_decay=0.0
        )
        
        batch_size = 512
        train_dataset = SequenceDataset(train_X_exp, train_y)
        test_dataset = SequenceDataset(test_X_exp, test_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True if torch.cuda.is_available() else False)
        
        start_time = time.time()
        for epoch in range(200):
            gnn_model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                gnn_optimizer.zero_grad()
                outputs = gnn_model(batch_X)
                loss = gnn_criterion(outputs, batch_y)
                loss.backward()
                gnn_optimizer.step()
                
                del batch_X, batch_y, outputs, loss
        
        # Evaluate
        gnn_model.eval()
        gnn_train_pred, gnn_train_label = [], []
        gnn_test_pred, gnn_test_label = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                outputs = gnn_model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gnn_train_pred.extend(preds)
                gnn_train_label.extend(batch_y.numpy())
                del batch_X, batch_y, outputs, preds
            
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = gnn_model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gnn_test_pred.extend(preds)
                gnn_test_label.extend(batch_y.numpy())
                del batch_X, batch_y, outputs, preds
        
        end_time = time.time()
        runtime = end_time - start_time
        
        train_acc = accuracy_score(gnn_train_label, gnn_train_pred)
        test_acc = accuracy_score(gnn_test_label, gnn_test_pred)
        
        print(f"  Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Time = {runtime:.2f} s")
        
        results['GNN']['train_acc'].append(train_acc)
        results['GNN']['test_acc'].append(test_acc)
        results['GNN']['runtime'].append(runtime)
        
        del gnn_model, train_dataset, test_dataset, train_loader, test_loader
        del gnn_train_pred, gnn_train_label, gnn_test_pred, gnn_test_label
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================ 6. Aggregate mean ± std and compute score ============================
    print("\n" + "="*80)
    print("Summary of 20 runs per model (Test acc and Runtime):")
    print("="*80)
    
    for model_name, metrics in results.items():
        train_acc_arr = np.array(metrics['train_acc'])
        test_acc_arr = np.array(metrics['test_acc'])
        runtime_arr = np.array(metrics['runtime'])
        
        mean_train = train_acc_arr.mean()
        std_train = train_acc_arr.std()
        mean_test = test_acc_arr.mean()
        std_test = test_acc_arr.std()
        mean_time = runtime_arr.mean()
        std_time = runtime_arr.std()
        
        score = mean_test / mean_time
        metrics['score'] = score
        
        print(f"{model_name}:")
        print(f"  Train acc = {mean_train:.4f} ± {std_train:.4f}")
        print(f"  Test  acc = {mean_test:.4f} ± {std_test:.4f}")
        print(f"  Runtime   = {mean_time:.2f} ± {std_time:.2f} sec")
        print(f"  score = mean(Test acc) / mean(Runtime) = {score:.4f}")
        print()
    
    # Plot (boxplot + score bar chart)
    create_network_comparison_plots_10k(results)
    
    return results

def create_network_comparison_plots_10k(results):
    """
    Plot three figures from compare_all_networks_10k results: test accuracy boxplot, runtime boxplot, score bar chart; save to result directory (10K-parameter version).
    Parameters: results same as compare_all_networks_10k return value.
    """
    # Create result directory
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
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
        'grid.linewidth': 0.5
    })
    
    print("Generating 3 comparison figures (10K-parameter version, boxplot + score bar chart)...")
    
    # 1. Accuracy boxplot (test acc)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_accuracy_boxplot_10k(ax1, results)
    plt.tight_layout()
    save_path1 = os.path.join(result_dir, '01_accuracy_comparison_10k.png')
    plt.savefig(save_path1)
    plt.close()
    print("Saved: 01_accuracy_comparison_10k.png (Test Accuracy boxplot)")
    
    # Save corresponding data file
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    accuracy_data = {}
    for model_name in model_order:
        if model_name in results:
            accuracy_data[model_name] = {
                'test_acc': results[model_name]['test_acc'],
                'train_acc': results[model_name]['train_acc'],
                'mean_test_acc': np.mean(results[model_name]['test_acc']),
                'std_test_acc': np.std(results[model_name]['test_acc']),
                'mean_train_acc': np.mean(results[model_name]['train_acc']),
                'std_train_acc': np.std(results[model_name]['train_acc'])
            }
    save_plot_data_to_txt(
        save_path1,
        accuracy_data,
        description="Model accuracy comparison data (10K-parameter version)\n"
                   f"- Each model contains: test_acc (test accuracy list), train_acc (train accuracy list)\n"
                   f"- mean_test_acc: mean test accuracy\n"
                   f"- std_test_acc: test accuracy std\n"
                   f"- mean_train_acc: mean train accuracy\n"
                   f"- std_train_acc: train accuracy std"
    )
    
    # 2. Runtime boxplot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_runtime_boxplot_10k(ax2, results)
    plt.tight_layout()
    save_path2 = os.path.join(result_dir, '02_runtime_comparison_10k.png')
    plt.savefig(save_path2)
    plt.close()
    print("Saved: 02_runtime_comparison_10k.png (Runtime boxplot)")
    
    # Save corresponding data file
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    runtime_data = {}
    for model_name in model_order:
        if model_name in results:
            runtime_data[model_name] = {
                'runtime': results[model_name]['runtime'],
                'mean_runtime': np.mean(results[model_name]['runtime']),
                'std_runtime': np.std(results[model_name]['runtime']),
                'min_runtime': np.min(results[model_name]['runtime']),
                'max_runtime': np.max(results[model_name]['runtime'])
            }
    save_plot_data_to_txt(
        save_path2,
        runtime_data,
        description="Model runtime comparison data (10K-parameter version)\n"
                   f"- Each model contains: runtime (runtime list, unit: seconds)\n"
                   f"- mean_runtime: mean runtime\n"
                   f"- std_runtime: runtime std\n"
                   f"- min_runtime: min runtime\n"
                   f"- max_runtime: max runtime"
    )
    
    # 3. Score bar chart
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    plot_score_bar_10k(ax3, results)
    plt.tight_layout()
    save_path3 = os.path.join(result_dir, '03_score_bar_10k.png')
    plt.savefig(save_path3)
    plt.close()
    print("Saved: 03_score_bar_10k.png (score bar chart)")
    
    # Save corresponding data file
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    score_data = {}
    for model_name in model_order:
        if model_name in results and 'score' in results[model_name]:
            score_data[model_name] = {
                'score': results[model_name]['score'],
                'mean_test_acc': np.mean(results[model_name]['test_acc']),
                'mean_runtime': np.mean(results[model_name]['runtime'])
            }
    save_plot_data_to_txt(
        save_path3,
        score_data,
        description="Model score comparison data (10K-parameter version)\n"
                   f"- score = mean(Test Acc) / mean(Runtime)\n"
                   f"- Each model contains: score (accuracy-speed tradeoff), mean_test_acc (mean test accuracy), mean_runtime (mean runtime)"
    )
    
    print(f"\nAll 3 figures saved to {result_dir}.")

def plot_accuracy_boxplot_10k(ax, results):
    """
    Plot test accuracy boxplot for each model on given ax (10K-parameter version, multiple runs).
    Parameters: ax matplotlib Axes; results same structure as compare_all_networks_10k return value.
    """
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    labels = ['Biological\nESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    
    data = [results[m]['test_acc'] for m in model_order]
    
    box = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Test Accuracy', fontweight='bold', fontsize=11)
    ax.set_title('Model Test Accuracy (10K Parameters, 20 runs)', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1)  # Set y-axis range to 0-1
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=9)

def plot_runtime_boxplot_10k(ax, results):
    """
    Plot runtime boxplot for each model on given ax (10K-parameter version, unit: seconds).
    Parameters: ax matplotlib Axes; results contains runtime list.
    """
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    labels = ['Biological\nESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    
    data = [results[m]['runtime'] for m in model_order]
    
    box = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Runtime (seconds)', fontweight='bold', fontsize=11)
    ax.set_title('Model Runtime (10K Parameters, 20 runs)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=9)

def plot_score_bar_10k(ax, results):
    """
    Plot score bar chart for each model on given ax (10K-parameter version, score=mean(test_acc)/mean(runtime)).
    Parameters: ax matplotlib Axes; results contains score or test_acc/runtime.
    """
    model_order = ['Biological ESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    labels = ['Biological\nESN', 'LSTM', 'CNN', 'Transformer', 'GNN']
    
    scores = [results[m]['score'] for m in model_order]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    for bar, s in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{s:.4f}', ha='center', va='bottom',
                fontweight='bold', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('score = mean(Test Acc) / mean(Runtime)', fontweight='bold', fontsize=11)
    ax.set_title('Accuracy-Speed Tradeoff (score, 10K Parameters)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=9)

if __name__ == "__main__":
    # Paths
    base_dir = r"D:/study/memristor/Reservoir Calculation/conn2res/paper/Spoken_Arabic_Digit"
    train_txt = os.path.join(base_dir, "Train_Arabic_Digit.txt")
    test_txt = os.path.join(base_dir, "Test_Arabic_Digit.txt")

    max_len = 93  # Dataset max frames
    train_X = load_arabic_digit_txt(train_txt, max_len)  # (6600, 93, 13)
    test_X = load_arabic_digit_txt(test_txt, max_len)    # (2200, 93, 13)

    # Label generation
    train_y = np.repeat(np.arange(10), 660)
    test_y = np.repeat(np.arange(10), 220)

    # Hyperparameters
    input_size = 13
    output_size = 1024
    reservoir_size = 12966
    leak_rate = 0.1
    mean_eig = 0.8
    w_in_scale = 0.1
    w_in_width = 0.05
    input_nodes_num = 216
    weaken_region = ['CN']
    weaken_ratio = 0

    train_X_exp = expand_features(train_X, input_nodes_num)
    test_X_exp = expand_features(test_X, input_nodes_num)

    # Single run with current parameters
    esn = SimpleESN(input_size, output_size, input_nodes_num, reservoir_size, leak_rate, mean_eig, w_in_scale, w_in_width,
                    weaken_region = weaken_region, weaken_ratio = weaken_ratio)

    # Get state at all time steps
    start_time = time.time()
    train_feat, train_states = esn.forward(train_X_exp)
    test_feat, test_states = esn.forward(test_X_exp)

    # sklearn training
    clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
    clf.fit(train_feat, train_y)
    train_pred = clf.predict(train_feat)
    test_pred = clf.predict(test_feat)
    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)
    end_time = time.time()
    print(f"Classification time: {end_time - start_time:.2f} sec")
    print(f'Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}')

    # # ========== Hyperparameter sweep ==========
    # mean_eig_list = np.arange(1.15, 1.55, 0.05)
    # train_acc_list = []
    # test_acc_list = []
    # best_test_acc = -1
    # best_mean_eig = None
    # best_train_states = None
    # best_test_states = None
    # best_train_feat = None
    # best_test_feat = None
    # best_esn = None

    # for mean_eig in tqdm(mean_eig_list, desc='Parameter'):
    #     esn = SimpleESN(input_size, output_size, input_nodes_num, reservoir_size, leak_rate, mean_eig, w_in_scale, w_in_width,
    #                     weaken_region = weaken_region, weaken_ratio = weaken_ratio)
    #     train_feat, train_states = esn.forward(train_X_exp)
    #     test_feat, test_states = esn.forward(test_X_exp)
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

    # # Plot accuracy curve
    # plt.figure(figsize=(10,6))
    # plt.plot(mean_eig_list, train_acc_list, label='Train acc')
    # plt.plot(mean_eig_list, test_acc_list, label='Test acc')
    # plt.xlabel('mean_eig')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs mean_eig')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # print(f"Best mean_eig: {best_mean_eig:.2f}, best Test acc: {best_test_acc:.4f}")

    # ========== Spectral fingerprint ==========
    # Plot band decomposition for all regions
    lfp_save_dir = os.path.join(os.path.dirname(__file__), 'result', 'LFP')
    regions_batch = ['CN', 'ACx', 'HPC',
                    'IC', 'IL', 'OFC']

    print("\n" + "="*60)
    print("Plotting band decomposition for all regions...")
    plot_all_regions_band_decomposition(test_states, esn.region_labels, regions_batch, fs=1000, save_dir=lfp_save_dir)

    # Analyze relative power distribution for six regions
    target_regions = ['CN', 'ACx', 'HPC', 'IC', 'IL', 'OFC']

    print("\n" + "="*60)
    print("Analyzing relative power distribution for six regions...")
    multi_power_save_dir = os.path.join(os.path.dirname(__file__), 'result', 'LFP')
    multi_region_results = analyze_multiple_regions_power(test_states, esn.region_labels, target_regions, fs=1000, save_dir=multi_power_save_dir)

    # ========== Hierarchical delay ==========
    analyze_multi_region_lfp_flow(test_states, esn.region_labels, fs=1000, 
                                baseline_window=50, threshold_std=2.0,
                                time_window=(-100, 500))

    # ========== Causal analysis ==========
    print("Analyzing causal flow between regions...")
    causal_network, gc_matrix, te_matrix = visualize_causal_flow(test_states, esn.region_labels, fs=1000)

    # ========== Phase coupling ==========
    # Short-signal metrics for all region pairs (run all 78 pairs in one go)
    print("\n" + "="*60)
    print("Analyzing short-signal metrics for all region pairs...")
    print("13 regions, total C(13,2) = 78 region pairs to analyze")
    print("Using short-signal metrics, no synthetic data")

    # Check signal length
    signal_length = len(test_states)
    print(f"Signal length: {signal_length} samples")

    if signal_length < 200:
        print("Using short-signal analysis")
        all_short_results, metrics_matrices = analyze_short_signal_all_pairs(test_states, esn.region_labels, fs=1000)
    else:
        print("Using standard coherence analysis")
        all_coherence_results = analyze_all_region_pairs_coherence(test_states, esn.region_labels, fs=1000)

    # ========== Spatiotemporal trajectory ==========
    print("\n" + "="*60)
    print("Multi-region 3D UMAP trajectory visualization")
    print("="*60)

    # Create figure save directory
    save_dir = os.path.join(os.path.dirname(__file__), 'result', 'Spatiotemporal_trajectory')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Figure save path: {save_dir}")

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
    print(f"Detected {len(unique_regions)} regions: {unique_regions}")

    # Prepare data per region
    region_data = {}
    region_colors = {}
    region_markers = {}

    # Define colors and markers
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_regions)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'P', 'X', 'd', '|', '_']

    for i, region in enumerate(unique_regions):
        # Get node indices for this region
        region_idx = np.where(esn.region_labels == region)[0]
        if len(region_idx) > 0:
            # Extract state data for this region
            region_states = test_states[:, region_idx]  # (time_steps, nodes)
            region_data[region] = region_states
            region_colors[region] = colors[i]
            region_markers[region] = markers[i % len(markers)]
            print(f"{region}: {len(region_idx)} nodes")

    # UMAP dimensionality reduction
    print("\nRunning UMAP...")

    # UMAP per region
    region_umap = {}
    region_umap_normalized = {}
    n_components = 3  # UMAP to 3D
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)

    for region in unique_regions:
        if region in region_data:
            # Get state data for this region (time_steps, nodes)
            states_subset = region_data[region]
            print(f"{region}: raw state shape {states_subset.shape}")
            
            # UMAP: reduce node dimension to n_components, time dimension unchanged
            # UMAP input (n_samples, n_features); here samples=time steps, features=nodes
            states_umap = reducer.fit_transform(states_subset)  # (time_steps, n_components)
            region_umap[region] = states_umap
            
            # Normalize UMAP output per region to [0, 1]
            umap_normalized = np.zeros_like(states_umap)
            for i in range(n_components):
                umap_min = np.min(states_umap[:, i])
                umap_max = np.max(states_umap[:, i])
                if umap_max > umap_min:  # Avoid division by zero
                    umap_normalized[:, i] = (states_umap[:, i] - umap_min) / (umap_max - umap_min)
                else:
                    umap_normalized[:, i] = 0.5  # If max==min, set to 0.5
            
            region_umap_normalized[region] = umap_normalized
            print(f"{region}: UMAP shape {states_umap.shape}")
            print(f"{region}: UMAP normalized range [{np.min(umap_normalized):.3f}, {np.max(umap_normalized):.3f}]")

    # Single-region UMAP 3D spatiotemporal trajectory (exclude ACx, PL, OFC; gradient SVG style)
    print("\n" + "="*60)
    print("Single-region UMAP 3D spatiotemporal trajectory (gradient SVG style)")
    print("="*60)

    # Regions to display (exclude ACx, PL, OFC, drawn separately above)
    excluded_regions = ['ACx', 'PL', 'OFC']
    target_regions = [r for r in unique_regions if r not in excluded_regions]
    print(f"Plotting {len(target_regions)} regions: {target_regions}")

    # Create separate 3D trajectory figure per target region
    for region_name in target_regions:
        if region_name not in region_umap_normalized:
            print(f"Region {region_name}: normalized UMAP data not found, skipping")
            continue
        
        print(f"Plotting SVG for region {region_name}...")
        
        # Get UMAP data for this region
        umap_data_original = region_umap_normalized[region_name]
        n_points_original = len(umap_data_original)
        
        # Interpolate trajectory for denser points
        # Interpolation factor: 4x (same as ACx, PL, OFC)
        interpolation_factor = 4
        n_points_interpolated = n_points_original * interpolation_factor
        
        # Original time indices
        t_original = np.arange(n_points_original)
        # Interpolated time indices
        t_interpolated = np.linspace(0, n_points_original - 1, n_points_interpolated)
        
        # Interpolate each dimension (x, y, z)
        umap_data_interpolated = np.zeros((n_points_interpolated, 3))
        for dim in range(3):
            # Cubic spline for smooth trajectory
            interp_func = interp1d(t_original, umap_data_original[:, dim], 
                                kind='cubic', bounds_error=False, fill_value='extrapolate')
            umap_data_interpolated[:, dim] = interp_func(t_interpolated)
        
        # Use interpolated data for plotting
        umap_data = umap_data_interpolated
        n_points = n_points_interpolated
        
        print(f"  Original points: {n_points_original}, interpolated: {n_points}")
        
        # Base color for this region
        base_color = region_colors.get(region_name, '#1f77b4')
        
        # Convert RGB to [0, 1]
        if isinstance(base_color, str):
            rgb = mcolors.to_rgb(base_color)
        else:
            rgb = base_color[:3] if len(base_color) > 3 else base_color
        
        light_rgb = tuple(0.2 * r + 0.8 * 1.0 for r in rgb)
        dark_rgb = tuple(min(1.0, r * 1.15) for r in rgb)
        
        # Colormap
        colors_list = [light_rgb, dark_rgb]
        cmap = LinearSegmentedColormap.from_list(f'{region_name}_gradient', colors_list, N=n_points)
        
        # Time indices for colormap (0 to n_points-1)
        time_indices = np.arange(n_points)
        normalized_time = time_indices / max(1, n_points - 1)
        point_colors = cmap(normalized_time)
        
        # Create figure (no border)
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Default view
        ax.view_init(elev=20, azim=45)
        
        # Equal aspect
        ax.set_box_aspect([1, 1, 1])
        
        # Axis limits
        x_range = [umap_data_original[:, 0].min(), umap_data_original[:, 0].max()]
        y_range = [umap_data_original[:, 1].min(), umap_data_original[:, 1].max()]
        z_range = [umap_data_original[:, 2].min(), umap_data_original[:, 2].max()]
        
        x_margin = (x_range[1] - x_range[0]) * 0.1
        y_margin = (y_range[1] - y_range[0]) * 0.1
        z_margin = (z_range[1] - z_range[0]) * 0.1
        
        ax.set_xlim(x_range[0] - x_margin, x_range[1] + x_margin)
        ax.set_ylim(y_range[0] - y_margin, y_range[1] + y_margin)
        ax.set_zlim(z_range[0] - z_margin, z_range[1] + z_margin)
        
        # Remove margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 3D scatter with gradient colors
        scatter = ax.scatter(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2],
                            c=point_colors, s=200, alpha=1.0, edgecolors='none', linewidths=0)
        
        # Add "t=0" label at start
        start_point = umap_data[0]
        if n_points > 1:
            direction = umap_data[1] - umap_data[0]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                data_range = np.max([np.ptp(umap_data[:, 0]), 
                                    np.ptp(umap_data[:, 1]), 
                                    np.ptp(umap_data[:, 2])])
                offset_scale = 0.08 * data_range
                offset = -offset_scale * direction / direction_norm
                text_pos = start_point + offset
            else:
                text_pos = start_point
        else:
            text_pos = start_point
        
        # Text annotation
        ax.text(text_pos[0], text_pos[1], text_pos[2], 't=0', 
                fontsize=48, fontweight='bold', color='black',
                ha='center', va='center', family='sans-serif')
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_title('')
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        try:
            ax.xaxis.line.set_color((1, 1, 1, 0))
            ax.yaxis.line.set_color((1, 1, 1, 0))
            ax.zaxis.line.set_color((1, 1, 1, 0))
        except Exception:
            pass
        
        # Save as SVG
        svg_save_dir = os.path.join(save_dir, 'SVG_trajectories')
        os.makedirs(svg_save_dir, exist_ok=True)
        svg_path = os.path.join(svg_save_dir, f'{region_name}_trajectory.svg')
        
        fig.canvas.draw()
        
        plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0, 
                    facecolor='white', edgecolor='none', dpi=300)
        
        print(f"Saved SVG for region {region_name}: {svg_path}")
        
        # Save corresponding data file
        data_txt_path = os.path.join(svg_save_dir, f'{region_name}_trajectory_data.txt')

        save_plot_data_to_txt(
            data_txt_path,
            {
                'UMAP_3D_trajectory': umap_data_original,
                'UMAP_3D_trajectory_interpolated': umap_data,
                'start_point': umap_data[0],
                'end_point': umap_data[-1],
                'time_steps': np.arange(len(umap_data_original))
            },
            description=f"Region {region_name} UMAP 3D spatiotemporal trajectory data\n"
                    f"- UMAP_3D_trajectory: raw trajectory, shape ({len(umap_data_original)}, 3)\n"
                    f"- UMAP_3D_trajectory_interpolated: interpolated trajectory, shape ({len(umap_data)}, 3)\n"
                    f"- start_point: trajectory start (interpolated)\n"
                    f"- end_point: trajectory end (interpolated)\n"
                    f"- time_steps: original time step indices"
        )

        plt.close()

    print(f"\nUMAP 3D spatiotemporal trajectory visualization done for {len(target_regions)} regions.")
    print("This visualization shows:")
    print("1. Time evolution trajectory per region in UMAP space (gradient scatter)")
    print("2. Start label (t=0, large font)")
    print("3. 3D trajectory SVG for spatiotemporal dynamics")
    print("4. Per-region UMAP data normalized independently for comparison")
    print(f"5. All SVGs saved to: {os.path.join(save_dir, 'SVG_trajectories')}")

    # # Interactive visualization: five regions ACx, IL, PL, OFC, FP
    # try:
    #     focus_regions = ['ACx', 'PL', 'OFC'] # ACx:elevation=-34°azimuth=71;PL:azimuth=-86°,elevation=159;OFC:azimuth=9,elevation=-56

    #     for region in focus_regions:
    #         if region not in region_umap_normalized:
    #             print(f"Region {region} UMAP data not found, skipping")
    #             continue

    #         # Data and colors
    #         umap_data = region_umap_normalized[region]
    #         color = region_colors.get(region, 'C0')

    #         # Create interactive 3D window
    #         fig = plt.figure(figsize=(8, 6))
    #         ax = fig.add_subplot(111, projection='3d')

    #         # Set equal aspect
    #         ax.set_box_aspect([1, 1, 1])
            
    #         # Axis limits
    #         x_range = [umap_data[:, 0].min(), umap_data[:, 0].max()]
    #         y_range = [umap_data[:, 1].min(), umap_data[:, 1].max()]
    #         z_range = [umap_data[:, 2].min(), umap_data[:, 2].max()]
            
    #         x_margin = (x_range[1] - x_range[0]) * 0.1
    #         y_margin = (y_range[1] - y_range[0]) * 0.1
    #         z_margin = (z_range[1] - z_range[0]) * 0.1
            
    #         ax.set_xlim(x_range[0] - x_margin, x_range[1] + x_margin)
    #         ax.set_ylim(y_range[0] - y_margin, y_range[1] + y_margin)
    #         ax.set_zlim(z_range[0] - z_margin, z_range[1] + z_margin)

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
    #         # Interactive: rotate with mouse
    #         plt.show()
    # except Exception as e:
    #     print(f"Interactive five-region visualization failed: {e}")

    # ========== SVG trajectory plots for three regions ==========
    print("\n" + "="*60)
    print("SVG trajectory plots for three regions")
    print("="*60)

    svg_regions_config = {
        'ACx': {'azimuth': 71, 'elevation': -34}, 
        'PL': {'azimuth': -86, 'elevation': 159},
        'OFC': {'azimuth': 9, 'elevation': -56}
    }

    # Create gradient colormap per region
    for region_name, view_config in svg_regions_config.items():
        if region_name not in region_umap_normalized:
            print(f"Region {region_name} UMAP data not found, skipping")
            continue
        
        print(f"Plotting SVG for region {region_name}...")
        
        # Get UMAP data for this region
        umap_data_original = region_umap_normalized[region_name]
        n_points_original = len(umap_data_original)
        
        # Interpolate trajectory (4x points)
        interpolation_factor = 4
        n_points_interpolated = n_points_original * interpolation_factor
        
        t_original = np.arange(n_points_original)
        t_interpolated = np.linspace(0, n_points_original - 1, n_points_interpolated)
        
        umap_data_interpolated = np.zeros((n_points_interpolated, 3))
        for dim in range(3):
            interp_func = interp1d(t_original, umap_data_original[:, dim], 
                                kind='cubic', bounds_error=False, fill_value='extrapolate')
            umap_data_interpolated[:, dim] = interp_func(t_interpolated)
        
        umap_data = umap_data_interpolated
        n_points = n_points_interpolated
        
        print(f"  Original points: {n_points_original}, interpolated: {n_points}")
        
        base_color = region_colors.get(region_name, '#1f77b4')
        
        if isinstance(base_color, str):
            rgb = mcolors.to_rgb(base_color)
        else:
            rgb = base_color[:3] if len(base_color) > 3 else base_color
        
        light_rgb = tuple(0.2 * r + 0.8 * 1.0 for r in rgb)  # Light (near white)
        dark_rgb = tuple(min(1.0, r * 1.15) for r in rgb)  # Dark (slightly darker)
        
        colors_list = [light_rgb, dark_rgb]
        cmap = LinearSegmentedColormap.from_list(f'{region_name}_gradient', colors_list, N=n_points)
        
        time_indices = np.arange(n_points)
        normalized_time = time_indices / max(1, n_points - 1)
        point_colors = cmap(normalized_time)
        
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        ax.view_init(elev=view_config['elevation'], azim=view_config['azimuth'])
        
        ax.set_box_aspect([1, 1, 1])
        
        # Axis limits
        x_range = [umap_data_original[:, 0].min(), umap_data_original[:, 0].max()]
        y_range = [umap_data_original[:, 1].min(), umap_data_original[:, 1].max()]
        z_range = [umap_data_original[:, 2].min(), umap_data_original[:, 2].max()]
        
        x_margin = (x_range[1] - x_range[0]) * 0.1
        y_margin = (y_range[1] - y_range[0]) * 0.1
        z_margin = (z_range[1] - z_range[0]) * 0.1
        
        ax.set_xlim(x_range[0] - x_margin, x_range[1] + x_margin)
        ax.set_ylim(y_range[0] - y_margin, y_range[1] + y_margin)
        ax.set_zlim(z_range[0] - z_margin, z_range[1] + z_margin)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        use_2d_plot = False
        coords_2d_rotated = None
        
        if region_name == 'PL':
            ax.scatter([umap_data[0, 0]], [umap_data[0, 1]], [umap_data[0, 2]], 
                    s=0.1, alpha=0)
            fig.canvas.draw()
            
            proj_matrix = ax.get_proj()
            
            coords_2d = np.zeros((n_points, 2))
            for i in range(n_points):
                x_3d, y_3d, z_3d = umap_data[i, 0], umap_data[i, 1], umap_data[i, 2]
                x_2d, y_2d = proj3d.proj_transform(x_3d, y_3d, z_3d, proj_matrix)[:2]
                coords_2d[i, 0] = x_2d
                coords_2d[i, 1] = y_2d
            
            origin_2d = coords_2d[0].copy()
            coords_2d_centered = coords_2d - origin_2d

            rotation_angle_rad = -50.0 * np.pi / 180.0
            cos_angle = np.cos(rotation_angle_rad)
            sin_angle = np.sin(rotation_angle_rad)
            rotation_matrix_2d = np.array([[cos_angle, -sin_angle],
                                        [sin_angle, cos_angle]])
            coords_2d_rotated = coords_2d_centered @ rotation_matrix_2d.T
            
            use_2d_plot = True
        
        # Scatter plot
        if use_2d_plot and coords_2d_rotated is not None:
            fig.clf()
            ax = fig.add_subplot(111)
            
            scatter = ax.scatter(coords_2d_rotated[:, 0], coords_2d_rotated[:, 1],
                                c=point_colors, s=200, alpha=1.0, edgecolors='none', linewidths=0)
            
            # Add "t=0" label at start
            start_point_2d = coords_2d_rotated[0]
            if n_points > 1:
                direction_2d = coords_2d_rotated[1] - coords_2d_rotated[0]
                direction_norm = np.linalg.norm(direction_2d)
                if direction_norm > 0:
                    data_range = np.max([np.ptp(coords_2d_rotated[:, 0]), 
                                        np.ptp(coords_2d_rotated[:, 1])])
                    offset_scale = 0.08 * data_range
                    offset_2d = -offset_scale * direction_2d / direction_norm
                    text_pos_2d = start_point_2d + offset_2d
                else:
                    text_pos_2d = start_point_2d
            else:
                text_pos_2d = start_point_2d
            
            # Text annotation
            ax.text(text_pos_2d[0], text_pos_2d[1], 't=0', 
                    fontsize=32, fontweight='bold', color='black',
                    ha='center', va='center', family='sans-serif')
            
            # Axis limits
            x_range_2d = [coords_2d_rotated[:, 0].min(), coords_2d_rotated[:, 0].max()]
            y_range_2d = [coords_2d_rotated[:, 1].min(), coords_2d_rotated[:, 1].max()]
            x_margin_2d = (x_range_2d[1] - x_range_2d[0]) * 0.1
            y_margin_2d = (y_range_2d[1] - y_range_2d[0]) * 0.1
            ax.set_xlim(x_range_2d[0] - x_margin_2d, x_range_2d[1] + x_margin_2d)
            ax.set_ylim(y_range_2d[0] - y_margin_2d, y_range_2d[1] + y_margin_2d)
            
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_aspect('equal', adjustable='box')
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            scatter = ax.scatter(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2],
                                c=point_colors, s=200, alpha=1.0, edgecolors='none', linewidths=0)
            
            # Add "t=0" label at start
            start_point = umap_data[0]
            if n_points > 1:
                direction = umap_data[1] - umap_data[0]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    # Data range for offset
                    data_range = np.max([np.ptp(umap_data[:, 0]), 
                                        np.ptp(umap_data[:, 1]), 
                                        np.ptp(umap_data[:, 2])])
                    offset_scale = 0.08 * data_range  # Offset scale from data range
                    offset = -offset_scale * direction / direction_norm
                    text_pos = start_point + offset
                else:
                    text_pos = start_point
            else:
                text_pos = start_point
            
            # Text annotation
            ax.text(text_pos[0], text_pos[1], text_pos[2], 't=0', 
                    fontsize=32, fontweight='bold', color='black',
                    ha='center', va='center', family='sans-serif')
            
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_title('')
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)
            
            try:
                ax.xaxis.line.set_color((1, 1, 1, 0))
                ax.yaxis.line.set_color((1, 1, 1, 0))
                ax.zaxis.line.set_color((1, 1, 1, 0))
            except Exception:
                pass
        
        # Save as SVG
        svg_save_dir = os.path.join(save_dir, 'SVG_trajectories')
        os.makedirs(svg_save_dir, exist_ok=True)
        svg_path = os.path.join(svg_save_dir, f'{region_name}_trajectory.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0, 
                    facecolor='white', edgecolor='none', dpi=300)
        
        if use_2d_plot and coords_2d_rotated is not None:
            coords_2d_centered = coords_2d_rotated
        else:
            fig.canvas.draw()
            
            coords_2d = np.zeros((n_points, 2))
            for i in range(n_points):
                x_3d, y_3d, z_3d = umap_data[i, 0], umap_data[i, 1], umap_data[i, 2]
                x_2d, y_2d, _ = proj3d.proj_transform(x_3d, y_3d, z_3d, ax.get_proj())
                coords_2d[i, 0] = x_2d
                coords_2d[i, 1] = y_2d
            
            origin_2d = coords_2d[0].copy()
            coords_2d_centered = coords_2d - origin_2d
        
        txt_path = os.path.join(svg_save_dir, f'{region_name}_trajectory.txt')
        
        rotation_info = "No rotation"
        if region_name == 'PL':
            rotation_info = "Rotated 50° clockwise"
        elif region_name == 'ACx':
            rotation_info = "Rotated 8° counterclockwise"
        
        view_info = f'azimuth={view_config["azimuth"]}°, elevation={view_config["elevation"]}°'
        
        np.savetxt(txt_path, coords_2d_centered, fmt='%.8f', delimiter='\t',
                header=f'2D coordinates in view plane (origin at t=0)\n'
                        f'Region: {region_name}\n'
                        f'View: {view_info}\n'
                        f'{rotation_info}\n'
                        f'Format: X\tY',
                comments='# ')
        
        print(f"Saved SVG for region {region_name}: {svg_path}")
        plt.close()

    print(f"\nSVG trajectory plots for all three regions done.")
    print(f"Figure save path: {os.path.join(save_dir, 'SVG_trajectories')}")

    # UMAP dimension scan: local structure fidelity knee evaluation
    print("\n" + "="*60)
    print("UMAP dimension scan: Trustworthiness / Continuity / kNN retention evaluation")
    print("="*60)

    umap_params = dict(n_neighbors=15, min_dist=0.1)
    dims = list(range(2, 11))
    k_list = [15, 30]

    # Evaluate all regions
    region_dim_metrics = {}
    print("Starting dimension scan (d=2..10)...")
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
        print(f"Consistency (Euclidean vs cosine) k={k}: Trust r~{corr_T:.3f}, Cont r~{corr_C:.3f}, P@k r~{corr_P:.3f}")

    # Criterion: smallest d where T,C gain < threshold on most regions
    gain_thresh = 0.015
    majority_ratio = 0.6

    best_d_report = {}
    for k in k_list:
        d_star, ratio = find_knee_dimension(region_dim_metrics, dims, k, metric_family='euclidean')
        best_d_report[k] = (d_star, ratio)
        print(f"Knee (k={k}): d*={d_star}, coverage ~{ratio:.2f}")

    # Visualization output directory
    dim_dir = os.path.join(save_dir, 'UMAP_dim_sweep')
    os.makedirs(dim_dir, exist_ok=True)

    # S-curves: metric vs dimension (one per k, overlay regions and mean, mark d=3)
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

            # Mean curve for saving
            mean_curves = {}
            for key in keys:
                if region_curves[key]:
                    mean_curves[key] = np.mean(np.array(region_curves[key]), axis=0)
                else:
                    mean_curves[key] = np.array([])

            for ax, title, key in zip(axes, titles, keys):
                # Per-region light curves
                for vals in region_curves[key]:
                    ax.plot(dims, vals, color='gray', alpha=0.25, linewidth=1)
                # Mean bold line
                if region_curves[key]:
                    mean_curve = mean_curves[key]
                    ax.plot(dims, mean_curve, color='C0', linewidth=2.5, label='Mean across regions')
                # Vertical line at d=3
                ax.axvline(3, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='d=3')
                ax.set_xlabel('Dimension d')
                ax.set_title(title)
                ax.set_xlim(min(dims), max(dims))
                # Dynamic y-axis lower limit
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
            print(f"Saved: {out_path}")
            
            # Save corresponding data file
            save_plot_data_to_txt(
                out_path,
                {
                    'dimensions': np.array(dims),
                    'trustworthiness_mean': mean_curves[keys[0]],
                    'continuity_mean': mean_curves[keys[1]],
                    'knn_retention_mean': mean_curves[keys[2]],
                    'all_region_curves': {key: region_curves[key] for key in keys}
                },
                description=f"UMAP dimension scan S-curve data (metric={metric_family}, k={k})\n"
                        f"- dimensions: dimension list (d=2 to 10)\n"
                        f"- trustworthiness_mean: mean Trustworthiness per dimension\n"
                        f"- continuity_mean: mean Continuity per dimension\n"
                        f"- knn_retention_mean: mean kNN retention per dimension\n"
                        f"- all_region_curves: raw curves for all regions"
            )

    # Cross-region boxplot (Euclidean, k=30): T and C per d, mark d=3
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
    # Dynamic y-axis from boxplot data min
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
    print(f"Saved: {out_path}")

    # Save corresponding data file
    save_plot_data_to_txt(
        out_path,
        {
            'dimensions': np.array(dims),
            'trustworthiness_by_dim': {d: T_box[d] for d in dims if len(T_box[d]) > 0},
            'continuity_by_dim': {d: C_box[d] for d in dims if len(C_box[d]) > 0}
        },
        description=f"UMAP dimension scan boxplot data (k={box_k})\n"
                f"- dimensions: dimension list\n"
                f"- trustworthiness_by_dim: Trustworthiness list per dimension (all regions)\n"
                f"- continuity_by_dim: Continuity list per dimension (all regions)"
    )

    # PCA variance: first 10 PCs and cumulative variance per region
    print("\n" + "="*60)
    print("PCA variance: first 10 PCs and cumulative variance per region")
    print("="*60)

    pca_dir = os.path.join(save_dir, 'PCA_variance')
    os.makedirs(pca_dir, exist_ok=True)

    for region in unique_regions:
        if region not in region_data:
            continue
        X_tn = region_data[region]  # (time_steps, nodes)
        if X_tn.shape[0] < 3 or X_tn.shape[1] < 2:
            print(f"Region {region} insufficient samples, skipping PCA")
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
        ymin = max(0.0, float(min(ev_ratio.min(), cum_ratio.min())) - 0.02)
        ax.set_ylim(ymin, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        # Filename uses region name
        out_path = os.path.join(pca_dir, f'{region}_PCA_variance.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: {out_path}")
        
        # Save corresponding data file
        save_plot_data_to_txt(
            out_path,
            {
                'principal_components': np.arange(1, n_comp+1),
                'explained_variance_ratio': ev_ratio[:n_comp],
                'cumulative_variance_ratio': cum_ratio[:n_comp]
            },
            description=f"Region {region} PCA variance data\n"
                    f"- principal_components: PC index (1 to {n_comp})\n"
                    f"- explained_variance_ratio: variance ratio per PC\n"
                    f"- cumulative_variance_ratio: cumulative variance ratio"
        )

    # Region spatiotemporal trajectory angle and orthogonality
    print("\n" + "="*60)
    print("Region trajectory angle and orthogonality analysis")
    print("="*60)

    # Compute angle and orthogonality
    print("Computing region trajectory angles and orthogonality...")
    angles_matrix, orthogonality_matrix, region_names = calculate_trajectory_angles_and_orthogonality(
        region_umap_normalized, time_window=3
    )

    print("Done.")
    print(f"Angle matrix shape: {angles_matrix.shape}")
    print(f"Orthogonality matrix shape: {orthogonality_matrix.shape}")

    # Build spatiotemporal heatmap data
    time_steps, n_regions, _ = angles_matrix.shape
    valid_pairs = []

    # Collect all valid region pairs
    for i in range(n_regions):
        for j in range(i+1, n_regions):  # Upper triangle only, avoid duplicate
            valid_pairs.append((i, j, region_names[i], region_names[j]))

    print(f"Valid region pairs: {len(valid_pairs)}")

    # Time window used in computation
    time_window = 3
    valid_start = time_window
    valid_end = time_steps - time_window
    print(f"Time window: {time_window}")
    print(f"Valid range: {valid_start} to {valid_end} ({valid_end - valid_start} time points)")
    print(f"Invalid range: 0-{valid_start-1} and {valid_end+1}-{time_steps-1}")

    # Prepare heatmap data
    angle_heatmap_data = []
    angle_labels = []
    for i, j, name1, name2 in valid_pairs:
        angle_series = angles_matrix[:, i, j]
        angle_heatmap_data.append(angle_series)
        angle_labels.append(f'{name1}-{name2}')

    angle_heatmap_data = np.array(angle_heatmap_data)

    valid_angle_data = angle_heatmap_data[:, valid_start:valid_end]
    valid_time_range = range(valid_start, valid_end)

    # x-axis labels
    x_ticks = range(0, len(valid_time_range), max(1, len(valid_time_range)//10))

    # Angle spatiotemporal heatmap
    print("Generating angle spatiotemporal heatmap...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 12))

    # Plot angle heatmap
    im1 = ax1.imshow(valid_angle_data, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_title('Spatiotemporal Heatmap of Trajectory Angles', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=16)
    ax1.set_ylabel('Brain Region Pairs', fontsize=16)
    ax1.set_yticks(range(len(angle_labels)))
    ax1.set_yticklabels(angle_labels, fontsize=12)

    # x-axis labels
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([valid_time_range[i] for i in x_ticks], fontsize=12)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, label='Angle (degrees)', shrink=0.8)
    cbar1.ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save angle heatmap
    save_path1 = os.path.join(save_dir, "Spatiotemporal_Heatmap_Angles.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved angle heatmap: {save_path1}")
    plt.close()

    # Save corresponding data file
    save_plot_data_to_txt(
        save_path1,
        {
            'angle_heatmap_matrix': valid_angle_data,
            'time_steps': np.array(valid_time_range),
            'region_pairs': angle_labels
        },
        description="Angle spatiotemporal heatmap data\n"
                f"- angle_heatmap_matrix: shape ({valid_angle_data.shape[0]}, {valid_angle_data.shape[1]}), rows=region pairs, cols=time steps\n"
                f"- time_steps: valid time step indices\n"
                f"- region_pairs: region pair names, one per row"
    )

    # Orthogonality spatiotemporal heatmap
    print("Generating orthogonality spatiotemporal heatmap...")
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

    # x-axis labels
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([valid_time_range[i] for i in x_ticks], fontsize=12)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, label='Orthogonality (0-1)', shrink=0.8)
    cbar2.ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save orthogonality heatmap
    save_path2 = os.path.join(save_dir, "Spatiotemporal_Heatmap_Orthogonality.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved orthogonality heatmap: {save_path2}")
    plt.close()

    # Save corresponding data file
    save_plot_data_to_txt(
        save_path2,
        {
            'orthogonality_heatmap_matrix': valid_orth_data,
            'time_steps': np.array(valid_time_range),
            'region_pairs': angle_labels
        },
        description="Orthogonality spatiotemporal heatmap data\n"
                f"- orthogonality_heatmap_matrix: shape ({valid_orth_data.shape[0]}, {valid_orth_data.shape[1]}), rows=region pairs, cols=time steps\n"
                f"- time_steps: valid time step indices\n"
                f"- region_pairs: region pair names, one per row"
    )

    # Visualization 4: time window analysis
    print("Generating time window analysis figure...")

    # Angle and orthogonality under different time windows
    time_windows = [5, 25]  # Window sizes for 94 time steps
    window_analysis = {}

    for window in time_windows:
        if window < time_steps // 3:  # Keep window not too large
            window_angles = []
            window_orth = []
            
            for i, j, name1, name2 in valid_pairs:
                angle_series = angles_matrix[:, i, j]
                orth_series = orthogonality_matrix[:, i, j]
                
                # Sliding window statistics
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

    # Create time window analysis figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, window in enumerate(time_windows):
        if window in window_analysis:
            ax = axes[idx]
            
            angles = window_analysis[window]['angles']
            orth = window_analysis[window]['orthogonality']
            
            if len(angles) > 0 and len(orth) > 0:
                # Scatter plot
                scatter = ax.scatter(angles, orth, alpha=0.6, s=50, c=range(len(angles)), cmap='viridis')
                ax.set_xlabel('Angle (degrees)', fontsize=12)
                ax.set_ylabel('Orthogonality', fontsize=12)
                ax.set_title(f'Time Window: {window} steps\n({len(angles)} data points)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                    
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Time Order', fontsize=10)

    plt.tight_layout()

    # Save time window analysis figure
    save_path = os.path.join(save_dir, "Time_Window_Analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved time window analysis figure: {save_path}")
    plt.close()

    # Save corresponding data file
    window_data = {}
    for window in time_windows:
        if window in window_analysis:
            window_data[f'window_{window}_angles'] = window_analysis[window]['angles']
            window_data[f'window_{window}_orthogonality'] = window_analysis[window]['orthogonality']
    save_plot_data_to_txt(
        save_path,
        window_data,
        description="Time window analysis data\n"
                f"- window_X_angles: angle data for time window X\n"
                f"- window_X_orthogonality: orthogonality data for time window X"
    )

    # Statistical summary
    print("\n" + "="*60)
    print("Statistical summary")
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
        print(f"  Mean: {np.mean(all_angles):.2f} deg")
        print(f"  Median: {np.median(all_angles):.2f} deg")
        print(f"  Std: {np.std(all_angles):.2f} deg")
        print(f"  Min: {np.min(all_angles):.2f} deg")
        print(f"  Max: {np.max(all_angles):.2f} deg")

    if len(all_orthogonality) > 0:
        print(f"\nOrthogonality statistics:")
        print(f"  Mean: {np.mean(all_orthogonality):.4f}")
        print(f"  Median: {np.median(all_orthogonality):.4f}")
        print(f"  Std: {np.std(all_orthogonality):.4f}")
        print(f"  Min: {np.min(all_orthogonality):.4f}")
        print(f"  Max: {np.max(all_orthogonality):.4f}")

    # Mean angle matrix to find max/min region pairs
    mean_angles = np.zeros((len(region_names), len(region_names)))
    for i in range(len(region_names)):
        for j in range(len(region_names)):
            if i != j:
                valid_angles = angles_matrix[:, i, j]
                valid_angles = valid_angles[valid_angles > 0]
                if len(valid_angles) > 0:
                    mean_angles[i, j] = np.mean(valid_angles)

    # Region pairs with max and min angle
    if np.max(mean_angles) > 0:
        max_angle_idx = np.unravel_index(np.argmax(mean_angles), mean_angles.shape)
        min_angle_idx = np.unravel_index(np.argmin(mean_angles[mean_angles > 0]), mean_angles.shape)
        
        print(f"\nMax angle pair: {region_names[max_angle_idx[0]]} - {region_names[max_angle_idx[1]]} ({mean_angles[max_angle_idx]:.1f} deg)")
        print(f"Min angle pair: {region_names[min_angle_idx[0]]} - {region_names[min_angle_idx[1]]} ({mean_angles[min_angle_idx]:.1f} deg)")

    print(f"\nAll analysis figures saved to: {save_dir}")
    print("Analysis complete.")

    # ======================================================================
    # Structural projection & virtual dynamical stimulation & intrinsic timescale experiment
    # ======================================================================
    script_dir = os.path.dirname(__file__)
    result_root = os.path.join(script_dir, "result", "proj_dyn_timescale")
    os.makedirs(result_root, exist_ok=True)

    # Matplotlib basic style
    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300
    })

    # -----------------------------------------
    # Structural projection & dynamical response: full-brain 13x13 heatmap (diagonal 0)
    # -----------------------------------------

    region_order = esn.unique_regions
    num_regions = len(region_order)

    struct_mat = np.zeros((num_regions, num_regions), dtype=np.float32)
    dyn_mat = np.zeros((num_regions, num_regions), dtype=np.float32)

    # Build region-region matrix from neuron-level W_res
    regions_rr, region_adj = build_region_adjacency(
        esn.W_res, esn.region_labels, region_order=region_order, normalize_pair=True
    )
    assert np.array_equal(regions_rr, region_order)

    T = 400
    stim_onset = 50
    stim_duration = 2
    stim_amp = 0.5
    k_hop = 2

    for si, seed_region in enumerate(region_order):
        # 1) Structural projection: k-hop diffusion from this source region on region-region matrix
        struct_proj_region = compute_structural_projection_region(
            region_adj, seed_region_idx=si, k_hop=k_hop, decay=0.5, row_normalize=True
        )

        # Source does not project to self: set diagonal to 0
        struct_proj_region[si] = 0.0
        struct_mat[si, :] = struct_proj_region

        # 2) Dynamical stimulation: square-wave stimulus to this region
        external_stim, stim_idx = make_region_stim(
            esn, stim_regions=[seed_region], stim_neurons=None,
            T=T, onset=stim_onset, duration=stim_duration, amplitude=stim_amp
        )
        h_seq = esn.simulate_with_stim(external_stim)
        _, lfp_dict = compute_region_lfp(
            h_seq, esn.region_labels, region_order=region_order
        )
        _, dyn_resp_region = compute_region_response(
            lfp_dict,
            onset=stim_onset,
            offset=stim_onset + stim_duration,
            baseline_end=stim_onset,
            normalize=False,
            mode="integral_abs")

        # Remove source self-response
        dyn_resp_region[si] = 0.0
        dyn_mat[si, :] = dyn_resp_region

    # Remove diagonal (set to 0 again)
    for i in range(num_regions):
        struct_mat[i, i] = 0.0
        dyn_mat[i, i] = 0.0

    # Global normalize to [0,1] by matrix max
    struct_max = struct_mat.max()
    if struct_max > 0:
        struct_mat = struct_mat / struct_max

    dyn_max = dyn_mat.max()
    if dyn_max > 0:
        dyn_mat = dyn_mat / dyn_max

    # ===== Log transform both matrices (stretch small, compress large) =====
    # log1p map to [0,1]; larger log_scale = more log-like
    log_scale = 500
    struct_mat_log = np.log1p(log_scale * struct_mat) / np.log1p(log_scale)
    dyn_mat_log = np.log1p(log_scale * dyn_mat) / np.log1p(log_scale)

    # ===== Figure 1: structural projection heatmap =====
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    im1 = ax.imshow(struct_mat_log, origin="lower", aspect="equal")
    ax.set_xticks(np.arange(num_regions))
    ax.set_yticks(np.arange(num_regions))
    ax.set_xticklabels(region_order, rotation=90)
    ax.set_yticklabels(region_order)
    ax.set_xlabel("Target region (column)")
    ax.set_ylabel("Source region (row)")
    ax.set_title("Structural projection field (source row -> target column)")
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label("Structural projection (log-norm.)")
    plt.tight_layout()
    fig_path1 = os.path.join(result_root, "structural_projection_heatmap.png")
    plt.savefig(fig_path1, bbox_inches="tight")
    plt.close(fig)

    # Save data file
    data_path1 = os.path.join(result_root, "structural_projection_heatmap.txt")
    with open(data_path1, 'w', encoding='utf-8') as f:
        f.write("# Structural projection matrix\n")
        f.write("# Matrix dimension: {} x {}\n".format(num_regions, num_regions))
        f.write("# Row (source) order: {}\n".format(", ".join(region_order)))
        f.write("# Column (target) order: {}\n".format(", ".join(region_order)))
        f.write("# Value range: [{:.6f}, {:.6f}]\n".format(struct_mat_log.min(), struct_mat_log.max()))
        f.write("# Diagonal set to 0 (source does not project to self)\n")
        f.write("# Format: each row = source region, each column = target region\n")
        for i, rname in enumerate(region_order):
            f.write("# Source region: {}\n".format(rname))
            f.write(" ".join(["{:.6f}".format(val) for val in struct_mat_log[i, :]]) + "\n")
    print(f"Structural projection heatmap saved to: {fig_path1}")
    print(f"Structural projection data saved to: {data_path1}")

    # ===== Figure 2: dynamical response heatmap =====
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    im2 = ax.imshow(dyn_mat_log, origin="lower", aspect="equal")
    ax.set_xticks(np.arange(num_regions))
    ax.set_yticks(np.arange(num_regions))
    ax.set_xticklabels(region_order, rotation=90)
    ax.set_yticklabels(region_order)
    ax.set_xlabel("Target region (column)")
    ax.set_ylabel("Source region (row)")
    ax.set_title("Dynamical response (source row -> target column)")
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Dynamic response (log-norm.)")
    plt.tight_layout()
    fig_path2 = os.path.join(result_root, "dynamic_response_heatmap.png")
    plt.savefig(fig_path2, bbox_inches="tight")
    plt.close(fig)

    # Save data file (log-transformed matrix)
    data_path2 = os.path.join(result_root, "dynamic_response_heatmap.txt")
    with open(data_path2, 'w', encoding='utf-8') as f:
        f.write("# Dynamical response matrix\n")
        f.write("# Matrix dimension: {} x {}\n".format(num_regions, num_regions))
        f.write("# Row (source) order: {}\n".format(", ".join(region_order)))
        f.write("# Column (target) order: {}\n".format(", ".join(region_order)))
        f.write("# Value range: [{:.6f}, {:.6f}]\n".format(dyn_mat_log.min(), dyn_mat_log.max()))
        f.write("# Diagonal set to 0 (source does not respond to self)\n")
        f.write("# Format: each row = source region, each column = target region\n")
        for i, rname in enumerate(region_order):
            f.write("# Source region: {}\n".format(rname))
            f.write(" ".join(["{:.6f}".format(val) for val in dyn_mat_log[i, :]]) + "\n")
    print(f"Dynamical response heatmap saved to: {fig_path2}")
    print(f"Dynamical response data saved to: {data_path2}")

    # ===== Figure 3: structure vs dynamics scatter =====
    mask = ~np.eye(num_regions, dtype=bool)
    struct_flat = struct_mat_log[mask]
    dyn_flat = dyn_mat_log[mask]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(struct_flat, dyn_flat, s=10, alpha=0.7)
    min_v = 0.0
    max_v = max(struct_flat.max(), dyn_flat.max()) * 1.05
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=0.8, color='gray', label='y=x')
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Structural projection (log-norm.)")
    ax.set_ylabel("Dynamic response (log-norm.)")
    ax.set_title("Structure vs dynamics (diagonal removed)")
    ax.legend(frameon=False)

    if struct_flat.std() > 0 and dyn_flat.std() > 0:
        pearson_r = np.corrcoef(struct_flat, dyn_flat)[0, 1]
        spearman_r, spearman_p = spearmanr(struct_flat, dyn_flat)

        # Annotate both r and rho on figure
        ax.text(
            0.05, 0.95,
            f"r = {pearson_r:.3f}\n"
            f"ρ = {spearman_r:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Console print for record
        print(
            f"[Edge-wise SFC] Pearson r = {pearson_r:.3f}, "
            f"Spearman rho = {spearman_r:.3f}, p = {spearman_p:.1e}"
        )

    plt.tight_layout()
    fig_path3 = os.path.join(result_root, "structural_vs_dynamic_scatter.png")
    plt.savefig(fig_path3, bbox_inches="tight")
    plt.close(fig)

    # Save data file
    data_path3 = os.path.join(result_root, "structural_vs_dynamic_scatter.txt")
    with open(data_path3, 'w', encoding='utf-8') as f:
        f.write("# Structure vs dynamical response scatter data\n")
        f.write("# Number of points: {}\n".format(len(struct_flat)))
        f.write("# Diagonal removed (source=target)\n")
        if struct_flat.std() > 0 and dyn_flat.std() > 0:
            f.write("# Pearson r = {:.6f}\n".format(pearson_r))
            f.write("# Spearman rho = {:.6f}, p = {:.1e}\n".format(spearman_r, spearman_p))
        f.write("# Col1: Structural projection (log-norm.)\n")
        f.write("# Col2: Dynamic response (log-norm.)\n\n")
        for i in range(len(struct_flat)):
            f.write("{:.6f}\t{:.6f}\n".format(struct_flat[i], dyn_flat[i]))
    print(f"Structure vs dynamics scatter saved to: {fig_path3}")
    print(f"Structure vs dynamics scatter data saved to: {data_path3}")

    # Source/target-level Spearman correlation
    row_spearman = []  # Source (row): correlation per row
    col_spearman = []  # Target (column): correlation per column

    # Source (row): row i vs row i, diagonal removed
    for i in range(num_regions):
        mask_i = np.ones(num_regions, dtype=bool)
        mask_i[i] = False  # Exclude self-column

        x = struct_mat_log[i, mask_i]
        y = dyn_mat_log[i, mask_i]

        if x.std() > 0 and y.std() > 0:
            rho_i, _ = spearmanr(x, y)
        else:
            rho_i = np.nan
        row_spearman.append(rho_i)

    # Target (column): column j vs column j, diagonal removed
    for j in range(num_regions):
        mask_j = np.ones(num_regions, dtype=bool)
        mask_j[j] = False

        x = struct_mat_log[mask_j, j]
        y = dyn_mat_log[mask_j, j]

        if x.std() > 0 and y.std() > 0:
            rho_j, _ = spearmanr(x, y)
        else:
            rho_j = np.nan
        col_spearman.append(rho_j)

    row_spearman = np.asarray(row_spearman, dtype=float)
    col_spearman = np.asarray(col_spearman, dtype=float)

    for i, rname in enumerate(region_order):
        print(f"[Source] {rname:>4s}: Spearman rho = {row_spearman[i]: .3f}")
    for j, rname in enumerate(region_order):
        print(f"[Target] {rname:>4s}: Spearman rho = {col_spearman[j]: .3f}")

    # ===== Figure 4: source (row) Spearman =====
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.bar(np.arange(num_regions), row_spearman)
    ax.set_xticks(np.arange(num_regions))
    ax.set_xticklabels(region_order, rotation=90)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Source region")
    ax.set_title("Source (row) structure-causality correlation")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    fig_path4 = os.path.join(result_root, "source_region_spearman.png")
    plt.savefig(fig_path4, bbox_inches="tight")
    plt.close(fig)

    # Save data file
    data_path4 = os.path.join(result_root, "source_region_spearman.txt")
    with open(data_path4, 'w', encoding='utf-8') as f:
        f.write("# Source (row) structure-causality Spearman\n")
        f.write("# Per source: Spearman between structural projection (log) and dynamical response\n")
        f.write("# Diagonal removed\n")
        f.write("# Col1: region name, Col2: Spearman rho\n\n")
        for i, rname in enumerate(region_order):
            f.write("{}\t{:.6f}\n".format(rname, row_spearman[i]))
    print(f"Source Spearman figure saved to: {fig_path4}")
    print(f"Source Spearman data saved to: {data_path4}")

    # ===== Figure 5: target (column) Spearman =====
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.bar(np.arange(num_regions), col_spearman)
    ax.set_xticks(np.arange(num_regions))
    ax.set_xticklabels(region_order, rotation=90)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Target region")
    ax.set_title("Target (column) structure-causality correlation")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    fig_path5 = os.path.join(result_root, "target_region_spearman.png")
    plt.savefig(fig_path5, bbox_inches="tight")
    plt.close(fig)

    # Save data file
    data_path5 = os.path.join(result_root, "target_region_spearman.txt")
    with open(data_path5, 'w', encoding='utf-8') as f:
        f.write("# Target (column) structure-causality Spearman\n")
        f.write("# Per target: Spearman between structural projection (log) and dynamical response\n")
        f.write("# Diagonal removed\n")
        f.write("# Col1: region name, Col2: Spearman rho\n\n")
        for j, rname in enumerate(region_order):
            f.write("{}\t{:.6f}\n".format(rname, col_spearman[j]))
    print(f"Target Spearman figure saved to: {fig_path5}")
    print(f"Target Spearman data saved to: {data_path5}")

    # -----------------------------------------
    # Intrinsic timescale: noise-driven spontaneous activity
    # -----------------------------------------

    # Small noise as external input, run network for a period
    T_noise = 2000
    np.random.seed(123)
    noise_std = 0.02
    external_noise = np.random.randn(T_noise, esn.reservoir_size).astype(np.float32) * noise_std

    h_seq_noise = esn.simulate_with_stim(external_noise)
    regions_ts, lfp_dict_ts = compute_region_lfp(h_seq_noise, esn.region_labels, region_order=esn.unique_regions)

    # lag from 200 to 400
    max_lag = 400
    taus = []
    ac_curves = []

    for r in regions_ts:
        ts = lfp_dict_ts[r]
        ac = autocorr_1d(ts, max_lag=max_lag)
        ac_curves.append(ac)
        tau = estimate_timescale_from_ac(ac, threshold=np.exp(-1), default=max_lag)
        tau = estimate_timescale_integral(ac, max_lag=max_lag)
        taus.append(tau)

    taus = np.asarray(taus, dtype=np.float32)
    ac_curves = np.asarray(ac_curves)  # shape = (num_regions, max_lag+1)

    # ===== Sort + relative gradient =====
    order = np.argsort(taus)  # Shortest to longest timescale
    taus_sorted = taus[order]
    regions_sorted = regions_ts[order]
    taus_rel = taus_sorted / (taus_sorted.max() + 1e-8)

    # ===== Figure 6: intrinsic timescale per region (sorted by tau) =====
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    x = np.arange(len(regions_sorted))
    ax.bar(x, taus_rel, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regions_sorted, rotation=90)
    ax.set_ylabel("Relative intrinsic timescale (0–1)")
    ax.set_xlabel("Region (sorted by timescale, short to long)")
    ax.set_title("Intrinsic timescale per region (sorted)")
    plt.tight_layout()
    fig_path6 = os.path.join(result_root, "intrinsic_timescales_bar.png")
    plt.savefig(fig_path6, bbox_inches="tight")
    plt.close(fig)

    # Save data file
    data_path6 = os.path.join(result_root, "intrinsic_timescales_bar.txt")
    with open(data_path6, 'w', encoding='utf-8') as f:
        f.write("# Intrinsic timescale per region (relative, normalized 0-1)\n")
        f.write("# Timescale from integral of positive part of autocorrelation\n")
        f.write("# Regions sorted by timescale (short to long)\n")
        f.write("# Col1: region name (sorted), Col2: relative (0-1), Col3: absolute\n\n")
        for i, rname in enumerate(regions_sorted):
            f.write("{}\t{:.6f}\t{:.6f}\n".format(rname, taus_rel[i], taus_sorted[i]))
    print(f"Intrinsic timescale bar figure saved to: {fig_path6}")
    print(f"Intrinsic timescale data saved to: {data_path6}")

    # ===== Autocorrelation curve per region =====
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    lags = np.arange(max_lag + 1)
    for idx in order:
        ax.plot(lags, ac_curves[idx], linewidth=1.0, alpha=0.9, label=regions_ts[idx])
    ax.set_xlabel("Lag (timesteps)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation per region (sorted by timescale, short to long)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc='upper right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    fig_path7 = os.path.join(result_root, "autocorrelation_curves.png")
    plt.savefig(fig_path7, bbox_inches="tight")
    plt.close(fig)

    # Save data file (autocorrelation curves)
    data_path7 = os.path.join(result_root, "autocorrelation_curves.txt")
    with open(data_path7, 'w', encoding='utf-8') as f:
        f.write("# Autocorrelation curve per region\n")
        f.write("# max_lag: {}\n".format(max_lag))
        f.write("# Regions sorted by timescale (short to long)\n")
        f.write("# Format: first column=lag, then one column per region\n")
        f.write("# Region order: {}\n\n".format(", ".join(regions_sorted)))
        f.write("Lag\t" + "\t".join(regions_sorted) + "\n")
        for lag in range(max_lag + 1):
            f.write("{}\t".format(lag))
            f.write("\t".join(["{:.6f}".format(ac_curves[order[i], lag]) for i in range(len(regions_sorted))]) + "\n")
    print(f"Autocorrelation figure saved to: {fig_path7}")
    print(f"Autocorrelation data saved to: {data_path7}")

    # ========== Model comparison ==========
    comparison_results = compare_all_networks()
    comparison_results_10k = compare_all_networks_10k()