"""
Data loader for real QMC output from SmoQyDQMC (new format).

Each SmoQyDQMC simulation folder contains per-bin JLD2 files at:
    time-displaced/greens/position/bin-{n}_pID-0.jld2

Each file has a top-level 'correlations' dataset (object reference) that
resolves to a structured array of shape (L_tau+1, N_R2, N_R1) with fields
('re', 'im').  The last tau index (L_tau) is tau=beta and is excluded.

Physical parameters (beta, dtau, L_tau) are read automatically from
model_summary.toml inside the simulation folder.

Usage
-----
    from data_process_real import QMCPositionDataset, load_covariance_from_qmc_position

    sim_dir = "Data/datasets/real/bond_holstein_square_w1.00_a2.5820_b0.35_L12-1"

    dataset = QMCPositionDataset(sim_dir, r1=0, r2=0)
    dataset.summary()            # prints beta, dtau, N_bins, shape
    dataset[0].shape             # (L_tau,)

    C = load_covariance_from_qmc_position(sim_dir, r1=0, r2=0)
    # C: ndarray (L_tau, L_tau) — drop-in for load_covariance_from_dqmc
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# model_summary.toml reader
# ---------------------------------------------------------------------------

def read_model_params(sim_dir):
    """Read beta, dtau, L_tau from model_summary.toml.

    Parameters
    ----------
    sim_dir : str — path to the SmoQyDQMC simulation folder

    Returns
    -------
    dict with keys 'beta', 'dtau', 'L_tau' (all float/int)
    """
    toml_path = os.path.join(sim_dir, "model_summary.toml")
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"model_summary.toml not found in {sim_dir}")

    params = {}
    with open(toml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("beta"):
                params["beta"] = float(line.split("=")[1])
            elif line.startswith("dtau"):
                params["dtau"] = float(line.split("=")[1])
            elif line.startswith("L_tau"):
                params["L_tau"] = int(line.split("=")[1])

    for key in ("beta", "dtau", "L_tau"):
        if key not in params:
            raise ValueError(f"'{key}' not found in {toml_path}")

    return params


# ---------------------------------------------------------------------------
# JLD2 helpers
# ---------------------------------------------------------------------------

def _deref(f, x):
    import h5py
    if isinstance(x, (h5py.Reference, h5py.h5r.Reference)):
        return f[x]
    return x


def _sorted_bin_paths(bin_dir):
    """Return full paths to bin JLD2 files sorted numerically."""
    files = [
        f for f in os.listdir(bin_dir)
        if re.match(r"bin-\d+_pID-\d+\.jld2", f)
    ]
    return [
        os.path.join(bin_dir, f)
        for f in sorted(files, key=lambda s: int(re.search(r"bin-(\d+)_", s).group(1)))
    ]


def _read_bin(jld2_path, L_tau, r2, r1):
    """Read G(R=(r1,r2), tau) for one bin file. Returns ndarray (L_tau,)."""
    import h5py

    with h5py.File(jld2_path, "r") as f:
        corr_arr = f["correlations"][()]          # object array shape (1,)
        corr = _deref(f, corr_arr.flat[0])[()]    # (L_tau+1, N_R2, N_R1) structured
        return corr["re"][:L_tau, r2, r1].copy()  # (L_tau,)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_position_gbins(sim_dir, r1=0, r2=0):
    """Extract all position-space G(R, tau) bins for R=(r1, r2).

    Reads beta, dtau, L_tau from model_summary.toml automatically.

    Parameters
    ----------
    sim_dir : str — SmoQyDQMC simulation folder
    r1, r2  : int — real-space displacement indices (default 0, 0)

    Returns
    -------
    G_bins : ndarray (N_bins, L_tau), float64
    params : dict with 'beta', 'dtau', 'L_tau'
    """
    params  = read_model_params(sim_dir)
    L_tau   = params["L_tau"]
    bin_dir = os.path.join(sim_dir, "time-displaced", "greens", "position")

    if not os.path.isdir(bin_dir):
        raise FileNotFoundError(f"Position greens bin directory not found: {bin_dir}")

    bin_paths = _sorted_bin_paths(bin_dir)
    if not bin_paths:
        raise FileNotFoundError(f"No bin JLD2 files found in {bin_dir}")

    bins = [_read_bin(p, L_tau, r2, r1) for p in bin_paths]
    return np.stack(bins, axis=0), params   # (N_bins, L_tau)


def _covariance(G_bins):
    N  = G_bins.shape[0]
    dG = G_bins - G_bins.mean(axis=0, keepdims=True)
    return (dG.T @ dG) / (N - 1)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class QMCPositionDataset(Dataset):
    """Binned G(R, τ) from a SmoQyDQMC simulation folder.

    Each item is a float32 tensor of shape (L_tau,) — the real part of
    G(R=(r1,r2), τ) for one QMC bin.  Drop-in for GreenFunctionDataset.

    beta, dtau, and L_tau are read automatically from model_summary.toml.

    Parameters
    ----------
    sim_dir : str — path to the SmoQyDQMC simulation folder
    r1, r2  : int — real-space displacement indices (default 0, 0)
    """

    def __init__(self, sim_dir, r1=0, r2=0):
        self.sim_dir = sim_dir
        self.r1, self.r2 = r1, r2

        G_bins, params = extract_position_gbins(sim_dir, r1=r1, r2=r2)
        self.beta  = params["beta"]
        self.dtau  = params["dtau"]
        self.L_tau = params["L_tau"]
        self.data  = torch.tensor(G_bins, dtype=torch.float32)   # (N_bins, L_tau)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def summary(self):
        G = self.data.numpy()
        print(f"QMCPositionDataset")
        print(f"  sim_dir : {self.sim_dir}")
        print(f"  beta={self.beta}, dtau={self.dtau}, L_tau={self.L_tau}")
        print(f"  R=({self.r1}, {self.r2})")
        print(f"  N_bins={len(self)}, shape={tuple(self.data.shape)}")
        print(f"  G(tau=0):  mean={G[:,0].mean():.6f}  std={G[:,0].std():.2e}")
        print(f"  G(tau=-1): mean={G[:,-1].mean():.6f}  std={G[:,-1].std():.2e}")


# ---------------------------------------------------------------------------
# Covariance helper
# ---------------------------------------------------------------------------

def load_covariance_from_qmc_position(sim_dir, r1=0, r2=0):
    """Sample covariance matrix from position-space QMC bins.

    Drop-in replacement for synthetic_data.load_covariance_from_dqmc.

    Parameters
    ----------
    sim_dir : str — SmoQyDQMC simulation folder
    r1, r2  : int — real-space displacement (default 0, 0)

    Returns
    -------
    C : ndarray (L_tau, L_tau)
    """
    G_bins, _ = extract_position_gbins(sim_dir, r1=r1, r2=r2)
    return _covariance(G_bins)


# ---------------------------------------------------------------------------
# Holstein model JLD2 dataset (single-file format)
# ---------------------------------------------------------------------------

# Physical grid — fixed across the entire Holstein dataset
_HOLSTEIN_DTAU = 0.1
_HOLSTEIN_NS     = [round(v, 10) for v in [i * 0.05 for i in range(1, 21)]]   # 0.05..1.00 (20)
_HOLSTEIN_OMEGAS = [0.5, 1.0, 1.5, 2.0]                                        # 4
_HOLSTEIN_BETAS  = [float(b) for b in range(5, 21)]                            # 5..20 (16)


def _ntau_holstein(beta):
    """Number of valid tau points for a given beta (dtau=0.1)."""
    return int(round(beta / _HOLSTEIN_DTAU)) + 1


def _load_holstein_jld2(jld2_path):
    """Load the Holstein JLD2 file and return the three arrays.

    Returns
    -------
    G_r  : ndarray (20, 4, 16, 201, 100) — padded tau x bins
    dos  : ndarray (20, 4, 16, 601)
    ws   : ndarray (601,)
    """
    import h5py

    with h5py.File(jld2_path, "r") as f:
        G_r = f["G_r"][()]    # (20, 4, 16, 201, 100)
        dos = f["dos"][()]    # (20, 4, 16, 601)
        ws  = f["ws"][()]     # (601,)
    return G_r, dos, ws


class HolsteinJLD2Dataset(Dataset):
    """Binned G(τ) from the pre-packed Holstein JLD2 file.

    Mirrors the QMCPositionDataset interface so run_finetune.py needs
    only a one-line change to switch data sources.

    Each item is a float32 tensor of shape (L_tau,).

    Parameters
    ----------
    jld2_path  : str   — path to the Holstein JLD2 file
    n_idx      : int   — filling index  (0..19), n = (n_idx+1)*0.05
    omega_idx  : int   — phonon freq index (0..3), Ω = 0.5 + omega_idx*0.5
    beta_idx   : int   — temperature index (0..15), β = 5 + beta_idx
    """

    DTAU = _HOLSTEIN_DTAU

    def __init__(self, jld2_path, n_idx, omega_idx, beta_idx):
        self.jld2_path  = jld2_path
        self.n_idx      = n_idx
        self.omega_idx  = omega_idx
        self.beta_idx   = beta_idx

        self.beta  = _HOLSTEIN_BETAS[beta_idx]
        self.omega = _HOLSTEIN_OMEGAS[omega_idx]
        self.n     = _HOLSTEIN_NS[n_idx]
        self.dtau  = self.DTAU
        self.L_tau = _ntau_holstein(self.beta)

        G_r, dos, ws = _load_holstein_jld2(jld2_path)

        # G_r[n_idx, omega_idx, beta_idx] → (201, 100); trim to valid taus
        raw = G_r[n_idx, omega_idx, beta_idx, :self.L_tau, :]  # (L_tau, 100)
        # Each column is one bin; transpose to (100, L_tau)
        self.data = torch.tensor(raw.T.copy(), dtype=torch.float32)

        # Cache reference DOS for later comparison
        self.dos = dos[n_idx, omega_idx, beta_idx]   # (601,)
        self.ws  = ws                                  # (601,)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def summary(self):
        G = self.data.numpy()
        print(f"HolsteinJLD2Dataset")
        print(f"  file    : {self.jld2_path}")
        print(f"  n={self.n:.2f}  Ω={self.omega:.1f}  β={self.beta:.1f}")
        print(f"  dtau={self.dtau}  L_tau={self.L_tau}")
        print(f"  N_bins={len(self)}, shape={tuple(self.data.shape)}")
        print(f"  G(tau=0):  mean={G[:,0].mean():.6f}  std={G[:,0].std():.2e}")
        print(f"  G(tau=-1): mean={G[:,-1].mean():.6f}  std={G[:,-1].std():.2e}")


def load_covariance_from_holstein_jld2(jld2_path, n_idx, omega_idx, beta_idx):
    """Sample covariance matrix for one Holstein simulation point.

    Drop-in replacement for load_covariance_from_qmc_position.

    Returns
    -------
    C : ndarray (L_tau, L_tau)
    """
    G_r, _, _ = _load_holstein_jld2(jld2_path)
    beta  = _HOLSTEIN_BETAS[beta_idx]
    L_tau = _ntau_holstein(beta)
    raw   = G_r[n_idx, omega_idx, beta_idx, :L_tau, :]   # (L_tau, 100)
    G_bins = raw.T.copy()                                  # (100, L_tau)
    return _covariance(G_bins)


# ---------------------------------------------------------------------------
# SmoQyDQMC v2.x dataset  (bins/bins_pID-{n}.h5 format)
# ---------------------------------------------------------------------------

def _sorted_pID_paths(sim_dir):
    """Return sorted paths to bins_pID-*.h5 files."""
    bins_dir = os.path.join(sim_dir, "bins")
    files = [f for f in os.listdir(bins_dir) if re.match(r"bins_pID-\d+\.h5", f)]
    return [
        os.path.join(bins_dir, f)
        for f in sorted(files, key=lambda s: int(re.search(r"pID-(\d+)", s).group(1)))
    ]


def extract_greens_bins_v2(sim_dir, r1=0, r2=0):
    """Extract G(R=(r1,r2), τ) bins from SmoQyDQMC v2.x output.

    Reads from bins/bins_pID-*.h5 files.
    Shape of POSITION dataset: (N_bins, Lx, Ly, L_tau+1, N_pairs).

    Parameters
    ----------
    sim_dir : str
    r1, r2  : int — spatial displacement indices (default 0, 0)

    Returns
    -------
    G_bins : ndarray (N_total_bins, L_tau), float64
    params : dict with 'beta', 'dtau', 'L_tau'
    """
    import h5py

    params = read_model_params(sim_dir)
    L_tau  = params["L_tau"]

    all_bins = []
    for path in _sorted_pID_paths(sim_dir):
        with h5py.File(path, "r") as f:
            # Julia shape (column-major): (N_bins, Lx, Ly, L_tau+1, N_pairs)
            # Python/h5py shape (row-major, reversed): (N_pairs, L_tau+1, Ly, Lx, N_bins)
            data = f["CORRELATIONS/STANDARD/TIME-DISPLACED/greens/POSITION"][()]
            g = data[0, :L_tau, r2, r1, :].real.T   # (N_bins, L_tau)
            all_bins.append(g)

    return np.concatenate(all_bins, axis=0), params   # (N_total_bins, L_tau)


class SmoQyV2Dataset(Dataset):
    """Binned G(R, τ) from a SmoQyDQMC v2.x simulation folder.

    Reads from bins/bins_pID-*.h5 (new format introduced in v2.0).
    Drop-in replacement for QMCPositionDataset.

    Each item is a float32 tensor of shape (L_tau,).

    Parameters
    ----------
    sim_dir : str — SmoQyDQMC simulation folder
    r1, r2  : int — spatial displacement indices (default 0, 0)
    """

    def __init__(self, sim_dir, r1=0, r2=0):
        self.sim_dir    = sim_dir
        self.r1, self.r2 = r1, r2

        G_bins, params  = extract_greens_bins_v2(sim_dir, r1=r1, r2=r2)
        self.beta       = params["beta"]
        self.dtau       = params["dtau"]
        self.L_tau      = params["L_tau"]
        self.data       = torch.tensor(G_bins, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def summary(self):
        G = self.data.numpy()
        print(f"SmoQyV2Dataset")
        print(f"  sim_dir : {self.sim_dir}")
        print(f"  beta={self.beta}, dtau={self.dtau}, L_tau={self.L_tau}")
        print(f"  R=({self.r1}, {self.r2})")
        print(f"  N_bins={len(self)}, shape={tuple(self.data.shape)}")
        print(f"  G(tau=0):  mean={G[:,0].mean():.6f}  std={G[:,0].std():.2e}")
        print(f"  G(tau=-1): mean={G[:,-1].mean():.6f}  std={G[:,-1].std():.2e}")


def load_covariance_v2(sim_dir, r1=0, r2=0):
    """Sample covariance matrix from SmoQyDQMC v2.x bins.

    Drop-in replacement for load_covariance_from_qmc_position.
    """
    G_bins, _ = extract_greens_bins_v2(sim_dir, r1=r1, r2=r2)
    return _covariance(G_bins)
