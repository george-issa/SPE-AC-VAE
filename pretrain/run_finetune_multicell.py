"""
Two-cell VAE test on site-Holstein: train one model on (n=1, Omega=1) and
(n=0.5, Omega=2) at beta=10 simultaneously. Goal: test whether forcing the
encoder to handle physically-distinct cells revives engagement (AU > 0).

If AU > 0 and per-cell recovery looks reasonable, the cube approach is
alive and the VAE has work to do that pole-fit can't. If AU = 0, the
architecture is structurally not earning its keep at LW + P=10.

Usage:
    python pretrain/run_finetune_multicell.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model2_leaky import VariationalAutoEncoder2 as VAEModel  # type: ignore
from data_process_real import (  # type: ignore
    _load_holstein_jld2,
    _HOLSTEIN_BETAS,
    _HOLSTEIN_OMEGAS,
    _HOLSTEIN_NS,
    _HOLSTEIN_DTAU,
    _ntau_holstein,
)
from pretrain.pretrain_losses import (  # type: ignore
    KLDivergenceLoss, ChiSquaredLoss,
    SpectralSmoothnessLoss, SpectralPositivityLoss,
    NegativeGreenPenalty, NegativeSecondDerivativePenalty,
    NegativeFourthDerivativePenalty,
    finetune_total_loss, ledoit_wolf_shrinkage, spectral_from_poles,
)
from pretrain.train_finetune import train_finetune  # type: ignore
from pretrain.plot_results import plot_loss_curves  # type: ignore
from pretrain.analyze_multicell import analyze_run  # type: ignore
from utils import MakeOutPath  # type: ignore


# ==========================================================================
# CONFIGURATION
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLSTEIN_JLD2_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "real", "george_325.jld2"
)

# Two cells of the cube — same beta to keep input shape constant.
# (n_idx, omega_idx, beta_idx) with the convention from data_process_real.
#   _HOLSTEIN_NS     = [0.05, 0.10, ..., 1.00]   -> n_idx 0..19,  step 0.05
#   _HOLSTEIN_OMEGAS = [0.5, 1.0, 1.5, 2.0]      -> omega_idx 0..3
#   _HOLSTEIN_BETAS  = [5, 6, ..., 20]           -> beta_idx 0..15
CELLS = [
    {"n_idx": 19, "omega_idx": 0, "beta_idx": 5, "label": "A"},  # n=1.00, w=0.50
    {"n_idx": 19, "omega_idx": 3, "beta_idx": 5, "label": "B"},  # n=1.00, w=2.00
    {"n_idx":  0, "omega_idx": 0, "beta_idx": 5, "label": "C"},  # n=0.05, w=0.50
    {"n_idx":  0, "omega_idx": 3, "beta_idx": 5, "label": "D"},  # n=0.05, w=2.00
]

NUM_POLES   = 10
LATENT_DIM  = 2          # at minimum 1 bit needed (cell A vs B); give 2 for slack
PH_SYMMETRIC = False     # n=0.5 cell will not be even -> PH-on would be wrong
N_NODES     = 256
BATCH_SIZE  = 10         # 200 total samples -> 20 batches/epoch
MODEL_VERSION = "2L"

# Standard fine-tune hyperparameters (matched to the single-cell run).
FINETUNE_EPOCHS = 1000
# Lowered from 1e-3 after the multi-seed sweep showed every seed had
# chi^2_final >> chi^2_best (AdamW overshooting in the rank-deficient
# directions of the per-cell whiteners late in training). Smaller base
# lr keeps the trajectory in the basin once it descends.
FINETUNE_LR = 3e-4
FINETUNE_KL_ANNEAL_EPOCHS = 0
LAMBDA_CHI2  = 1.0
LAMBDA_SMOOTH = 0.0
LAMBDA_POS   = 0.1
ALPHA_KL     = 1e-6
ETA0 = 1.0
ETA2 = 1.0
ETA4 = 0.0
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -20.0
SMOOTHNESS_WMAX =  20.0
SEED = None
if os.environ.get("SWEEP_SEED") not in (None, "", "None", "none"):
    SEED = int(os.environ["SWEEP_SEED"])

# Per-cell covariance regularization mode.
#   False -> raw sample covariance per cell, 1e-12 numerical floor only
#            (no LW, no PCA). Tag: _covpc.
#   True  -> Ledoit-Wolf shrinkage applied independently per cell, then
#            "full" mode whitener on the shrunk C. Tag: _covpclw.
# Env var SWEEP_COV_LW_PER_CELL=1 flips this on without editing the file.
COVARIANCE_LW_PER_CELL = False
if os.environ.get("SWEEP_COV_LW_PER_CELL") not in (None, "", "0", "false", "False"):
    COVARIANCE_LW_PER_CELL = True

# Sweep env var hooks (mirror the single-cell runner so we can multi-seed later).
NUM_POLES   = int(os.environ.get("SWEEP_NUM_POLES",   NUM_POLES))
LATENT_DIM  = int(os.environ.get("SWEEP_LATENT_DIM",  LATENT_DIM))
BATCH_SIZE  = int(os.environ.get("SWEEP_BATCH_SIZE",  BATCH_SIZE))
FINETUNE_EPOCHS = int(os.environ.get("SWEEP_FINETUNE_EPOCHS", FINETUNE_EPOCHS))


# ==========================================================================
# Multi-cell dataset
# ==========================================================================

class MultiCellHolsteinDataset(Dataset):
    """Concatenate G(tau) bins across multiple Holstein cube cells.

    All cells must share the same beta (so L_tau is constant). Each item is
    a float32 tensor of shape (L_tau,). `cell_ids[i]` indexes into `cells`
    and gives the cell-of-origin of sample `i` — used downstream for AU
    diagnostics and per-cell evaluation.
    """

    def __init__(self, jld2_path, cells):
        beta_idxs = {c["beta_idx"] for c in cells}
        if len(beta_idxs) > 1:
            raise ValueError(
                f"All cells must share beta_idx; got {sorted(beta_idxs)}. "
                f"Different beta means different L_tau, which would break the "
                f"conv stack's input shape."
            )
        self.cells = cells
        self.beta = float(_HOLSTEIN_BETAS[cells[0]["beta_idx"]])
        self.dtau = float(_HOLSTEIN_DTAU)
        self.L_tau = _ntau_holstein(self.beta)

        G_r, dos, ws = _load_holstein_jld2(jld2_path)
        self.ws = ws  # (601,) shared across cells

        rows = []
        ids = []
        dos_per_cell = []
        n_per_cell   = []
        omega_per_cell = []
        for cid, c in enumerate(cells):
            raw = G_r[c["n_idx"], c["omega_idx"], c["beta_idx"], :self.L_tau, :]
            rows.append(torch.tensor(raw.T.copy(), dtype=torch.float32))  # (n_bins, L_tau)
            ids.append(torch.full((rows[-1].shape[0],), cid, dtype=torch.long))
            dos_per_cell.append(dos[c["n_idx"], c["omega_idx"], c["beta_idx"]])
            n_per_cell.append(_HOLSTEIN_NS[c["n_idx"]])
            omega_per_cell.append(_HOLSTEIN_OMEGAS[c["omega_idx"]])
        self.data     = torch.cat(rows, dim=0)
        self.cell_ids = torch.cat(ids,  dim=0)
        self.dos_per_cell  = dos_per_cell
        self.n_per_cell    = n_per_cell
        self.omega_per_cell = omega_per_cell

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Returns (G(tau), cell_id) so the training loop can dispatch chi^2
        # to the right cell-specific covariance. The default DataLoader
        # collate handles tuples automatically: a batch yields
        # (Tensor[B, L_tau], Tensor[B] of int64).
        return self.data[idx], int(self.cell_ids[idx])

    def summary(self):
        print(f"MultiCellHolsteinDataset")
        print(f"  beta={self.beta}, L_tau={self.L_tau}, total bins={len(self)}")
        for cid, c in enumerate(self.cells):
            mask = (self.cell_ids == cid).numpy()
            G = self.data[self.cell_ids == cid].numpy()
            print(f"  cell {c['label']} (n={self.n_per_cell[cid]:.2f}, "
                  f"Omega={self.omega_per_cell[cid]:.2f}): "
                  f"{int(mask.sum())} bins, G(0)={G[:,0].mean():.4f}, "
                  f"G(beta-)={G[:,-1].mean():.4f}")


def per_cell_covariances(ds, lw_shrink=False):
    """Per-cell covariance C_c for each cell in `ds`.

    lw_shrink=False (default):
        Raw sample C_c per cell. Rank-deficient by 1..few directions
        (sum-rule constraints); ChiSquaredLoss("full") floors eigenvalues
        at 1e-12 for numerical stability.
    lw_shrink=True:
        Ledoit-Wolf shrinkage applied per cell -> C_c full-rank and
        well-conditioned. Returns the per-cell rho values alongside.
    """
    cell_ids = ds.cell_ids.numpy()
    data     = ds.data.numpy()
    n_cells  = len(ds.cells)
    Cs   = []
    rhos = []
    for c in range(n_cells):
        bins = data[cell_ids == c]                          # (N_bins_c, L_tau)
        if lw_shrink:
            C, rho = ledoit_wolf_shrinkage(bins)
        else:
            C = np.cov(bins.T, ddof=1)
            rho = None
        Cs.append(C)
        rhos.append(rho)
    return Cs, rhos


class PerCellChiSquaredLoss(torch.nn.Module):
    """Routes chi^2 to per-cell ChiSquaredLoss based on `cell_ids`.

    Each cell uses its own raw covariance with no LW / PCA — only the
    1e-10 numerical floor inside ChiSquaredLoss("full"). Inter-cell
    variation is *not* absorbed into the noise model: a cell-A-shaped
    reconstruction scored against cell B's whitener gives a large chi^2,
    so the encoder has a strong gradient to differentiate cells.
    """

    def __init__(self, C_per_cell):
        super().__init__()
        self.chi2_per_cell = torch.nn.ModuleList([
            ChiSquaredLoss(C_c, covariance_estimator="full")
            for C_c in C_per_cell
        ])

    def forward(self, G_recon, G_input, cell_ids):
        total = G_recon.new_zeros(())
        n_total = 0
        for c, chi2_c in enumerate(self.chi2_per_cell):
            mask = (cell_ids == c)
            m = int(mask.sum().item())
            if m > 0:
                total = total + chi2_c(G_recon[mask], G_input[mask]) * m
                n_total += m
        return total / max(n_total, 1)


def train_multicell(
    model, optimizer,
    num_epochs, train_loader, val_loader,
    chi2_fn, kl_fn, smoothness_fn,
    neg_green_fn, neg_second_fn, neg_fourth_fn, positivity_fn,
    lambda_chi2, lambda_smooth, lambda_pos, alpha_kl,
    eta0, eta2, eta4,
    device, out_dir, tag="finetune",
):
    """Multi-cell variant of train_finetune: same loss stack except chi^2
    receives per-sample cell_ids so it can dispatch to per-cell whiteners.
    Strips KL annealing and EMA-anomaly detection (not needed here).
    """
    from tqdm import tqdm
    os.makedirs(f"{out_dir}/model",  exist_ok=True)
    os.makedirs(f"{out_dir}/losses", exist_ok=True)

    keys = ["chi2", "smooth", "pos", "kl",
            "neg_green", "neg_second", "neg_fourth", "train", "val"]
    history = {k: [] for k in keys}
    best_val = float("inf")
    best_epoch = -1
    best_path = f"{out_dir}/model/best_model_{tag}.pth"

    def _step(G_input, cell_ids, deterministic):
        mu, logvar, _, poles, residues, G_recon = model(
            G_input, deterministic=deterministic,
        )
        chi2       = chi2_fn(G_recon, G_input, cell_ids)
        kl         = kl_fn(mu, logvar)
        smoothness = smoothness_fn(poles, residues)
        positivity = positivity_fn(poles, residues)
        neg_g      = neg_green_fn(G_recon)
        neg_g2     = neg_second_fn(G_recon)
        neg_g4     = neg_fourth_fn(G_recon)
        total = (lambda_chi2  * chi2
                 + alpha_kl   * kl
                 + lambda_smooth * smoothness
                 + lambda_pos    * positivity
                 + eta0 * neg_g + eta2 * neg_g2 + eta4 * neg_g4)
        return total, dict(chi2=chi2, smooth=smoothness, pos=positivity,
                           kl=kl, neg_green=neg_g, neg_second=neg_g2,
                           neg_fourth=neg_g4, train=total)

    for epoch in tqdm(range(num_epochs), desc=f"{tag} (multicell)"):
        model.train()
        run_sums = {k: 0.0 for k in keys if k != "val"}
        n_seen = 0
        for G_input, cell_ids in train_loader:
            G_input  = G_input.to(device)
            cell_ids = cell_ids.to(device)
            B = G_input.shape[0]
            total, parts = _step(G_input, cell_ids, deterministic=False)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for k in run_sums:
                run_sums[k] += parts[k].item() * B
            n_seen += B
        for k in run_sums:
            history[k].append(run_sums[k] / n_seen)

        # Val pass: deterministic, same data (no held-out split — matches
        # the canonical single-cell convention in 11_Development_Standards).
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for G_input, cell_ids in val_loader:
                G_input  = G_input.to(device)
                cell_ids = cell_ids.to(device)
                B = G_input.shape[0]
                total, _ = _step(G_input, cell_ids, deterministic=True)
                val_sum += total.item() * B
                n_val += B
        val_loss = val_sum / max(n_val, 1)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    np.save(f"{out_dir}/losses/chi2_losses_{tag}.npy",       np.array(history["chi2"]))
    np.save(f"{out_dir}/losses/smooth_losses_{tag}.npy",     np.array(history["smooth"]))
    np.save(f"{out_dir}/losses/pos_losses_{tag}.npy",        np.array(history["pos"]))
    np.save(f"{out_dir}/losses/kl_losses_{tag}.npy",         np.array(history["kl"]))
    np.save(f"{out_dir}/losses/neg_green_losses_{tag}.npy",  np.array(history["neg_green"]))
    np.save(f"{out_dir}/losses/neg_second_losses_{tag}.npy", np.array(history["neg_second"]))
    np.save(f"{out_dir}/losses/neg_fourth_losses_{tag}.npy", np.array(history["neg_fourth"]))
    np.save(f"{out_dir}/losses/train_losses_{tag}.npy",      np.array(history["train"]))
    np.save(f"{out_dir}/losses/val_losses_{tag}.npy",        np.array(history["val"]))

    return best_val, best_epoch


# ==========================================================================
# MAIN
# ==========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    if SEED is not None:
        set_seed(SEED)

    print(f"Device: {DEVICE}    Seed: {SEED if SEED is not None else 'random'}")
    print(f"Cells: {[c['label'] for c in CELLS]}")
    for c in CELLS:
        n = _HOLSTEIN_NS[c["n_idx"]]
        w = _HOLSTEIN_OMEGAS[c["omega_idx"]]
        b = _HOLSTEIN_BETAS[c["beta_idx"]]
        print(f"  cell {c['label']}: (beta, Omega, n) = ({b}, {w}, {n})")

    ds = MultiCellHolsteinDataset(HOLSTEIN_JLD2_PATH, CELLS)
    ds.summary()

    INPUT_DIM = ds.L_tau
    BETA      = ds.beta

    # --- Output directory ---
    cell_tag = "_".join(c["label"] for c in CELLS)
    sim_label = f"site_holstein_b{BETA:.2f}_{len(CELLS)}cell_{cell_tag}"
    # Per-cell tags:
    #   `_covpc`   = per-cell raw covariance, 1e-12 floor only.
    #   `_covpclw` = per-cell Ledoit-Wolf shrinkage.
    # The covlw-tagged multi-cell runs from earlier in this experiment are
    # joint-LW only at vDEC-1; their params.json holds the truth.
    _cov_tag = "_covpclw" if COVARIANCE_LW_PER_CELL else "_covpc"
    _base_name = (f"finetune_real-{sim_label}"
                  f"_numpoles{NUM_POLES}_z{LATENT_DIM}{_cov_tag}"
                  f"-fresh-v{MODEL_VERSION}")
    _out_root = os.path.join(MAIN_PATH, "out")
    os.makedirs(_out_root, exist_ok=True)
    # Atomic claim on the next run-dir id: listdir + mkdir(exist_ok=False) in
    # a retry loop. Standard `_used_ids = max + 1` races when multiple
    # workers launch in parallel — they all pick the same id and clobber
    # each other (we hit this once already). With exist_ok=False, only one
    # process wins each id; the others FileExistsError out and try again.
    ft_out_dir = None
    for _attempt in range(50):
        _used_ids = [
            int(d[len(_base_name) + 1:])
            for d in os.listdir(_out_root)
            if os.path.isdir(os.path.join(_out_root, d))
            and d.startswith(_base_name + "-")
            and d[len(_base_name) + 1:].isdigit()
        ]
        _next_id   = max(_used_ids, default=0) + 1
        _candidate = os.path.join(_out_root, f"{_base_name}-{_next_id}")
        try:
            os.makedirs(_candidate, exist_ok=False)
            ft_out_dir = _candidate
            sID        = f"fresh-v{MODEL_VERSION}-{_next_id}"
            break
        except FileExistsError:
            continue
    if ft_out_dir is None:
        raise RuntimeError("Could not claim a new run dir after 50 attempts")
    print(f"Output directory: {ft_out_dir}")
    MakeOutPath(ft_out_dir)

    # --- Model ---
    model = VAEModel(
        input_dim=INPUT_DIM, num_poles=NUM_POLES,
        beta=BETA, N_nodes=N_NODES,
        latent_dim=LATENT_DIM, ph_symmetric=PH_SYMMETRIC,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE trainable params: {n_params:,}")

    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = train_loader

    # --- Loss ---
    C_per_cell, lw_rhos = per_cell_covariances(ds, lw_shrink=COVARIANCE_LW_PER_CELL)
    print(f"Per-cell covariance mode: "
          f"{'LW per cell' if COVARIANCE_LW_PER_CELL else 'raw per cell'}")
    for cid, Cc in enumerate(C_per_cell):
        eigs = np.linalg.eigvalsh(Cc)
        rho_str = (f"  rho={lw_rhos[cid]:.4f}" if lw_rhos[cid] is not None else "")
        print(f"Per-cell C[{ds.cells[cid]['label']}]: shape {Cc.shape}, "
              f"eigenvalue range [{eigs.min():.2e}, {eigs.max():.2e}], "
              f"rank-deficient by {int((eigs < 1e-12).sum())}{rho_str}")
    # Joint covariance kept only for the variance-weighted negative-G
    # penalties (those use diag(C)); the principal change is in chi^2.
    C_joint = np.cov(ds.data.numpy().T, ddof=1)

    kl_fn         = KLDivergenceLoss().to(DEVICE)
    chi2_fn       = PerCellChiSquaredLoss(C_per_cell).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX,
    ).to(DEVICE)
    positivity_fn = SpectralPositivityLoss().to(DEVICE)
    neg_green_fn  = NegativeGreenPenalty(C_joint).to(DEVICE)
    neg_second_fn = NegativeSecondDerivativePenalty(C_joint).to(DEVICE)
    neg_fourth_fn = NegativeFourthDerivativePenalty(C_joint).to(DEVICE)

    # --- params.json ---
    # Note: under per-cell whitening the chi^2 calibration is no longer
    # 1-at-noise-floor (the LW E[chi2] formula in 13_LW_Chi2_Calibration
    # does not apply). Compare runs by *change* in chi^2, not by value.
    params = {
        "sID": sID,
        "MODEL_VERSION": MODEL_VERSION,
        "ARCHITECTURE": "VAE_multicell",
        "NUM_POLES": NUM_POLES,
        "INPUT_DIM": INPUT_DIM,
        "N_NODES": N_NODES,
        "LATENT_DIM": LATENT_DIM,
        "PH_SYMMETRIC": PH_SYMMETRIC,
        "BETA": BETA,
        "DTAU": ds.dtau,
        "BATCH_SIZE": BATCH_SIZE,
        "N_TRAINABLE_PARAMS": n_params,
        "DATA_SOURCE": "holstein_jld2_multicell",
        "DATA_PATH": HOLSTEIN_JLD2_PATH,
        "CELLS": [{"n_idx": c["n_idx"], "omega_idx": c["omega_idx"],
                   "beta_idx": c["beta_idx"], "label": c["label"],
                   "n": _HOLSTEIN_NS[c["n_idx"]],
                   "omega": _HOLSTEIN_OMEGAS[c["omega_idx"]],
                   "beta":  _HOLSTEIN_BETAS[c["beta_idx"]]}
                  for c in CELLS],
        "FINETUNE_EPOCHS": FINETUNE_EPOCHS,
        "FINETUNE_LR": FINETUNE_LR,
        "FINETUNE_PATIENCE": FINETUNE_EPOCHS + 1,
        "FINETUNE_KL_ANNEAL_EPOCHS": FINETUNE_KL_ANNEAL_EPOCHS,
        "LAMBDA_CHI2": LAMBDA_CHI2,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "LAMBDA_POS": LAMBDA_POS,
        "ALPHA_KL": ALPHA_KL,
        "ETA0": ETA0, "ETA2": ETA2, "ETA4": ETA4,
        "SMOOTHNESS_NW": SMOOTHNESS_NW,
        "SMOOTHNESS_WMIN": SMOOTHNESS_WMIN,
        "SMOOTHNESS_WMAX": SMOOTHNESS_WMAX,
        "SEED": SEED,
        "COVARIANCE_ESTIMATOR": "per_cell_lw" if COVARIANCE_LW_PER_CELL else "per_cell_full",
        "LW_RHO_PER_CELL": (
            [None if r is None else float(r) for r in lw_rhos]
            if COVARIANCE_LW_PER_CELL else None
        ),
        "LOSS_MODE": "raw",
    }
    with open(f"{ft_out_dir}/params.json", "w") as _f:
        json.dump(params, _f, indent=2)

    # --- Optimizer + train (custom multi-cell loop) ---
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.0)
    best_val_loss, best_epoch = train_multicell(
        model=model, optimizer=optimizer,
        num_epochs=FINETUNE_EPOCHS,
        train_loader=train_loader, val_loader=val_loader,
        chi2_fn=chi2_fn, kl_fn=kl_fn, smoothness_fn=smoothness_fn,
        neg_green_fn=neg_green_fn, neg_second_fn=neg_second_fn,
        neg_fourth_fn=neg_fourth_fn, positivity_fn=positivity_fn,
        lambda_chi2=LAMBDA_CHI2, lambda_smooth=LAMBDA_SMOOTH,
        lambda_pos=LAMBDA_POS, alpha_kl=ALPHA_KL,
        eta0=ETA0, eta2=ETA2, eta4=ETA4,
        device=DEVICE, out_dir=ft_out_dir, tag="finetune",
    )

    chi2_losses = np.load(f"{ft_out_dir}/losses/chi2_losses_finetune.npy")
    best_chi2   = float(np.min(chi2_losses))
    best_chi2_ep = int(np.argmin(chi2_losses)) + 1
    final_chi2  = float(chi2_losses[-1])
    print(f"\nDone. chi2 best = {best_chi2:.4f} @ epoch {best_chi2_ep}, "
          f"final = {final_chi2:.4f}")

    model.load_state_dict(torch.load(
        f"{ft_out_dir}/model/best_model_finetune.pth", weights_only=True,
    ))

    # --- Eval pass: per-sample mu, A_i, grouped by cell ---
    model.eval()
    G_input_all, G_recon_all, poles_all, residues_all = [], [], [], []
    mu_all, logvar_all = [], []
    cell_ids_all = []
    eval_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x, cid in eval_loader:
            x = x.to(DEVICE)
            mu, logvar, _, poles, residues, G_recon = model(x, deterministic=True)
            G_input_all.append(x.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())
            mu_all.append(mu.cpu())
            logvar_all.append(logvar.cpu())
            cell_ids_all.append(cid)

    G_input_all  = torch.cat(G_input_all,  dim=0)
    G_recon_all  = torch.cat(G_recon_all,  dim=0)
    poles_all    = torch.cat(poles_all,    dim=0)
    residues_all = torch.cat(residues_all, dim=0)
    mu_all       = torch.cat(mu_all,       dim=0)
    logvar_all   = torch.cat(logvar_all,   dim=0)
    cell_ids_all = torch.cat(cell_ids_all, dim=0)

    # AU diagnostic
    var_mu = mu_all.numpy().var(axis=0, ddof=0)
    sigma  = torch.exp(0.5 * torch.clamp(logvar_all, -25, 25)).numpy()
    kl_per_dim = 0.5 * (mu_all.numpy() ** 2 + sigma ** 2 - 1.0
                        - logvar_all.numpy()).mean(axis=0)
    n_active = int((var_mu > 1e-2).sum())
    print(f"\nAU = {n_active}/{LATENT_DIM}  Var_x[mu] = {var_mu}")
    print(f"KL/dim = {kl_per_dim}  KL_total = {kl_per_dim.sum():.4f}")

    # Per-cell mu summary
    print("\nPer-cell encoder means:")
    for cid, c in enumerate(CELLS):
        mask = (cell_ids_all == cid).numpy()
        mu_c = mu_all.numpy()[mask]
        print(f"  cell {c['label']}: mu_bar = {mu_c.mean(axis=0)}, "
              f"std = {mu_c.std(axis=0)}")

    # Save summary.pt
    cell_meta = [{"label": c["label"],
                  "n":     _HOLSTEIN_NS[c["n_idx"]],
                  "omega": _HOLSTEIN_OMEGAS[c["omega_idx"]],
                  "beta":  _HOLSTEIN_BETAS[c["beta_idx"]],
                  "dos_ref": torch.tensor(ds.dos_per_cell[cid], dtype=torch.float64)}
                 for cid, c in enumerate(CELLS)]
    torch.save({
        "inputs": G_input_all, "recon": G_recon_all,
        "poles": poles_all,    "residues": residues_all,
        "mu": mu_all,          "logvar": logvar_all,
        "cell_ids": cell_ids_all,
        "cells": cell_meta,
        "ws_ref": torch.tensor(ds.ws, dtype=torch.float64),
        "beta": BETA, "Ltau": INPUT_DIM, "num_poles": NUM_POLES,
        "sID": sID, "architecture": "VAE_multicell",
        "best_chi2": best_chi2, "best_chi2_epoch": best_chi2_ep,
        "final_chi2": final_chi2, "n_epochs": int(len(chi2_losses)),
        "active_units":   n_active,
        "latent_dim":     LATENT_DIM,
        "var_mu_per_dim": torch.tensor(var_mu),
        "kl_per_dim":     torch.tensor(kl_per_dim),
        "kl_total_avg":   float(kl_per_dim.sum()),
        "noise_var": float(np.mean(np.diag(C_joint))),
        "data_source": "holstein_jld2_multicell",
        "sim_name": sim_label,
    }, f"{ft_out_dir}/summary.pt")

    # Brief run summary
    with open(f"{ft_out_dir}/run_summary.txt", "w") as f:
        f.write(f"VAE multi-cell run\n")
        f.write(f"sID: {sID}\n")
        f.write(f"cells: {[c['label'] for c in CELLS]}\n")
        for cid, c in enumerate(CELLS):
            f.write(f"  {c['label']}: (beta={_HOLSTEIN_BETAS[c['beta_idx']]}, "
                    f"Omega={_HOLSTEIN_OMEGAS[c['omega_idx']]}, "
                    f"n={_HOLSTEIN_NS[c['n_idx']]})\n")
        f.write(f"\nbest chi2 = {best_chi2:.4f} @ epoch {best_chi2_ep}\n")
        f.write(f"final chi2 = {final_chi2:.4f}\n")
        f.write(f"AU = {n_active}/{LATENT_DIM}\n")
        f.write(f"KL_total = {kl_per_dim.sum():.4f}\n")
        f.write(f"Var_x[mu] = {var_mu}\n")
        for cid, c in enumerate(CELLS):
            mask = (cell_ids_all == cid).numpy()
            mu_c = mu_all.numpy()[mask]
            f.write(f"cell {c['label']}: mu_bar = {mu_c.mean(axis=0)}\n")

    # Post-training plots: loss curves + analyze_multicell (latent scatter,
    # spectra per cell, Greens reconstructions, plus traversal panels if
    # we have exactly 2 cells). Every multi-cell run dir now gets a
    # complete plots/ subdir.
    print("\n--- Post-training plots ---")
    os.makedirs(f"{ft_out_dir}/plots", exist_ok=True)
    plot_loss_curves(
        f"{ft_out_dir}/losses", tag="finetune",
        save_path=f"{ft_out_dir}/plots/loss_curves_finetune.pdf",
        params=params,
    )
    analyze_run(ft_out_dir)

    print(f"\nOutput dir: {ft_out_dir}")


if __name__ == "__main__":
    main()
