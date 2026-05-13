"""
Fine-tune the encoder-less DecoderOnly variant on the same dataset / loss
stack as run_finetune.py. Differs only in the model: no encoder, no latent
space, KL = 0 by construction.

Mirrors run_finetune.py's CLI surface (env vars, output naming, summary
emission) so existing post-hoc tooling (benchmark_runs.py, MaxEnt
comparison, plot_results) works without changes. Dropped: LATENT_DIM,
LOAD_PRETRAIN (Stage-1 trains an encoder, irrelevant here), AU diagnostic
(always 0/1 by shim).

Usage:
    python pretrain/run_finetune_decoder_only.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_VERSION = "DEC"
from decoder_only import DecoderOnly as VAEModel  # type: ignore

from pretrain.plot_results import (  # type: ignore
    plot_finetune_eval, plot_loss_curves,
)
from pretrain.synthetic_data import (  # type: ignore
    load_covariance_from_dqmc,
)
from data_process_real import (  # type: ignore
    SmoQyV2Dataset,
    load_covariance_v2,
    extract_greens_bins_v2,
    read_model_params,
    HolsteinJLD2Dataset,
    load_covariance_from_holstein_jld2,
    _load_holstein_jld2,
    _HOLSTEIN_BETAS,
    _HOLSTEIN_OMEGAS,
    _HOLSTEIN_NS,
    _HOLSTEIN_DTAU,
    _ntau_holstein,
)
from pretrain.pretrain_losses import (  # type: ignore
    KLDivergenceLoss,
    ChiSquaredLoss,
    Chi2FloorTransform,
    Chi2OneSidedBarrier,
    SpectralSmoothnessLoss,
    SpectralPositivityLoss,
    NegativeGreenPenalty,
    NegativeSecondDerivativePenalty,
    NegativeFourthDerivativePenalty,
    finetune_total_loss,
    ledoit_wolf_shrinkage,
    spectral_from_poles,
)
from pretrain.train_finetune import train_finetune  # type: ignore

from data_process import GreenFunctionDataset  # type: ignore
from utils import MakeOutPath  # type: ignore

# ==========================================================================
# CONFIGURATION
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_SOURCE = "holstein_jld2"

NUM_POLES = 10
N_NODES   = 256
BATCH_SIZE = 5
PH_SYMMETRIC = False

NUM_POLES = int(os.environ.get("SWEEP_NUM_POLES", NUM_POLES))
BATCH_SIZE = int(os.environ.get("SWEEP_BATCH_SIZE", BATCH_SIZE))
if os.environ.get("SWEEP_PH_SYMMETRIC"):
    PH_SYMMETRIC = os.environ["SWEEP_PH_SYMMETRIC"].strip().lower() in ("1", "true", "yes")

# --- chi^2 / covariance config (matches run_finetune.py defaults) ---
COVARIANCE_ESTIMATOR = "ledoit_wolf"
VARIANCE_THRESHOLD   = 0.999
LOSS_MODE            = "raw"
FLOOR_TARGET         = 1.0
FLOOR_DELTA          = 0.1
FLOOR_WARMUP_THRESH  = 5.0
BARRIER_LAMBDA       = 50.0
if os.environ.get("SWEEP_COVARIANCE_ESTIMATOR"):
    COVARIANCE_ESTIMATOR = os.environ["SWEEP_COVARIANCE_ESTIMATOR"].strip()
if os.environ.get("SWEEP_VARIANCE_THRESHOLD"):
    VARIANCE_THRESHOLD = float(os.environ["SWEEP_VARIANCE_THRESHOLD"])
if os.environ.get("SWEEP_LOSS_MODE"):
    LOSS_MODE = os.environ["SWEEP_LOSS_MODE"].strip()

# --------------------------------------------------------------------------
# Synthetic data (DATA_SOURCE = "synthetic")
# --------------------------------------------------------------------------
SPECTRAL_TYPE = "gaussian_double"
NOISE_S = 1e-04
NOISE_XI = 0.5
INPUT_ID = "inputs-8"
SYNTHETIC_DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "synthetic",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "synthetic",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --------------------------------------------------------------------------
# Real QMC data (DATA_SOURCE = "real")
# --------------------------------------------------------------------------
QMC_SIM_DIR = os.path.join(
    MAIN_PATH, "Data", "datasets", "real",
    "hubbard_square_U8.00_mu0.00_L4_b6.00-1"
)

# --------------------------------------------------------------------------
# Site-Holstein cube (DATA_SOURCE = "holstein_jld2")
# --------------------------------------------------------------------------
HOLSTEIN_JLD2_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "real", "george_325.jld2"
)
HOLSTEIN_BETA  = 10.0
HOLSTEIN_OMEGA = 1.0
HOLSTEIN_N     = 1.0


def _holstein_idx(value, grid, label):
    arr = np.asarray(grid, dtype=float)
    i   = int(np.argmin(np.abs(arr - float(value))))
    if abs(arr[i] - float(value)) > 1e-6:
        raise ValueError(f"{label}={value} not in grid {grid}")
    return i


# Shared physical parameters
if DATA_SOURCE == "real":
    _qmc_params = read_model_params(QMC_SIM_DIR)
    BETA      = _qmc_params["beta"]
    DTAU      = _qmc_params["dtau"]
    INPUT_DIM = _qmc_params["L_tau"]
elif DATA_SOURCE == "holstein_jld2":
    BETA_IDX  = _holstein_idx(HOLSTEIN_BETA,  _HOLSTEIN_BETAS,  "HOLSTEIN_BETA")
    OMEGA_IDX = _holstein_idx(HOLSTEIN_OMEGA, _HOLSTEIN_OMEGAS, "HOLSTEIN_OMEGA")
    N_IDX     = _holstein_idx(HOLSTEIN_N,     _HOLSTEIN_NS,     "HOLSTEIN_N")
    BETA      = float(HOLSTEIN_BETA)
    DTAU      = float(_HOLSTEIN_DTAU)
    INPUT_DIM = _ntau_holstein(BETA)
else:
    BETA      = 10.0
    DTAU      = 0.05
    INPUT_DIM = int(round(BETA / DTAU))

FINETUNE_EPOCHS   = 1000
FINETUNE_EPOCHS   = int(os.environ.get("SWEEP_FINETUNE_EPOCHS", FINETUNE_EPOCHS))
FINETUNE_LR       = 1e-3
FINETUNE_PATIENCE = FINETUNE_EPOCHS + 1
FINETUNE_KL_ANNEAL_EPOCHS = 0

LAMBDA_CHI2  = 1.0
LAMBDA_SMOOTH = 0.0
LAMBDA_POS   = 0.1
ALPHA_KL     = 0.0          # No KL term — model has no encoder; KL = 0 anyway via shim
ETA0         = 1.0
ETA2         = 1.0
ETA4         = 0.0

USE_SCHEDULER = False
LR_FACTOR     = 0.5
LR_PATIENCE   = 6
LR_MIN        = 1e-6

SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -20.0
SMOOTHNESS_WMAX =  20.0

SEED   = None
if os.environ.get("SWEEP_SEED") not in (None, "", "None", "none"):
    SEED = int(os.environ["SWEEP_SEED"])
DO_PLOT = True


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
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED if SEED is not None else 'disabled (random)'}")
    print(f"Model: {MODEL_VERSION} (DecoderOnly)  NUM_POLES={NUM_POLES}  "
          f"PH={'on' if PH_SYMMETRIC else 'off'}  INPUT_DIM={INPUT_DIM}")

    model = VAEModel(
        input_dim=INPUT_DIM,
        num_poles=NUM_POLES,
        beta=BETA,
        N_nodes=N_NODES,
        ph_symmetric=PH_SYMMETRIC,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- Output directory ---
    if DATA_SOURCE == "real":
        _out_label = f"real-{os.path.basename(QMC_SIM_DIR)}"
    elif DATA_SOURCE == "holstein_jld2":
        _out_label = (
            f"real-site_holstein_b{HOLSTEIN_BETA:.2f}_w{HOLSTEIN_OMEGA:.2f}_n{HOLSTEIN_N:.2f}"
        )
    else:
        _out_label = f"synthetic-{SPECTRAL_TYPE}_s{NOISE_S:.0e}_xi{NOISE_XI}"
    _ph_tag    = "_ph" if PH_SYMMETRIC else ""
    _cov_tag   = "_covlw" if COVARIANCE_ESTIMATOR == "ledoit_wolf" else ""
    _loss_tag  = (
        "_lossfloor"   if LOSS_MODE == "warmup_floored" else
        "_lossbarrier" if LOSS_MODE == "barrier"        else
        ""
    )
    # `_dec` slot replaces the `_z{N}` slot of the VAE runner: same position
    # in the path, makes it visually obvious which architecture produced the
    # run when scanning out/.
    _base_name = f"finetune_{_out_label}_numpoles{NUM_POLES}{_ph_tag}_dec{_cov_tag}{_loss_tag}-fresh-v{MODEL_VERSION}"
    _out_root  = os.path.join(MAIN_PATH, "out")
    os.makedirs(_out_root, exist_ok=True)
    _used_ids  = [
        int(d[len(_base_name) + 1:])
        for d in os.listdir(_out_root)
        if os.path.isdir(os.path.join(_out_root, d))
        and d.startswith(_base_name + "-")
        and d[len(_base_name) + 1:].isdigit()
    ]
    _next_id   = max(_used_ids, default=0) + 1
    sID        = f"fresh-v{MODEL_VERSION}-{_next_id}"
    ft_out_dir = os.path.join(_out_root, f"{_base_name}-{_next_id}")
    print(f"Output directory: {ft_out_dir}")
    MakeOutPath(ft_out_dir)

    # --- Dataset + covariance ---
    lw_rho = None
    if DATA_SOURCE == "real":
        print(f"Loading real QMC dataset from: {QMC_SIM_DIR}")
        ft_dataset = SmoQyV2Dataset(QMC_SIM_DIR, r1=0, r2=0)
        ft_dataset.summary()
        if COVARIANCE_ESTIMATOR == "ledoit_wolf":
            G_bins, _ = extract_greens_bins_v2(QMC_SIM_DIR, r1=0, r2=0)
            C, lw_rho = ledoit_wolf_shrinkage(G_bins)
            print(f"Ledoit-Wolf shrinkage: rho={lw_rho:.4f}")
        else:
            C = load_covariance_v2(QMC_SIM_DIR, r1=0, r2=0)
    elif DATA_SOURCE == "holstein_jld2":
        print(f"Loading site-Holstein cube from: {HOLSTEIN_JLD2_PATH}")
        print(f"  cell: beta={HOLSTEIN_BETA}, Omega={HOLSTEIN_OMEGA}, n={HOLSTEIN_N}")
        ft_dataset = HolsteinJLD2Dataset(
            HOLSTEIN_JLD2_PATH,
            n_idx=N_IDX, omega_idx=OMEGA_IDX, beta_idx=BETA_IDX,
        )
        ft_dataset.summary()
        if COVARIANCE_ESTIMATOR == "ledoit_wolf":
            G_r_full, _, _ = _load_holstein_jld2(HOLSTEIN_JLD2_PATH)
            G_bins = G_r_full[N_IDX, OMEGA_IDX, BETA_IDX, :INPUT_DIM, :].T.copy()
            C, lw_rho = ledoit_wolf_shrinkage(G_bins)
            print(f"Ledoit-Wolf shrinkage: rho={lw_rho:.4f}")
        else:
            C = load_covariance_from_holstein_jld2(
                HOLSTEIN_JLD2_PATH,
                n_idx=N_IDX, omega_idx=OMEGA_IDX, beta_idx=BETA_IDX,
            )
    else:
        print(f"Loading synthetic dataset from: {SYNTHETIC_DATA_PATH}")
        ft_dataset = GreenFunctionDataset(file_path=SYNTHETIC_DATA_PATH)
        if COVARIANCE_ESTIMATOR == "ledoit_wolf":
            raise NotImplementedError(
                "Ledoit-Wolf is not wired for synthetic input here; same "
                "limitation as run_finetune.py."
            )
        C = load_covariance_from_dqmc(SYNTHETIC_DATA_PATH)

    N_samples = len(ft_dataset)

    _g = None
    if SEED is not None:
        _g = torch.Generator()
        _g.manual_seed(SEED)
    train_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, generator=_g)
    val_loader = train_loader
    print(f"  Full dataset: {N_samples} samples, batch_size={BATCH_SIZE}, "
          f"{len(train_loader)} batches/epoch")

    # --- Save run parameters ---
    params = {
        "sID": sID,
        "MODEL_VERSION": MODEL_VERSION,
        "ARCHITECTURE": "DecoderOnly",
        "NUM_POLES": NUM_POLES,
        "INPUT_DIM": INPUT_DIM,
        "N_NODES": N_NODES,
        "PH_SYMMETRIC": PH_SYMMETRIC,
        "NUM_POLES_EFFECTIVE": 2 * NUM_POLES if PH_SYMMETRIC else NUM_POLES,
        "BETA": BETA,
        "DTAU": DTAU,
        "BATCH_SIZE": BATCH_SIZE,
        "N_TRAINABLE_PARAMS": n_params,
        "DATA_SOURCE": DATA_SOURCE,
        **({
            "SPECTRAL_TYPE": SPECTRAL_TYPE,
            "NOISE_S": NOISE_S,
            "NOISE_XI": NOISE_XI,
            "INPUT_ID": INPUT_ID,
        } if DATA_SOURCE == "synthetic" else {}),
        **({
            "HOLSTEIN_BETA":  HOLSTEIN_BETA,
            "HOLSTEIN_OMEGA": HOLSTEIN_OMEGA,
            "HOLSTEIN_N":     HOLSTEIN_N,
            "HOLSTEIN_BETA_IDX":  BETA_IDX,
            "HOLSTEIN_OMEGA_IDX": OMEGA_IDX,
            "HOLSTEIN_N_IDX":     N_IDX,
        } if DATA_SOURCE == "holstein_jld2" else {}),
        "DATA_PATH": (
            QMC_SIM_DIR        if DATA_SOURCE == "real"          else
            HOLSTEIN_JLD2_PATH if DATA_SOURCE == "holstein_jld2" else
            SYNTHETIC_DATA_PATH
        ),
        "FINETUNE_EPOCHS": FINETUNE_EPOCHS,
        "FINETUNE_LR": FINETUNE_LR,
        "FINETUNE_PATIENCE": FINETUNE_PATIENCE,
        "FINETUNE_KL_ANNEAL_EPOCHS": FINETUNE_KL_ANNEAL_EPOCHS,
        "LAMBDA_CHI2": LAMBDA_CHI2,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "LAMBDA_POS": LAMBDA_POS,
        "ALPHA_KL": ALPHA_KL,
        "ETA0": ETA0,
        "ETA2": ETA2,
        "ETA4": ETA4,
        "USE_SCHEDULER": USE_SCHEDULER,
        "LR_FACTOR": LR_FACTOR,
        "LR_PATIENCE": LR_PATIENCE,
        "LR_MIN": LR_MIN,
        "SMOOTHNESS_NW": SMOOTHNESS_NW,
        "SMOOTHNESS_WMIN": SMOOTHNESS_WMIN,
        "SMOOTHNESS_WMAX": SMOOTHNESS_WMAX,
        "SEED": SEED,
        "COVARIANCE_ESTIMATOR": COVARIANCE_ESTIMATOR,
        "VARIANCE_THRESHOLD":   VARIANCE_THRESHOLD,
        "LW_RHO": lw_rho,
        "LOSS_MODE": LOSS_MODE,
        "FLOOR_TARGET": FLOOR_TARGET,
        "FLOOR_DELTA": FLOOR_DELTA,
        "FLOOR_WARMUP_THRESH": FLOOR_WARMUP_THRESH,
        "BARRIER_LAMBDA": BARRIER_LAMBDA,
    }
    with open(f"{ft_out_dir}/params.json", "w") as _f:
        json.dump(params, _f, indent=2)
    print(f"Run parameters saved to {ft_out_dir}/params.json")

    # --- Loss modules ---
    print(f"Covariance estimator: {COVARIANCE_ESTIMATOR}    Loss mode: {LOSS_MODE}")
    kl_fn         = KLDivergenceLoss().to(DEVICE)
    _whitening_mode = "full" if COVARIANCE_ESTIMATOR == "ledoit_wolf" else "pca_truncated"
    chi2_fn       = ChiSquaredLoss(
        C, covariance_estimator=_whitening_mode, variance_threshold=VARIANCE_THRESHOLD,
    ).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX,
    ).to(DEVICE)
    positivity_fn = SpectralPositivityLoss().to(DEVICE)
    neg_green_fn  = NegativeGreenPenalty(C).to(DEVICE)
    neg_second_fn = NegativeSecondDerivativePenalty(C).to(DEVICE)
    neg_fourth_fn = NegativeFourthDerivativePenalty(C).to(DEVICE)

    if LOSS_MODE == "warmup_floored":
        chi2_transform = Chi2FloorTransform(
            target=FLOOR_TARGET, delta=FLOOR_DELTA, warmup_threshold=FLOOR_WARMUP_THRESH,
        ).to(DEVICE)
    elif LOSS_MODE == "barrier":
        chi2_transform = Chi2OneSidedBarrier(
            lambda_=BARRIER_LAMBDA, target=FLOOR_TARGET,
        ).to(DEVICE)
    elif LOSS_MODE == "raw":
        chi2_transform = None
    else:
        raise ValueError(f"Unknown LOSS_MODE={LOSS_MODE!r}")

    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN,
    ) if USE_SCHEDULER else None

    print(f"\nLoss weights: chi2={LAMBDA_CHI2}, smooth={LAMBDA_SMOOTH}, "
          f"pos={LAMBDA_POS}, kl={ALPHA_KL} (model is encoder-less; KL is 0), "
          f"eta0={ETA0}, eta2={ETA2}, eta4={ETA4}")

    best_val_loss, best_epoch = train_finetune(
        model=model, optimizer=optimizer, scheduler=scheduler,
        num_epochs=FINETUNE_EPOCHS, input_dim=INPUT_DIM,
        train_loader=train_loader, val_loader=val_loader,
        kl_fn=kl_fn, chi2_fn=chi2_fn, smoothness_fn=smoothness_fn,
        neg_green_fn=neg_green_fn, neg_second_fn=neg_second_fn,
        neg_fourth_fn=neg_fourth_fn, positivity_fn=positivity_fn,
        lambda_chi2=LAMBDA_CHI2, lambda_smooth=LAMBDA_SMOOTH,
        lambda_pos=LAMBDA_POS, alpha_kl=ALPHA_KL,
        eta0=ETA0, eta2=ETA2, eta4=ETA4,
        device=DEVICE, out_dir=ft_out_dir, tag="finetune",
        patience=FINETUNE_PATIENCE,
        kl_anneal_epochs=FINETUNE_KL_ANNEAL_EPOCHS,
        chi2_transform=chi2_transform,
    )

    chi2_losses   = np.load(f"{ft_out_dir}/losses/chi2_losses_finetune.npy")
    best_chi2     = float(np.min(chi2_losses))
    best_chi2_ep  = int(np.argmin(chi2_losses)) + 1
    final_chi2    = float(chi2_losses[-1])
    n_epochs_run  = len(chi2_losses)

    print(f"\nDone. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")
    print(f"chi2 summary  |  best: {best_chi2:.4f} @ epoch {best_chi2_ep}/{n_epochs_run}"
          f"  |  final: {final_chi2:.4f}")

    model.load_state_dict(torch.load(
        f"{ft_out_dir}/model/best_model_finetune.pth", weights_only=True,
    ))

    # --- Evaluation pass (deterministic; encoder-less so MC is unnecessary) ---
    print("\n--- Full-dataset evaluation ---")
    model.eval()
    eval_loss = 0.0
    G_input_all, G_recon_all = [], []
    poles_all, residues_all = [], []

    with torch.no_grad():
        for batch in train_loader:
            B = batch.shape[0]
            batch = batch.view(B, INPUT_DIM).to(DEVICE)
            mu, logvar, z, poles, residues, G_recon = model(batch, deterministic=True)
            loss, *_ = finetune_total_loss(
                G_recon, batch, mu, logvar, poles, residues,
                kl_fn, chi2_fn, smoothness_fn,
                neg_green_fn, neg_second_fn, neg_fourth_fn, positivity_fn,
                LAMBDA_CHI2, LAMBDA_SMOOTH, LAMBDA_POS, ALPHA_KL,
                ETA0, ETA2, ETA4, chi2_transform=chi2_transform,
            )
            eval_loss += loss.item() * B
            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())

    eval_loss /= len(ft_dataset)
    G_input_all  = torch.cat(G_input_all,  dim=0)
    G_recon_all  = torch.cat(G_recon_all,  dim=0)
    poles_all    = torch.cat(poles_all,    dim=0)
    residues_all = torch.cat(residues_all, dim=0)
    print(f"Final Loss: {eval_loss:.4e}")

    # Forward pass on the dataset-averaged G(τ) — for the encoder-less model
    # the per-sample input is ignored, so this is mechanically the same as
    # any single-sample forward, but we keep the entry in summary.pt for
    # downstream-tool compatibility (compare_vae_maxent.py reads it).
    G_avg_input = G_input_all.mean(dim=0, keepdim=True).to(DEVICE)
    with torch.no_grad():
        _, _, _, poles_avg_in, residues_avg_in, G_recon_avg_in = model(
            G_avg_input, deterministic=True,
        )
    poles_from_avg    = poles_avg_in.squeeze(0).cpu()
    residues_from_avg = residues_avg_in.squeeze(0).cpu()
    recon_from_avg    = G_recon_avg_in.squeeze(0).cpu()

    # Self-consistency — flat by construction since per-sample A_i are
    # identical. Reported anyway for cross-run table compatibility.
    SC_NW   = 1000
    SC_WMIN, SC_WMAX = -20.0, 20.0
    omega_sc = torch.linspace(SC_WMIN, SC_WMAX, SC_NW)
    with torch.no_grad():
        A_samples_sc  = spectral_from_poles(poles_all, residues_all, omega_sc)
        A_from_avg_sc = spectral_from_poles(
            poles_from_avg.unsqueeze(0), residues_from_avg.unsqueeze(0), omega_sc,
        ).squeeze(0)
    diff_sc = (A_samples_sc - A_from_avg_sc.unsqueeze(0)).numpy()
    sc_per_sample = np.trapezoid(diff_sc ** 2, omega_sc.numpy(), axis=1)
    sc_mean = float(sc_per_sample.mean())
    sc_std  = float(sc_per_sample.std())
    sc_min, sc_max = float(sc_per_sample.min()), float(sc_per_sample.max())
    sc_linf_per_sample = np.max(np.abs(diff_sc), axis=1)
    sc_linf_mean = float(sc_linf_per_sample.mean())
    sc_linf_std  = float(sc_linf_per_sample.std())
    sc_linf_min, sc_linf_max = float(sc_linf_per_sample.min()), float(sc_linf_per_sample.max())

    if DATA_SOURCE == "holstein_jld2":
        _sim_name = (
            f"site_holstein_b{HOLSTEIN_BETA:.2f}_w{HOLSTEIN_OMEGA:.2f}_n{HOLSTEIN_N:.2f}"
        )
        _holstein_extra = {
            "dos_ref": torch.tensor(ft_dataset.dos, dtype=torch.float64),
            "ws_ref":  torch.tensor(ft_dataset.ws,  dtype=torch.float64),
            "holstein_beta":  HOLSTEIN_BETA,
            "holstein_omega": HOLSTEIN_OMEGA,
            "holstein_n":     HOLSTEIN_N,
        }
    else:
        _sim_name = os.path.basename(QMC_SIM_DIR) if DATA_SOURCE == "real" else None
        _holstein_extra = {}

    torch.save({
        "inputs":       G_input_all,
        "recon":        G_recon_all,
        "poles":        poles_all,
        "residues":     residues_all,
        "inputs_avg":   G_input_all.mean(dim=0),
        "recon_avg":    G_recon_all.mean(dim=0),
        "poles_avg":    poles_all.mean(dim=0),
        "residues_avg": residues_all.mean(dim=0),
        "poles_from_avg":    poles_from_avg,
        "residues_from_avg": residues_from_avg,
        "recon_from_avg":    recon_from_avg,
        "beta":               BETA,
        "Ltau":               INPUT_DIM,
        "spectral_input_path": SPECTRAL_INPUT_PATH if DATA_SOURCE == "synthetic" else None,
        "noise_var":    float(np.mean(np.diag(np.array(C)))),
        "data_source":        DATA_SOURCE,
        "sim_name":           _sim_name,
        "num_poles":          NUM_POLES,
        "sID":                sID,
        "architecture":       "DecoderOnly",
        "best_chi2":          best_chi2,
        "best_chi2_epoch":    best_chi2_ep,
        "final_chi2":         final_chi2,
        "n_epochs":           n_epochs_run,
        "self_consistency":                  sc_mean,
        "self_consistency_std":              sc_std,
        "self_consistency_per_sample":       sc_per_sample,
        "self_consistency_linfty":           sc_linf_mean,
        "self_consistency_linfty_std":       sc_linf_std,
        "self_consistency_linfty_per_sample": sc_linf_per_sample,
        "self_consistency_grid":             {"wmin": SC_WMIN, "wmax": SC_WMAX, "Nw": SC_NW},
        **_holstein_extra,
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved → {ft_out_dir}/summary.pt")
    print(f"SC-L2     = {sc_mean:.6f}  (std {sc_std:.6f}, range [{sc_min:.6f}, {sc_max:.6f}])")
    print(f"SC-Linfty = {sc_linf_mean:.6f}  (std {sc_linf_std:.6f}, "
          f"range [{sc_linf_min:.6f}, {sc_linf_max:.6f}])")

    # --- Run summary text ---
    smooth_f = float(np.load(f"{ft_out_dir}/losses/smooth_losses_finetune.npy")[-1])
    pos_f    = float(np.load(f"{ft_out_dir}/losses/pos_losses_finetune.npy")[-1])
    ng_f     = float(np.load(f"{ft_out_dir}/losses/neg_green_losses_finetune.npy")[-1])
    ng2_f    = float(np.load(f"{ft_out_dir}/losses/neg_second_losses_finetune.npy")[-1])
    ng4_f    = float(np.load(f"{ft_out_dir}/losses/neg_fourth_losses_finetune.npy")[-1])

    poles_np = poles_all.mean(dim=0).numpy()
    res_np   = residues_all.mean(dim=0).numpy()
    sim_label = (
        os.path.basename(QMC_SIM_DIR) if DATA_SOURCE == "real" else
        f"site_holstein_b{HOLSTEIN_BETA:.2f}_w{HOLSTEIN_OMEGA:.2f}_n{HOLSTEIN_N:.2f}"
        if DATA_SOURCE == "holstein_jld2" else
        os.path.basename(os.path.dirname(SPECTRAL_INPUT_PATH))
    )
    early_stop = n_epochs_run < FINETUNE_EPOCHS

    W = 78
    def _border(l, r, fill="-"): return l + fill * W + r
    def _line(c):                return f"|{c[:W].ljust(W)}|"
    def _row(label, val="", indent=2, lw=18):
        return _line(" " * indent + f"{label:<{lw}}{val}")
    def _sec(t):                 return f"+{(' -- ' + t + ' '):-<{W}}+"
    def _wrap(value, indent=4):
        room = W - indent
        out = []
        for i in range(0, len(value), room):
            out.append(_line(" " * indent + value[i:i + room]))
        return out or [_line(" " * indent)]
    def _fmt_complex(v, sign="+"):
        s = f"{v.real:{sign}.4f}{v.imag:+.4f}j"
        return f"{s:<16}"
    def _grid(values, sign="+", indent=4, sep="  "):
        item_w  = 16 + len(sep)
        per_row = max(1, (W - indent) // item_w)
        out = []
        for i in range(0, len(values), per_row):
            chunk = values[i:i + per_row]
            text  = sep.join(_fmt_complex(v, sign=sign) for v in chunk).rstrip()
            out.append(_line(" " * indent + text))
        return out

    lines = []
    lines.append(_border("+", "+", "="))
    lines.append(_line(f"{'RUN SUMMARY (DecoderOnly)':^{W}}"))
    lines.append(_border("+", "+", "="))
    lines.append(_row("tag",              "finetune"))
    lines.append(_row("sID",              str(sID)))
    lines.append(_row("data",             f"{DATA_SOURCE}   {sim_label}"))
    lines.append(_row("beta/Ltau/poles",  f"{BETA} / {INPUT_DIM} / {NUM_POLES}   "
                                          f"model = {MODEL_VERSION} (encoder-less)"))
    lines.append(_row("trainable params", f"{n_params:,}"))
    lines.append(_sec("TRAINING"))
    lines.append(_row("epochs",     f"{n_epochs_run} / {FINETUNE_EPOCHS}"
                                    + ("  [early stop]" if early_stop else "")))
    lines.append(_row("best epoch", f"{best_epoch+1}   val loss = {best_val_loss:.4e}"))
    lines.append(_row("eval loss",  f"{eval_loss:.4e}"))
    lines.append(_sec("chi^2"))
    lines.append(_row("final",  f"{final_chi2:.6f}"))
    lines.append(_row("best",   f"{best_chi2:.6f}   @ epoch {best_chi2_ep} / {n_epochs_run}"))
    lines.append(_sec(f"SELF-CONSISTENCY  (omega in [{SC_WMIN:g},{SC_WMAX:g}], Nw={SC_NW})"))
    lines.append(_row("SC-L2 mean",      f"{sc_mean:.6f}"))
    lines.append(_row("SC-L2 std",       f"{sc_std:.6f}"))
    lines.append(_row("SC-L2 range",     f"[{sc_min:.6f}, {sc_max:.6f}]   N={len(sc_per_sample)}"))
    lines.append(_row("SC-Linf mean",    f"{sc_linf_mean:.6f}"))
    lines.append(_row("SC-Linf std",     f"{sc_linf_std:.6f}"))
    lines.append(_row("SC-Linf range",   f"[{sc_linf_min:.6f}, {sc_linf_max:.6f}]"))
    lines.append(_sec("ENCODER STATUS"))
    lines.append(_row("architecture",    "DecoderOnly  (no encoder, no latent space)"))
    lines.append(_row("KL term",         "0  by construction (mu=logvar=0)"))
    lines.append(_sec("LOSS COMPONENTS  (final epoch, unweighted)"))
    lines.append(_row("chi^2",      f"{chi2_losses[-1]:.4e}",                   indent=4, lw=14))
    lines.append(_row("smooth",     f"{smooth_f:.4e}    pos        {pos_f:.4e}", indent=4, lw=14))
    lines.append(_row("KL",         f"0.0000e+00    neg G      {ng_f:.4e}",     indent=4, lw=14))
    lines.append(_row("neg G''",    f"{ng2_f:.4e}    neg G''''  {ng4_f:.4e}",   indent=4, lw=14))
    lines.append(_sec("POLES  (dataset mean)"))
    lines.extend(_grid(poles_np, sign="+"))
    lines.append(_sec("RESIDUES  (dataset mean)"))
    lines.extend(_grid(res_np, sign=""))
    lines.append(_border("+", "+", "="))
    lines.append(_line("  output"))
    lines.extend(_wrap(ft_out_dir, indent=4))
    lines.append(_border("+", "+", "="))

    summary_path = f"{ft_out_dir}/run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Run summary → {summary_path}")

    if DO_PLOT:
        print("\n--- Generating fine-tuning plots ---")
        plot_finetune_eval(f"{ft_out_dir}/summary.pt")
        plot_loss_curves(f"{ft_out_dir}/losses", tag="finetune",
                         save_path=f"{ft_out_dir}/plots/loss_curves_finetune.pdf",
                         params=params)


if __name__ == "__main__":
    main()
