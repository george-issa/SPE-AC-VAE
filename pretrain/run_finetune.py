"""
Standalone Stage 2: fine-tune a VAE on DQMC Green's function data.

Trains unsupervised on the full target dataset (all samples, no held-out split).
The scheduler (ReduceLROnPlateau) monitors the eval-mode training loss.

Loss:
  L = lambda_chi2  * chi2              (covariance-whitened reconstruction)
    + lambda_smooth * smoothness       (second-derivative regularizer on A(omega))
    + lambda_pos    * positivity       (penalizes negative A(omega))
    + alpha_kl      * KL              (latent prior regularization)
    + eta0          * neg_green       (G(tau) >= 0, variance-weighted by diag(C))
    + eta2          * neg_second      (G''(tau) >= 0, variance-weighted by diag(C))
    + eta4          * neg_fourth      (G''''(tau) >= 0, variance-weighted by diag(C))

Usage:
    python pretrain/run_finetune.py
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# MODEL_VERSION selects the architecture to train:
#   2   — model2.py: original (LeakyReLU/ReLU, no InstanceNorm)
#   "2t" — model2_tanh.py: Step-2 variant (InstanceNorm1d + tanh, same decoder FC structure)

MODEL_VERSION = "2L"

if MODEL_VERSION == 2:
    from model2_copy import VariationalAutoEncoder2 as VAEModel  # type: ignore
elif MODEL_VERSION == "2t":
    from model2_tanh import VariationalAutoEncoder2 as VAEModel  # type: ignore
elif MODEL_VERSION == "2s":
    from model2_silu import VariationalAutoEncoder2 as VAEModel # type: ignore
elif MODEL_VERSION == "2L":
    from model2_leaky import VariationalAutoEncoder2 as VAEModel # type: ignore
elif MODEL_VERSION == "2sc":
    from model2_silu_sym import VariationalAutoEncoder2 as VAEModel # type: ignore
else:
    raise ValueError(f"Unknown MODEL_VERSION={MODEL_VERSION!r}. Use 2, '2t', '2s")
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

# --- Data source: "synthetic", "real", or "holstein_jld2" ---
DATA_SOURCE = "holstein_jld2"   # synthetic CSV, SmoQyDQMC v2 dir, or site-Holstein cube

# --- Model ---
LOAD_PRETRAIN = False
NUM_POLES = 9
N_NODES = 256
BATCH_SIZE = 5

# Latent bottleneck dim. None -> default (= 4*NUM_POLES - 2, legacy behaviour).
# Set to an int to force a smaller latent (decoupled from num_poles); useful for
# studying intrinsic-capacity / reproducibility experiments.
LATENT_DIM = 1

# Particle-hole symmetry on the decoder. When True, NUM_POLES denotes the
# *free* pole count; the model concatenates each free pole with its PH partner
# (-eps, gamma, a, -b), so the spectral function is even (A(w)=A(-w)) and the
# effective pole count is 2*NUM_POLES. The free count is what the network has
# to predict; downstream wall-clock and parameter count track NUM_POLES.
PH_SYMMETRIC = False  # site-Holstein cell A(w) is not even — keep off for first run

# Allow pretrain/run_finetune_sweep.py to override these without editing the file.
NUM_POLES = int(os.environ.get("SWEEP_NUM_POLES", NUM_POLES))
if os.environ.get("SWEEP_LATENT_DIM"):
    LATENT_DIM = int(os.environ["SWEEP_LATENT_DIM"])
if os.environ.get("SWEEP_PH_SYMMETRIC"):
    PH_SYMMETRIC = os.environ["SWEEP_PH_SYMMETRIC"].strip().lower() in ("1", "true", "yes")

# --- chi^2 floor diagnostic (N1) ---
# Covariance estimator for the whitening matrix in ChiSquaredLoss:
#   "pca_truncated" (default) — drop low-eigenvalue directions at variance_threshold.
#   "ledoit_wolf"             — shrinkage toward scaled identity; uses raw bins.
# Loss mode controls whether chi^2 enters the gradient as-is or gets wrapped:
#   "raw"             (default)        — pass-through.
#   "warmup_floored"  — symmetric pseudo-Huber centred at 1 with warmup. Note:
#       in practice this collapses the latent (chi^2 gradient flat near target).
#   "barrier"         — one-sided ReLU^2 barrier below 1; raw chi^2 above 1.
#       Equilibrium at chi^2* = 1 - 1/(2*BARRIER_LAMBDA).
COVARIANCE_ESTIMATOR = "ledoit_wolf"
# Cumulative-variance fraction kept by ChiSquaredLoss when
# COVARIANCE_ESTIMATOR == "pca_truncated". Higher → more components retained.
# On site-Holstein (b=10, w=1, n=1.0): 0.99 → 4/100 components, 0.999 → 7/100,
# 0.9999 → 13/100. Ignored when COVARIANCE_ESTIMATOR == "ledoit_wolf".
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
# *** Only this line should change when uploading a new simulation folder ***
# --------------------------------------------------------------------------
QMC_SIM_DIR = os.path.join(
    MAIN_PATH, "Data", "datasets", "real",
    "hubbard_square_U8.00_mu0.00_L4_b6.00-1"
)

# --------------------------------------------------------------------------
# Site-Holstein cube (DATA_SOURCE = "holstein_jld2")
# Single .jld2 stores G(tau) and reference DOS for the (n, Omega, beta) grid.
# --------------------------------------------------------------------------
HOLSTEIN_JLD2_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "real", "george_325.jld2"
)
HOLSTEIN_BETA  = 10.0   # in {5,6,...,20}
HOLSTEIN_OMEGA = 1.0    # in {0.5, 1.0, 1.5, 2.0}
HOLSTEIN_N     = 1.0    # in {0.05, 0.10, ..., 1.00}; n=1.0 is half-filling

def _holstein_idx(value, grid, label):
    arr = np.asarray(grid, dtype=float)
    i   = int(np.argmin(np.abs(arr - float(value))))
    if abs(arr[i] - float(value)) > 1e-6:
        raise ValueError(f"{label}={value} not in grid {grid}")
    return i

# --------------------------------------------------------------------------
# Shared physical parameters
# --------------------------------------------------------------------------
if DATA_SOURCE == "real":
    _qmc_params = read_model_params(QMC_SIM_DIR)
    BETA      = _qmc_params["beta"]
    DTAU      = _qmc_params["dtau"]
    INPUT_DIM = _qmc_params["L_tau"]   # authoritative — no floating-point precision issue
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

# --- Pretrained checkpoint ---
PRETRAIN_DIR = os.path.join(
    MAIN_PATH, "out",
    f"pretrain_synthetic_numpoles{NUM_POLES}"
)
PRETRAIN_CHECKPOINT = os.path.join(PRETRAIN_DIR, "model", "best_model_pretrain.pth")

# --- Fine-tuning hyperparameters ---
FINETUNE_EPOCHS = 3000
FINETUNE_EPOCHS = int(os.environ.get("SWEEP_FINETUNE_EPOCHS", FINETUNE_EPOCHS))
FINETUNE_LR = 1e-3
FINETUNE_PATIENCE = FINETUNE_EPOCHS + 1              # Early stopping enabled with a wide window
FINETUNE_KL_ANNEAL_EPOCHS = 0
_model_tag = {2: "v2", "2t": "v2t", "2s": "v2s"}.get(MODEL_VERSION, f"v{MODEL_VERSION}")

# Loss weights
LAMBDA_CHI2 = 1.0       # Covariance-weighted reconstruction (chi^2 → 1 at perfect fit)
LAMBDA_SMOOTH = 0.0     # Smoothness loss disabled; fights narrow Gaussian peaks
LAMBDA_POS = 0.1        # Positivity loss 
ALPHA_KL = 1e-6         # Very mild KL regularization (matches Ben)
ETA0 = 1.0              # G(tau) >= 0 penalty — variance-weighted via diag(C)
ETA2 = 1.0              # G''(tau) >= 0 penalty — variance-weighted via diag(C)
ETA4 = 0.0              # G''''(tau) >= 0 penalty — keep OFF (causes divergence at 1.0)

# ReduceLROnPlateau on val loss. Val is deterministic (z = mu) so this is a
# clean plateau signal; the scheduler drops LR once the basin tightens,
# preventing late-training blowups driven by a constant LR of 1e-3.
USE_SCHEDULER = False
LR_FACTOR = 0.5         # Multiply LR by this on plateau
LR_PATIENCE = 6        # Epochs to wait before reducing LR
LR_MIN = 1e-6           # Floor for LR

# Spectral evaluation grid
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -20.0
SMOOTHNESS_WMAX = 20.0

# --- VAE diagnostics ---
# Active-units (Burda et al. 2016): count latent dims with Var_x[mu_i(x)] > AU_THRESHOLD.
# Cheap to compute (one extra batch-loop tensor); use as a sanity check that the
# encoder is actually using its latent capacity. With ALPHA_KL ~ 0 you expect
# active_units ~ latent_dim; collapse should not be an issue at that KL pressure.
COMPUTE_ACTIVE_UNITS = True
AU_THRESHOLD = 1e-2

# Reproducibility — set to an integer (e.g. 42) for deterministic runs, or None to disable
SEED = None

# Set False on the cluster — all plotting stays on the local machine
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
    print(f"Model version: {MODEL_VERSION}")
    print(f"Model: NUM_POLES={NUM_POLES}, INPUT_DIM={INPUT_DIM}")

    # ------------------------------------------------------------------
    # Initialize model and load pretrained weights
    # ------------------------------------------------------------------
    model = VAEModel(
        input_dim=INPUT_DIM,
        num_poles=NUM_POLES,
        beta=BETA,
        N_nodes=N_NODES,
        latent_dim=LATENT_DIM,
        ph_symmetric=PH_SYMMETRIC,
    ).to(DEVICE)

    if LOAD_PRETRAIN and os.path.exists(PRETRAIN_CHECKPOINT):
        print(f"\nLoading pretrained model from: {PRETRAIN_CHECKPOINT}")
        model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT, map_location=DEVICE, weights_only=True))
        print("Pretrained weights loaded successfully.")
    else:
        print(f"\nWARNING: Pretrained checkpoint at {PRETRAIN_CHECKPOINT} not found or not being used.")
        print("Starting from scratch (random initialization).\n")

    # ------------------------------------------------------------------
    # STAGE 2: Fine-tuning on target dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STAGE 2: FINE-TUNING ON {DATA_SOURCE.upper()} DATA")
    print("=" * 80 + "\n")

    tag = "finetune"

    # --- Output directory ---
    if DATA_SOURCE == "real":
        _out_label = f"real-{os.path.basename(QMC_SIM_DIR)}"
    elif DATA_SOURCE == "holstein_jld2":
        _out_label = (
            f"real-site_holstein_b{HOLSTEIN_BETA:.2f}_w{HOLSTEIN_OMEGA:.2f}_n{HOLSTEIN_N:.2f}"
        )
    else:
        _out_label = f"synthetic-{SPECTRAL_TYPE}_s{NOISE_S:.0e}_xi{NOISE_XI}"
    _init_tag  = "pretrained" if LOAD_PRETRAIN else "fresh"
    # Latent-bottleneck override gets a `_z{N}` suffix so runs are kept distinct
    # from default-latent_dim runs at the same num_poles. PH-symmetric runs get
    # a `_ph` suffix for the same reason — same NUM_POLES, different decoder.
    _z_tag     = f"_z{LATENT_DIM}" if LATENT_DIM is not None else ""
    _ph_tag    = "_ph" if PH_SYMMETRIC else ""
    _cov_tag   = "_covlw" if COVARIANCE_ESTIMATOR == "ledoit_wolf" else ""
    _loss_tag  = (
        "_lossfloor"   if LOSS_MODE == "warmup_floored" else
        "_lossbarrier" if LOSS_MODE == "barrier"        else
        ""
    )
    _base_name = f"finetune_{_out_label}_numpoles{NUM_POLES}{_ph_tag}{_z_tag}{_cov_tag}{_loss_tag}-{_init_tag}-{_model_tag}"
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
    sID        = f"{_init_tag}-{_model_tag}-{_next_id}"
    ft_out_dir = os.path.join(_out_root,
                              f"finetune_{_out_label}_numpoles{NUM_POLES}{_ph_tag}{_z_tag}{_cov_tag}{_loss_tag}-{sID}")
    print(f"Output directory: {ft_out_dir}")
    MakeOutPath(ft_out_dir)

    # --- Load full dataset — train on all samples, no held-out split ---
    lw_rho = None  # captured into params.json when COVARIANCE_ESTIMATOR=="ledoit_wolf"
    if DATA_SOURCE == "real":
        print(f"Loading real QMC dataset from: {QMC_SIM_DIR}")
        ft_dataset = SmoQyV2Dataset(QMC_SIM_DIR, r1=0, r2=0)
        ft_dataset.summary()
        if COVARIANCE_ESTIMATOR == "ledoit_wolf":
            G_bins, _ = extract_greens_bins_v2(QMC_SIM_DIR, r1=0, r2=0)
            C, lw_rho = ledoit_wolf_shrinkage(G_bins)
            print(f"Ledoit-Wolf shrinkage: rho={lw_rho:.4f} (N_bins={G_bins.shape[0]}, "
                  f"L_tau={G_bins.shape[1]})")
        else:
            C = load_covariance_v2(QMC_SIM_DIR, r1=0, r2=0)
    elif DATA_SOURCE == "holstein_jld2":
        print(f"Loading site-Holstein cube from: {HOLSTEIN_JLD2_PATH}")
        print(f"  cell: beta={HOLSTEIN_BETA}, Omega={HOLSTEIN_OMEGA}, n={HOLSTEIN_N}  "
              f"(idx beta={BETA_IDX}, Omega={OMEGA_IDX}, n={N_IDX})")
        ft_dataset = HolsteinJLD2Dataset(
            HOLSTEIN_JLD2_PATH,
            n_idx=N_IDX, omega_idx=OMEGA_IDX, beta_idx=BETA_IDX,
        )
        ft_dataset.summary()
        if COVARIANCE_ESTIMATOR == "ledoit_wolf":
            G_r_full, _, _ = _load_holstein_jld2(HOLSTEIN_JLD2_PATH)
            G_bins = G_r_full[N_IDX, OMEGA_IDX, BETA_IDX, :INPUT_DIM, :].T.copy()
            C, lw_rho = ledoit_wolf_shrinkage(G_bins)
            print(f"Ledoit-Wolf shrinkage: rho={lw_rho:.4f} (N_bins={G_bins.shape[0]}, "
                  f"L_tau={G_bins.shape[1]})")
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
                "Ledoit-Wolf covariance estimator is currently only wired for "
                "DATA_SOURCE='real'. The synthetic loader returns C directly, "
                "not raw bins."
            )
        C = load_covariance_from_dqmc(SYNTHETIC_DATA_PATH)

    N_samples = len(ft_dataset)

    _g = None
    if SEED is not None:
        _g = torch.Generator()
        _g.manual_seed(SEED)
    train_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=_g)
    val_loader = train_loader  # Same data; ReduceLROnPlateau monitors eval-mode training loss
    print(f"  Full dataset: {N_samples} samples, batch_size={BATCH_SIZE}, {len(train_loader)} batches/epoch")

    # --- Save run parameters for reference ---
    params = {
        "sID": sID,
        "MODEL_VERSION": str(MODEL_VERSION),
        "NUM_POLES": NUM_POLES,
        "INPUT_DIM": INPUT_DIM,
        "N_NODES": N_NODES,
        "LATENT_DIM": LATENT_DIM,
        "PH_SYMMETRIC": PH_SYMMETRIC,
        "NUM_POLES_EFFECTIVE": 2 * NUM_POLES if PH_SYMMETRIC else NUM_POLES,
        "BETA": BETA,
        "DTAU": DTAU,
        "BATCH_SIZE": BATCH_SIZE,
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
        "LOAD_PRETRAIN": LOAD_PRETRAIN,
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

    # --- Load covariance and build loss modules ---
    print(f"Covariance shape: {C.shape}")
    print(f"Covariance estimator: {COVARIANCE_ESTIMATOR}")
    print(f"Loss mode: {LOSS_MODE}")

    kl_fn         = KLDivergenceLoss().to(DEVICE)
    # Translate the user-facing estimator name to the whitening mode that
    # ChiSquaredLoss expects. LW already shrinks C upstream, so the whitener
    # runs without truncation; PCA truncation is the legacy/default path.
    _whitening_mode = "full" if COVARIANCE_ESTIMATOR == "ledoit_wolf" else "pca_truncated"
    chi2_fn       = ChiSquaredLoss(
        C,
        covariance_estimator=_whitening_mode,
        variance_threshold=VARIANCE_THRESHOLD,
    ).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX).to(DEVICE)
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
        print(f"  Chi2OneSidedBarrier: lambda={BARRIER_LAMBDA}, target={FLOOR_TARGET}, "
              f"equilibrium chi^2* = {FLOOR_TARGET - 1.0 / (2.0 * BARRIER_LAMBDA):.4f}")
    elif LOSS_MODE == "raw":
        chi2_transform = None
    else:
        raise ValueError(f"Unknown LOSS_MODE={LOSS_MODE!r}. "
                         f"Expected 'raw', 'warmup_floored', or 'barrier'.")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN
    ) if USE_SCHEDULER else None

    # --- Train ---
    print(f"\nLoss weights: lambda_chi2={LAMBDA_CHI2}, lambda_smooth={LAMBDA_SMOOTH}, "
          f"lambda_pos={LAMBDA_POS}, alpha_kl={ALPHA_KL}, eta0={ETA0}, eta2={ETA2}, eta4={ETA4}")
    print(f"Scheduler: ReduceLROnPlateau(factor={LR_FACTOR}, patience={LR_PATIENCE}, min_lr={LR_MIN})")
    print(f"Training on full dataset ({len(ft_dataset)} samples), early stopping disabled.")

    best_val_loss, best_epoch = train_finetune(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=FINETUNE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=train_loader,
        val_loader=val_loader,
        kl_fn=kl_fn,
        chi2_fn=chi2_fn,
        smoothness_fn=smoothness_fn,
        neg_green_fn=neg_green_fn,
        neg_second_fn=neg_second_fn,
        neg_fourth_fn=neg_fourth_fn,
        positivity_fn=positivity_fn,
        lambda_chi2=LAMBDA_CHI2,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_pos=LAMBDA_POS,
        alpha_kl=ALPHA_KL,
        eta0=ETA0,
        eta2=ETA2,
        eta4=ETA4,
        device=DEVICE,
        out_dir=ft_out_dir,
        tag=tag,
        patience=FINETUNE_PATIENCE,
        kl_anneal_epochs=FINETUNE_KL_ANNEAL_EPOCHS,
        chi2_transform=chi2_transform,
    )

    chi2_losses   = np.load(f"{ft_out_dir}/losses/chi2_losses_{tag}.npy")
    best_chi2     = float(np.min(chi2_losses))
    best_chi2_ep  = int(np.argmin(chi2_losses)) + 1
    final_chi2    = float(chi2_losses[-1])
    n_epochs_run  = len(chi2_losses)

    print(f"\nFine-tuning done. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")
    print(f"chi2 summary  |  best: {best_chi2:.4f} @ epoch {best_chi2_ep}/{n_epochs_run}"
          f"  |  final: {final_chi2:.4f}")

    # Load best model
    model.load_state_dict(torch.load(f"{ft_out_dir}/model/best_model_{tag}.pth", weights_only=True))

    # ------------------------------------------------------------------
    # Full-dataset evaluation
    # ------------------------------------------------------------------
    print("\n--- Full-dataset evaluation ---")
    model.eval()
    eval_loss = 0.0
    G_input_all, G_recon_all = [], []
    poles_all, residues_all = [], []
    mu_all, logvar_all = [], []   # only filled if COMPUTE_ACTIVE_UNITS

    with torch.no_grad():
        for batch in train_loader:
            B = batch.shape[0]
            batch = batch.view(B, INPUT_DIM).to(DEVICE)

            mu, logvar, z, poles, residues, G_recon = model(batch, deterministic=True)

            loss, _, _, _, _, _, _, _ = finetune_total_loss(
                G_recon, batch,
                mu, logvar,
                poles, residues,
                kl_fn, chi2_fn, smoothness_fn,
                neg_green_fn, neg_second_fn, neg_fourth_fn,
                positivity_fn,
                LAMBDA_CHI2, LAMBDA_SMOOTH, LAMBDA_POS, ALPHA_KL,
                ETA0, ETA2, ETA4,
                chi2_transform=chi2_transform,
            )
            eval_loss += loss.item() * B

            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())
            if COMPUTE_ACTIVE_UNITS:
                mu_all.append(mu.cpu())
                logvar_all.append(logvar.cpu())

    eval_loss /= len(ft_dataset)
    print(f"Final Loss: {eval_loss:.4e}")

    G_input_all   = torch.cat(G_input_all,   dim=0)
    G_recon_all   = torch.cat(G_recon_all,   dim=0)
    poles_all     = torch.cat(poles_all,     dim=0)
    residues_all  = torch.cat(residues_all,  dim=0)

    # Forward pass on the dataset-averaged G(τ): a single deterministic
    # prediction whose A(ω) summarises the dataset as one input.
    G_avg_input = G_input_all.mean(dim=0, keepdim=True).to(DEVICE)
    with torch.no_grad():
        mu_a, logvar_a, _, poles_avg_in, residues_avg_in, G_recon_avg_in = model(
            G_avg_input, deterministic=True
        )
    poles_from_avg    = poles_avg_in.squeeze(0).cpu()
    residues_from_avg = residues_avg_in.squeeze(0).cpu()
    recon_from_avg    = G_recon_avg_in.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # Self-consistency metric (model-selection signal, no ground truth needed)
    #   SC = (1/N) sum_i  int |A_i(w | G_i) - A(w | <G>)|^2 dw
    # Lower = more consistent across samples. Reported per-run so it can be
    # compared across e.g. NUM_POLES values without needing a MaxEnt run.
    # ------------------------------------------------------------------
    SC_NW   = 1000
    SC_WMIN = -20.0
    SC_WMAX = 20.0
    omega_sc = torch.linspace(SC_WMIN, SC_WMAX, SC_NW)
    with torch.no_grad():
        A_samples_sc  = spectral_from_poles(poles_all, residues_all, omega_sc)  # (N, Nw)
        A_from_avg_sc = spectral_from_poles(
            poles_from_avg.unsqueeze(0),
            residues_from_avg.unsqueeze(0),
            omega_sc,
        ).squeeze(0)                                                            # (Nw,)
    diff_sc    = (A_samples_sc - A_from_avg_sc.unsqueeze(0)).numpy()           # (N, Nw)
    diff_sq_sc = diff_sc ** 2
    # SC-L2: integrated squared deviation, averaged over samples
    sc_per_sample = np.trapezoid(diff_sq_sc, omega_sc.numpy(), axis=1)          # (N,)
    sc_mean = float(sc_per_sample.mean())
    sc_std  = float(sc_per_sample.std())
    sc_min  = float(sc_per_sample.min())
    sc_max  = float(sc_per_sample.max())
    # SC-Linfty: max pointwise deviation, averaged over samples
    sc_linf_per_sample = np.max(np.abs(diff_sc), axis=1)                        # (N,)
    sc_linf_mean = float(sc_linf_per_sample.mean())
    sc_linf_std  = float(sc_linf_per_sample.std())
    sc_linf_min  = float(sc_linf_per_sample.min())
    sc_linf_max  = float(sc_linf_per_sample.max())

    # ------------------------------------------------------------------
    # Active-units diagnostic (Burda et al. 2016): count latent dims where
    # Var_x[mu_i(x)] > AU_THRESHOLD. Also report per-dim KL contribution.
    # Toggleable via COMPUTE_ACTIVE_UNITS — set False to skip.
    # ------------------------------------------------------------------
    au_data = {}
    if COMPUTE_ACTIVE_UNITS:
        mu_cat     = torch.cat(mu_all,     dim=0).numpy()    # (N, D)
        logvar_cat = torch.cat(logvar_all, dim=0).numpy()    # (N, D)
        var_mu     = mu_cat.var(axis=0, ddof=0)              # (D,)
        sigma2     = np.exp(logvar_cat)
        # KL(q(z_i|x) || N(0,1)) per dim, averaged over samples
        kl_per_dim = 0.5 * (mu_cat ** 2 + sigma2 - 1.0 - logvar_cat).mean(axis=0)  # (D,)
        n_active   = int((var_mu > AU_THRESHOLD).sum())
        latent_dim = int(var_mu.shape[0])
        kl_total   = float(kl_per_dim.sum())
        au_data = {
            "active_units":   n_active,
            "latent_dim":     latent_dim,
            "au_threshold":   AU_THRESHOLD,
            "var_mu_per_dim": var_mu,
            "kl_per_dim":     kl_per_dim,
            "kl_total_avg":   kl_total,
        }
        print(f"Active units: {n_active}/{latent_dim}  "
              f"(threshold Var[mu]>{AU_THRESHOLD:g})  KL_total={kl_total:.4e}")

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
        **au_data,
        **_holstein_extra,
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved → {ft_out_dir}/summary.pt")
    print(f"SC-L2     = {sc_mean:.6f}  (std {sc_std:.6f}, "
          f"min {sc_min:.6f}, max {sc_max:.6f})  over {len(sc_per_sample)} samples")
    print(f"SC-Linfty = {sc_linf_mean:.6f}  (std {sc_linf_std:.6f}, "
          f"min {sc_linf_min:.6f}, max {sc_linf_max:.6f})")

    # ── End-of-run summary ────────────────────────────────────────────────────
    smooth_f = float(np.load(f"{ft_out_dir}/losses/smooth_losses_{tag}.npy")[-1])
    kl_f     = float(np.load(f"{ft_out_dir}/losses/kl_losses_{tag}.npy")[-1])
    pos_f    = float(np.load(f"{ft_out_dir}/losses/pos_losses_{tag}.npy")[-1])
    ng_f     = float(np.load(f"{ft_out_dir}/losses/neg_green_losses_{tag}.npy")[-1])
    ng2_f    = float(np.load(f"{ft_out_dir}/losses/neg_second_losses_{tag}.npy")[-1])
    ng4_f    = float(np.load(f"{ft_out_dir}/losses/neg_fourth_losses_{tag}.npy")[-1])

    poles_np = poles_all.mean(dim=0).numpy()
    res_np   = residues_all.mean(dim=0).numpy()
    if DATA_SOURCE == "real":
        sim_label = os.path.basename(QMC_SIM_DIR)
    elif DATA_SOURCE == "holstein_jld2":
        sim_label = (
            f"site_holstein_b{HOLSTEIN_BETA:.2f}_w{HOLSTEIN_OMEGA:.2f}_n{HOLSTEIN_N:.2f}"
        )
    else:
        sim_label = os.path.basename(os.path.dirname(SPECTRAL_INPUT_PATH))
    early_stop = n_epochs_run < FINETUNE_EPOCHS

    # Switch to ASCII box-drawing: every char is exactly 1 column wide,
    # so no font/editor can break the alignment of the table.
    W = 78  # total inner width (between the two '|' borders)

    def _border(ch_l, ch_r, fill="-"):
        return ch_l + fill * W + ch_r

    def _line(content):
        # Truncate to W chars, then left-pad to W with spaces.
        s = content[:W].ljust(W)
        return f"|{s}|"

    def _row(label, value="", indent=2, lw=18):
        return _line(" " * indent + f"{label:<{lw}}{value}")

    def _sec(title):
        head = f"  -- {title} "
        return f"+{head:-<{W}}+"

    def _wrap(value, indent=4):
        """Hard-wrap a long string across multiple table rows."""
        room = W - indent
        out = []
        for i in range(0, len(value), room):
            out.append(_line(" " * indent + value[i:i + room]))
        return out or [_line(" " * indent)]

    def _fmt_complex(v, sign="+"):
        # 16-char tuple e.g. "+1.2345-3.6789j" (15) — pad to 16 for column alignment.
        s = f"{v.real:{sign}.4f}{v.imag:+.4f}j"
        return f"{s:<16}"

    def _grid(values, sign="+", indent=4, sep="  "):
        """Lay out complex values in a grid that fits inside the table."""
        item_w   = 16 + len(sep)
        per_row  = max(1, (W - indent) // item_w)
        out = []
        for i in range(0, len(values), per_row):
            chunk = values[i:i + per_row]
            text  = sep.join(_fmt_complex(v, sign=sign) for v in chunk).rstrip()
            out.append(_line(" " * indent + text))
        return out

    lines = []
    lines.append(_border("+", "+", "="))
    lines.append(_line(f"{'RUN SUMMARY':^{W}}"))
    lines.append(_border("+", "+", "="))
    lines.append(_row("tag",              tag))
    lines.append(_row("sID",              str(sID)))
    lines.append(_row("data",             f"{DATA_SOURCE}   {sim_label}"))
    lines.append(_row("beta/Ltau/poles",  f"{BETA} / {INPUT_DIM} / {NUM_POLES}   model = {MODEL_VERSION}"))
    lines.append(_sec("TRAINING"))
    lines.append(_row("epochs",     f"{n_epochs_run} / {FINETUNE_EPOCHS}" + ("  [early stop]" if early_stop else "")))
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
    lines.append(_row("L2 formula",      "(1/N) sum_i  int |A_i(w|G_i) - A(w|<G>)|^2 dw"))
    lines.append(_row("Linf formula",    "(1/N) sum_i  max_w |A_i(w|G_i) - A(w|<G>)|"))
    if au_data:
        lines.append(_sec(f"ACTIVE UNITS  (Var[mu]>{AU_THRESHOLD:g})"))
        lines.append(_row("active units", f"{au_data['active_units']} / {au_data['latent_dim']}"))
        lines.append(_row("KL total",     f"{au_data['kl_total_avg']:.4e}"))
        top_dims = np.argsort(au_data['kl_per_dim'])[::-1][:5]
        top_str  = "  ".join(
            f"d{int(i)}:KL={au_data['kl_per_dim'][i]:.2e}" for i in top_dims
        )
        lines.append(_row("top-5 KL dims", top_str))
    lines.append(_sec("LOSS COMPONENTS  (final epoch, unweighted)"))
    lines.append(_row("chi^2",      f"{chi2_losses[-1]:.4e}",                               indent=4, lw=14))
    lines.append(_row("smooth",     f"{smooth_f:.4e}    pos        {pos_f:.4e}",            indent=4, lw=14))
    lines.append(_row("KL",         f"{kl_f:.4e}    neg G      {ng_f:.4e}",                 indent=4, lw=14))
    lines.append(_row("neg G''",    f"{ng2_f:.4e}    neg G''''  {ng4_f:.4e}",               indent=4, lw=14))
    lines.append(_sec("POLES  (dataset mean)"))
    lines.extend(_grid(poles_np, sign="+"))
    lines.append(_sec("RESIDUES  (dataset mean)"))
    lines.extend(_grid(res_np, sign=""))
    lines.append(_border("+", "+", "="))
    lines.append(_line(f"  output"))
    lines.extend(_wrap(ft_out_dir, indent=4))
    lines.append(_border("+", "+", "="))

    summary_path = f"{ft_out_dir}/run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Run summary → {summary_path}")

    # --- Plot fine-tuning results (local only) ---
    if DO_PLOT:
        print("\n--- Generating fine-tuning plots ---")
        plot_finetune_eval(f"{ft_out_dir}/summary.pt")
        plot_loss_curves(f"{ft_out_dir}/losses", tag=tag,
                         save_path=f"{ft_out_dir}/plots/loss_curves_{tag}.pdf",
                         params=params)
    else:
        print("\nDO_PLOT=False — skipping plots. Run locally after downloading out/.")


if __name__ == "__main__":
    main()
