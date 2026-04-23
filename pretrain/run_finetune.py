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
    read_model_params,
)
from pretrain.pretrain_losses import (  # type: ignore
    KLDivergenceLoss,
    ChiSquaredLoss,
    SpectralSmoothnessLoss,
    SpectralPositivityLoss,
    NegativeGreenPenalty,
    NegativeSecondDerivativePenalty,
    NegativeFourthDerivativePenalty,
    finetune_total_loss,
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

# --- Data source: "synthetic" or "real" ---
DATA_SOURCE = "real"   # switch between synthetic CSV and real QMC JLD2

# --- Model ---
LOAD_PRETRAIN = False
NUM_POLES = 3
N_NODES = 256
BATCH_SIZE = 10

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
# Shared physical parameters
# --------------------------------------------------------------------------
if DATA_SOURCE == "real":
    _qmc_params = read_model_params(QMC_SIM_DIR)
    BETA      = _qmc_params["beta"]
    DTAU      = _qmc_params["dtau"]
    INPUT_DIM = _qmc_params["L_tau"]   # authoritative — no floating-point precision issue
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
FINETUNE_EPOCHS = 1000
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

# ReduceLROnPlateau: uncomment and set USE_SCHEDULER = True to enable.
USE_SCHEDULER = False
LR_FACTOR = 0.5         # Multiply LR by this on plateau
LR_PATIENCE = 10        # Epochs to wait before reducing LR
LR_MIN = 1e-6           # Floor for LR

# Spectral evaluation grid
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -20.0
SMOOTHNESS_WMAX = 20.0

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
    _out_label = (
        f"real-{os.path.basename(QMC_SIM_DIR)}"
        if DATA_SOURCE == "real"
        else f"synthetic-{SPECTRAL_TYPE}_s{NOISE_S:.0e}_xi{NOISE_XI}"
    )
    _init_tag  = "pretrained" if LOAD_PRETRAIN else "fresh"
    _base_name = f"finetune_{_out_label}_numpoles{NUM_POLES}-{_init_tag}-{_model_tag}"
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
    ft_out_dir = os.path.join(_out_root, f"finetune_{_out_label}_numpoles{NUM_POLES}-{sID}")
    print(f"Output directory: {ft_out_dir}")
    MakeOutPath(ft_out_dir)

    # --- Load full dataset — train on all samples, no held-out split ---
    if DATA_SOURCE == "real":
        print(f"Loading real QMC dataset from: {QMC_SIM_DIR}")
        ft_dataset = SmoQyV2Dataset(QMC_SIM_DIR, r1=0, r2=0)
        ft_dataset.summary()
        C = load_covariance_v2(QMC_SIM_DIR, r1=0, r2=0)
    else:
        print(f"Loading synthetic dataset from: {SYNTHETIC_DATA_PATH}")
        ft_dataset = GreenFunctionDataset(file_path=SYNTHETIC_DATA_PATH)
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
        "BETA": BETA,
        "DTAU": DTAU,
        "BATCH_SIZE": BATCH_SIZE,
        "DATA_SOURCE": DATA_SOURCE,
        **({} if DATA_SOURCE == "real" else {
            "SPECTRAL_TYPE": SPECTRAL_TYPE,
            "NOISE_S": NOISE_S,
            "NOISE_XI": NOISE_XI,
            "INPUT_ID": INPUT_ID,
        }),
        "DATA_PATH": QMC_SIM_DIR if DATA_SOURCE == "real" else SYNTHETIC_DATA_PATH,
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
    }
    with open(f"{ft_out_dir}/params.json", "w") as _f:
        json.dump(params, _f, indent=2)
    print(f"Run parameters saved to {ft_out_dir}/params.json")

    # --- Load covariance and build loss modules ---
    print(f"Covariance shape: {C.shape}")

    kl_fn         = KLDivergenceLoss().to(DEVICE)
    chi2_fn       = ChiSquaredLoss(C).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX).to(DEVICE)
    positivity_fn = SpectralPositivityLoss().to(DEVICE)
    neg_green_fn  = NegativeGreenPenalty(C).to(DEVICE)
    neg_second_fn = NegativeSecondDerivativePenalty(C).to(DEVICE)
    neg_fourth_fn = NegativeFourthDerivativePenalty(C).to(DEVICE)

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
            )
            eval_loss += loss.item() * B

            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())

    eval_loss /= len(ft_dataset)
    print(f"Final Loss: {eval_loss:.4e}")

    G_input_all   = torch.cat(G_input_all,   dim=0)
    G_recon_all   = torch.cat(G_recon_all,   dim=0)
    poles_all     = torch.cat(poles_all,     dim=0)
    residues_all  = torch.cat(residues_all,  dim=0)

    torch.save({
        "inputs":       G_input_all,
        "recon":        G_recon_all,
        "poles":        poles_all,
        "residues":     residues_all,
        "inputs_avg":   G_input_all.mean(dim=0),
        "recon_avg":    G_recon_all.mean(dim=0),
        "poles_avg":    poles_all.mean(dim=0),
        "residues_avg": residues_all.mean(dim=0),
        "beta":               BETA,
        "Ltau":               INPUT_DIM,
        "spectral_input_path": None if DATA_SOURCE == "real" else SPECTRAL_INPUT_PATH,
        "noise_var":    float(np.mean(np.diag(np.array(C)))),
        "data_source":        DATA_SOURCE,
        "sim_name":           os.path.basename(QMC_SIM_DIR) if DATA_SOURCE == "real" else None,
        "num_poles":          NUM_POLES,
        "sID":                sID,
        "best_chi2":          best_chi2,
        "best_chi2_epoch":    best_chi2_ep,
        "final_chi2":         final_chi2,
        "n_epochs":           n_epochs_run,
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved to {ft_out_dir}/summary.pt")
    print(f"Poles mean: {poles_all.mean(dim=0)}")
    print(f"Residues mean: {residues_all.mean(dim=0)}")

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
