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
MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"

# --- Model (must match pretraining) ---
LOAD_PRETRAIN = False
NUM_POLES = 6
BETA = 10.0
DTAU = 0.05
INPUT_DIM = int(BETA / DTAU)        # Must match the dimension of the input Green's function
N_NODES = 256
BATCH_SIZE = 50

# --- Pretrained checkpoint ---
PRETRAIN_DIR = os.path.join(
    MAIN_PATH, "out",
    f"pretrain_synthetic_numpoles{NUM_POLES}"
)
PRETRAIN_CHECKPOINT = os.path.join(PRETRAIN_DIR, "model", "best_model_pretrain.pth")

# --- Target dataset ---
SPECTRAL_TYPE = "gaussian_double"
NOISE_S = 1e-04
NOISE_XI = 0.5
INPUT_ID = "inputs-8"  # Which inputs folder to use
DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)

# --- Covariance source (same dataset used for chi2 loss) ---
COVARIANCE_SOURCE = DATA_PATH

# --- Ground truth spectral function (for evaluation plots) ---
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --- Fine-tuning hyperparameters ---
FINETUNE_EPOCHS = 50
FINETUNE_LR = 1e-3        
FINETUNE_PATIENCE = FINETUNE_EPOCHS + 1              # Early stopping enabled with a wide window
FINETUNE_KL_ANNEAL_EPOCHS = 0
_model_tag = {2: "v2", "2t": "v2t", "2s": "v2s", 3: "v3", 4: "v3L"}.get(MODEL_VERSION, f"v{MODEL_VERSION}")
_sim_tag = "-4"
sID = ("pretrained" if LOAD_PRETRAIN else "fresh") + f"-{_model_tag}" + f"{_sim_tag}"

# Loss weights
LAMBDA_CHI2 = 1.0       # Covariance-weighted reconstruction (chi^2 → 1 at perfect fit)
LAMBDA_SMOOTH = 0.0     # Smoothness loss disabled; fights narrow Gaussian peaks
LAMBDA_POS = 0.0        # Positivity loss disabled; pole structure enforces A(omega) >= 0
ALPHA_KL = 1e-6         # Very mild KL regularization (matches Ben)
ETA0 = 1.0              # G(tau) >= 0 penalty — variance-weighted via diag(C)
ETA2 = 1.0              # G''(tau) >= 0 penalty — variance-weighted via diag(C)
ETA4 = 0.0              # G''''(tau) >= 0 penalty — variance-weighted via diag(C)

# ReduceLROnPlateau: uncomment and set USE_SCHEDULER = True to enable.
USE_SCHEDULER = False
LR_FACTOR = 0.5         # Multiply LR by this on plateau
LR_PATIENCE = 10        # Epochs to wait before reducing LR
LR_MIN = 1e-6           # Floor for LR

# Spectral evaluation grid
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -8.0
SMOOTHNESS_WMAX = 8.0

# Reproducibility — set to an integer (e.g. 42) for deterministic runs, or None to disable
SEED = None


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
    print(f"STAGE 2: FINE-TUNING ON {SPECTRAL_TYPE.upper()} DATA")
    print("=" * 80 + "\n")

    tag = "finetune"

    # --- Output directory ---
    ft_out_dir = os.path.join(
        MAIN_PATH, "out",
        f"finetune_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_s{NOISE_S:.0e}_xi{NOISE_XI}-{sID}"
    )
    MakeOutPath(ft_out_dir)

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
        "SPECTRAL_TYPE": SPECTRAL_TYPE,
        "NOISE_S": NOISE_S,
        "NOISE_XI": NOISE_XI,
        "INPUT_ID": INPUT_ID,
        "DATA_PATH": DATA_PATH,
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

    # --- Load full dataset — train on all samples, no held-out split ---
    print(f"Loading dataset from: {DATA_PATH}")
    ft_dataset = GreenFunctionDataset(file_path=DATA_PATH)
    _g = None
    if SEED is not None:
        _g = torch.Generator()
        _g.manual_seed(SEED)
    train_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=_g)
    val_loader = train_loader  # Same data; ReduceLROnPlateau monitors eval-mode training loss
    print(f"  Full dataset: {len(ft_dataset)} samples, {len(train_loader)} batches/epoch")

    # --- Load covariance and build loss modules ---
    print(f"Loading covariance from: {COVARIANCE_SOURCE}")
    C = load_covariance_from_dqmc(COVARIANCE_SOURCE)

    kl_fn         = KLDivergenceLoss().to(DEVICE)
    chi2_fn       = ChiSquaredLoss(C).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX).to(DEVICE)
    positivity_fn = SpectralPositivityLoss(Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX).to(DEVICE)
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

    print(f"\nFine-tuning done. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")

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

    # Deterministic spectral evaluation: z = mu (mode of the posterior)
    print("\n--- Deterministic spectral evaluation (z = mu) ---")
    omega_eval_grid = torch.linspace(SMOOTHNESS_WMIN, SMOOTHNESS_WMAX, SMOOTHNESS_NW)
    with torch.no_grad():
        A_mean = spectral_from_poles(
            poles_all.to(DEVICE), residues_all.to(DEVICE), omega_eval_grid.to(DEVICE)
        ).cpu()
    A_std = torch.zeros_like(A_mean)
    print(f"Deterministic spectral shape: {A_mean.shape}")

    torch.save({
        "inputs":       G_input_all,
        "recon":        G_recon_all,
        "poles":        poles_all,
        "residues":     residues_all,
        "inputs_avg":   G_input_all.mean(dim=0),
        "recon_avg":    G_recon_all.mean(dim=0),
        "poles_avg":    poles_all.mean(dim=0),
        "residues_avg": residues_all.mean(dim=0),
        "A_mean":       A_mean,
        "A_std":        A_std,
        "omega_eval_grid":    omega_eval_grid,
        "beta":               BETA,
        "Ltau":               INPUT_DIM,
        "spectral_input_path": SPECTRAL_INPUT_PATH,
        "noise_var":    float(np.mean(np.diag(np.array(C)))),
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved to {ft_out_dir}/summary.pt")
    print(f"Poles mean: {poles_all.mean(dim=0)}")
    print(f"Residues mean: {residues_all.mean(dim=0)}")

    # --- Plot fine-tuning results ---
    print("\n--- Generating fine-tuning plots ---")
    plot_finetune_eval(f"{ft_out_dir}/summary.pt")
    plot_loss_curves(f"{ft_out_dir}/losses", tag=tag,
                     save_path=f"{ft_out_dir}/plots/loss_curves_{tag}.pdf",
                     params=params)


if __name__ == "__main__":
    main()
