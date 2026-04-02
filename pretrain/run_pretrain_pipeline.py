"""
Main pretraining pipeline orchestrator — model2_leaky ("2L").

Stage 1: Pretrain VAE on synthetic single-Gaussian spectral functions.
         The covariance matrix is loaded from the Stage 2 target DQMC dataset
         so that the pretraining chi-squared loss uses the same noise structure
         the model will see during fine-tuning.

Stage 2: Fine-tune on the target DQMC dataset using chi-squared loss and
         variance-weighted negativity penalties.

Hyperparameters are kept identical to run_finetune.py (LOAD_PRETRAIN=False)
so that the pretrained+finetuned run is a fair comparison against fresh init.

Usage:
    python pretrain/run_pretrain_pipeline.py
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

from model2_leaky import VariationalAutoEncoder2  # type: ignore
from pretrain.plot_results import (  # type: ignore
    plot_pretrain_eval, plot_loss_curves, plot_finetune_eval,
)
from pretrain.synthetic_data import (  # type: ignore
    SavedSyntheticDataset,
    SyntheticGaussianDataset,
    generate_and_save,
    load_covariance_from_dqmc,
    cholesky_sqrt,
    compute_covariance_from_bins,
)
from pretrain.pretrain_losses import (  # type: ignore
    KLDivergenceLoss,
    SpectralMSELoss,
    ChiSquaredLoss,
    SpectralSmoothnessLoss,
    SpectralPositivityLoss,
    SpectralMomentLoss,
    NegativeGreenPenalty,
    NegativeSecondDerivativePenalty,
    NegativeFourthDerivativePenalty,
    finetune_total_loss,
    pretrain_total_loss,
    spectral_from_poles,
)
from pretrain.train_pretrain import train_pretrain  # type: ignore
from pretrain.train_finetune import train_finetune  # type: ignore

from data_process import GreenFunctionDataset  # type: ignore
from utils import MakeOutPath  # type: ignore

# ==========================================================================
# CONFIGURATION
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"

# --- Model (must match run_finetune.py) ---
NUM_POLES = 6
BETA = 10.0
DTAU = 0.05
INPUT_DIM = int(BETA / DTAU)   # 200
N_NODES = 256
BATCH_SIZE = 50

# --- Run label (used in output dir names) ---
sID = "pretrained-v2L"

# Reproducibility — set to an integer (e.g. 42) for deterministic runs, or None to disable
SEED = None

# --------------------------------------------------------------------------
# STAGE 2 — Target DQMC dataset
# Change these to switch between gaussian, gaussian_double, semicircle, etc.
# The covariance loaded from this dataset is also used in Stage 1.
# --------------------------------------------------------------------------
SPECTRAL_TYPE = "gaussian_double"
NOISE_S = 1e-04
NOISE_XI = 0.5
INPUT_ID = "inputs-8"

DQMC_DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --------------------------------------------------------------------------
# STAGE 1 — Pretraining hyperparameters (synthetic single-Gaussian data)
# --------------------------------------------------------------------------

# Synthetic data generation
N_SAMPLES = 1000
MU_MAX = 0.5                 # Peak centers: Uniform(-MU_MAX, +MU_MAX)
SIGMA_MIN = 0.3              # Peak widths: LogUniform(SIGMA_MIN, SIGMA_MAX)
SIGMA_MAX = 3.0
GENERATE_FRESH = False        # If True, generate new data; if False, load from disk
USE_ON_THE_FLY = False       # If True, use on-the-fly dataset (infinite variety)
SYNTHETIC_DATA_DIR = os.path.join(MAIN_PATH, "Data", "datasets", "synthetic-gaussian-pretrain")

PRE_EPOCHS = 100
PRE_LR = 1e-3
PRE_LAMBDA_SPEC = 1.0
PRE_LAMBDA_MOMENT = 1.0
PRE_LAMBDA_SMOOTH = 0.0
PRE_LAMBDA_CHI2 = 0.0
PRE_ALPHA_KL = 0.0
PRE_PATIENCE = PRE_EPOCHS + 1  # Early stopping disabled — train for all epochs
PRE_LR_FACTOR = 0.5
PRE_LR_PATIENCE = 20
PRE_LR_MIN = 1e-6

# Spectral / smoothness evaluation grids
SPECTRAL_W = 6.0
SPECTRAL_N_GL = 512
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -8.0
SMOOTHNESS_WMAX = 8.0

# --------------------------------------------------------------------------
# STAGE 2 — Fine-tuning hyperparameters
# MUST be kept identical to run_finetune.py (LOAD_PRETRAIN=False) for a fair
# fresh vs. pretrained comparison.
# --------------------------------------------------------------------------

DO_FINETUNE = True
FINETUNE_EPOCHS = 50
FINETUNE_LR = 1e-3
FINETUNE_PATIENCE = FINETUNE_EPOCHS + 1  # Disabled — train for all epochs
FINETUNE_KL_ANNEAL_EPOCHS = 0
FINETUNE_LAMBDA_CHI2 = 1.0
FINETUNE_LAMBDA_SMOOTH = 0.0     # Disabled — fights narrow Gaussian peaks
FINETUNE_LAMBDA_POS = 0.0        # Disabled — pole structure enforces A(omega) >= 0
FINETUNE_ALPHA_KL = 1e-6         # Very mild KL regularization (matches Ben)
FINETUNE_ETA0 = 1.0              # G(tau) >= 0 penalty
FINETUNE_ETA2 = 1.0              # G''(tau) >= 0 penalty
FINETUNE_ETA4 = 0.0              # G''''(tau) >= 0 penalty — keep OFF (causes divergence)
FINETUNE_USE_SCHEDULER = False   # ReduceLROnPlateau disabled (matches run_finetune.py)
FINETUNE_LR_FACTOR = 0.5
FINETUNE_LR_PATIENCE = 10
FINETUNE_LR_MIN = 1e-6

# ==========================================================================
# HELPERS
# ==========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    if SEED is not None:
        set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED if SEED is not None else 'disabled (random)'}")
    print(f"Model: NUM_POLES={NUM_POLES}, INPUT_DIM={INPUT_DIM}, N_NODES={N_NODES}")
    print(f"Target dataset: {DQMC_DATA_PATH}")

    # ------------------------------------------------------------------
    # Load covariance once — shared by Stage 1 (chi2) and Stage 2 losses
    # ------------------------------------------------------------------
    print(f"\nLoading covariance from: {DQMC_DATA_PATH}")
    C = load_covariance_from_dqmc(DQMC_DATA_PATH)
    sqrt_C = cholesky_sqrt(C)

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    model = VariationalAutoEncoder2(
        input_dim=INPUT_DIM,
        num_poles=NUM_POLES,
        beta=BETA,
        N_nodes=N_NODES,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # STAGE 1: Pretraining on synthetic single-Gaussian data
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: PRETRAINING ON SYNTHETIC GAUSSIAN DATA")
    print("=" * 80 + "\n")

    # --- Generate / load synthetic data ---
    if GENERATE_FRESH and not USE_ON_THE_FLY:
        print("Generating fresh synthetic dataset...")
        generate_and_save(
            N=N_SAMPLES,
            mu_max=MU_MAX,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            beta=BETA,
            dtau=DTAU,
            output_dir=SYNTHETIC_DATA_DIR,
            covariance_source=DQMC_DATA_PATH,
            use_quadrature=False,
            N_gl=256,
            omega_max=20.0,
            seed=42,
        )

    if USE_ON_THE_FLY:
        print("Using on-the-fly synthetic dataset")
        dataset = SyntheticGaussianDataset(
            N_samples=N_SAMPLES,
            mu_max=MU_MAX,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            beta=BETA,
            dtau=DTAU,
            sqrt_C=sqrt_C,
        )
    else:
        print(f"Loading saved synthetic dataset from {SYNTHETIC_DATA_DIR}")
        dataset = SavedSyntheticDataset(SYNTHETIC_DATA_DIR)

    _g_pre = None
    if SEED is not None:
        _g_pre = torch.Generator()
        _g_pre.manual_seed(SEED)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, generator=_g_pre)
    val_loader = train_loader  # Same data; scheduler monitors eval-mode loss
    print(f"Dataset: {len(dataset)} samples, {len(train_loader)} batches/epoch")

    # --- Loss modules (shared covariance C) ---
    spectral_mse_fn = SpectralMSELoss(W=SPECTRAL_W, N_gl=SPECTRAL_N_GL).to(DEVICE)
    chi2_fn = ChiSquaredLoss(C).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    moment_fn = SpectralMomentLoss(W=SPECTRAL_W, N_gl=SPECTRAL_N_GL).to(DEVICE)

    # --- Output directory — include sID to avoid overwriting previous runs ---
    pre_out_dir = os.path.join(
        MAIN_PATH, "out",
        f"pretrain_synthetic_numpoles{NUM_POLES}-{sID}"
    )
    MakeOutPath(pre_out_dir)

    # --- Optimizer ---
    pre_optimizer = optim.AdamW(model.parameters(), lr=PRE_LR, weight_decay=1e-3)
    pre_scheduler = ReduceLROnPlateau(
        pre_optimizer, mode="min", factor=PRE_LR_FACTOR,
        patience=PRE_LR_PATIENCE, min_lr=PRE_LR_MIN
    )

    # --- Train Stage 1 ---
    print(f"Loss weights: lambda_chi2={PRE_LAMBDA_CHI2}, lambda_spec={PRE_LAMBDA_SPEC}, "
          f"lambda_smooth={PRE_LAMBDA_SMOOTH}, alpha_kl={PRE_ALPHA_KL}, "
          f"lambda_moment={PRE_LAMBDA_MOMENT}")

    best_pre_loss, best_pre_epoch = train_pretrain(
        model=model,
        optimizer=pre_optimizer,
        scheduler=pre_scheduler,
        num_epochs=PRE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=train_loader,
        val_loader=val_loader,
        spectral_mse_fn=spectral_mse_fn,
        chi2_fn=chi2_fn,
        smoothness_fn=smoothness_fn,
        lambda_chi2=PRE_LAMBDA_CHI2,
        lambda_spec=PRE_LAMBDA_SPEC,
        lambda_smooth=PRE_LAMBDA_SMOOTH,
        alpha_kl=PRE_ALPHA_KL,
        device=DEVICE,
        out_dir=pre_out_dir,
        tag="pretrain",
        patience=PRE_PATIENCE,
        deterministic=False,
        moment_fn=moment_fn,
        lambda_moment=PRE_LAMBDA_MOMENT,
    )

    print(f"\nPretraining done. Best val loss: {best_pre_loss:.4e} at epoch {best_pre_epoch+1}")

    # Load best pretrained weights
    model.load_state_dict(
        torch.load(f"{pre_out_dir}/model/best_model_pretrain.pth", weights_only=True)
    )

    # --- Evaluate pretraining ---
    print("\n--- Pretraining evaluation ---")
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=False)
    with torch.no_grad():
        batch_pre = next(iter(eval_loader))
        G_tilde, mu_targets, sigma_targets = batch_pre
        G_tilde = G_tilde.to(DEVICE)
        mu_targets = mu_targets.to(DEVICE)
        sigma_targets = sigma_targets.to(DEVICE)
        mu_vae, logvar_vae, z, poles, residues, G_recon = model(G_tilde)

    torch.save({
        "G_input":       G_tilde.cpu(),
        "G_recon":       G_recon.cpu(),
        "poles":         poles.cpu(),
        "residues":      residues.cpu(),
        "mu_targets":    mu_targets.cpu(),
        "sigma_targets": sigma_targets.cpu(),
        "beta":          BETA,
        "Ltau":          INPUT_DIM,
    }, f"{pre_out_dir}/pretrain_eval.pt")
    print(f"Pretraining evaluation saved to {pre_out_dir}/pretrain_eval.pt")

    plot_pretrain_eval(f"{pre_out_dir}/pretrain_eval.pt")
    plot_loss_curves(f"{pre_out_dir}/losses", tag="pretrain",
                     save_path=f"{pre_out_dir}/plots/loss_curves_pretrain.pdf")

    # ------------------------------------------------------------------
    # STAGE 2: Fine-tuning on target DQMC dataset
    # ------------------------------------------------------------------
    if not DO_FINETUNE:
        print("\nSkipping fine-tuning (DO_FINETUNE=False).")
        return

    print("\n" + "=" * 80)
    print(f"STAGE 2: FINE-TUNING ON {SPECTRAL_TYPE.upper()} DATA")
    print("=" * 80 + "\n")

    tag = "finetune"

    ft_out_dir = os.path.join(
        MAIN_PATH, "out",
        f"finetune_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_s{NOISE_S:.0e}_xi{NOISE_XI}-{sID}"
    )
    MakeOutPath(ft_out_dir)

    # --- Save run parameters for reference ---
    params = {
        "sID": sID,
        "MODEL_VERSION": "2L",
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
        "DATA_PATH": DQMC_DATA_PATH,
        "LOAD_PRETRAIN": True,
        "PRETRAIN_EPOCHS": PRE_EPOCHS,
        "PRETRAIN_LR": PRE_LR,
        "PRETRAIN_LAMBDA_CHI2": PRE_LAMBDA_CHI2,
        "PRETRAIN_LAMBDA_SPEC": PRE_LAMBDA_SPEC,
        "PRETRAIN_LAMBDA_SMOOTH": PRE_LAMBDA_SMOOTH,
        "PRETRAIN_ALPHA_KL": PRE_ALPHA_KL,
        "PRETRAIN_LAMBDA_MOMENT": PRE_LAMBDA_MOMENT,
        "FINETUNE_EPOCHS": FINETUNE_EPOCHS,
        "FINETUNE_LR": FINETUNE_LR,
        "FINETUNE_PATIENCE": FINETUNE_PATIENCE,
        "FINETUNE_KL_ANNEAL_EPOCHS": FINETUNE_KL_ANNEAL_EPOCHS,
        "LAMBDA_CHI2": FINETUNE_LAMBDA_CHI2,
        "LAMBDA_SMOOTH": FINETUNE_LAMBDA_SMOOTH,
        "LAMBDA_POS": FINETUNE_LAMBDA_POS,
        "ALPHA_KL": FINETUNE_ALPHA_KL,
        "ETA0": FINETUNE_ETA0,
        "ETA2": FINETUNE_ETA2,
        "ETA4": FINETUNE_ETA4,
        "USE_SCHEDULER": FINETUNE_USE_SCHEDULER,
        "LR_FACTOR": FINETUNE_LR_FACTOR,
        "LR_PATIENCE": FINETUNE_LR_PATIENCE,
        "LR_MIN": FINETUNE_LR_MIN,
        "SMOOTHNESS_NW": SMOOTHNESS_NW,
        "SMOOTHNESS_WMIN": SMOOTHNESS_WMIN,
        "SMOOTHNESS_WMAX": SMOOTHNESS_WMAX,
        "SEED": SEED,
    }
    with open(f"{ft_out_dir}/params.json", "w") as _f:
        json.dump(params, _f, indent=2)
    print(f"Run parameters saved to {ft_out_dir}/params.json")

    # --- Load full DQMC dataset ---
    print(f"Loading dataset from: {DQMC_DATA_PATH}")
    ft_dataset = GreenFunctionDataset(file_path=DQMC_DATA_PATH)
    _g_ft = None
    if SEED is not None:
        _g_ft = torch.Generator()
        _g_ft.manual_seed(SEED)
    ft_train_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 drop_last=False, generator=_g_ft)
    ft_val_loader = ft_train_loader  # Scheduler monitors eval-mode training loss
    print(f"  Full dataset: {len(ft_dataset)} samples, {len(ft_train_loader)} batches/epoch")

    # --- Build fine-tuning loss modules (same C as Stage 1) ---
    ft_kl_fn         = KLDivergenceLoss().to(DEVICE)
    ft_chi2_fn       = ChiSquaredLoss(C).to(DEVICE)
    ft_smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    ft_positivity_fn = SpectralPositivityLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    ft_neg_green_fn  = NegativeGreenPenalty(C).to(DEVICE)
    ft_neg_second_fn = NegativeSecondDerivativePenalty(C).to(DEVICE)
    ft_neg_fourth_fn = NegativeFourthDerivativePenalty(C).to(DEVICE)

    # --- Optimizer ---
    ft_optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.0)
    ft_scheduler = ReduceLROnPlateau(
        ft_optimizer, mode="min", factor=FINETUNE_LR_FACTOR,
        patience=FINETUNE_LR_PATIENCE, min_lr=FINETUNE_LR_MIN
    ) if FINETUNE_USE_SCHEDULER else None

    # --- Fine-tune ---
    print(f"\nLoss weights: lambda_chi2={FINETUNE_LAMBDA_CHI2}, lambda_smooth={FINETUNE_LAMBDA_SMOOTH}, "
          f"lambda_pos={FINETUNE_LAMBDA_POS}, alpha_kl={FINETUNE_ALPHA_KL}, "
          f"eta0={FINETUNE_ETA0}, eta2={FINETUNE_ETA2}, eta4={FINETUNE_ETA4}")
    print(f"Scheduler: {'ReduceLROnPlateau' if FINETUNE_USE_SCHEDULER else 'disabled'}")
    print(f"Training on full dataset ({len(ft_dataset)} samples), early stopping disabled.")

    best_ft_loss, best_ft_epoch = train_finetune(
        model=model,
        optimizer=ft_optimizer,
        scheduler=ft_scheduler,
        num_epochs=FINETUNE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=ft_train_loader,
        val_loader=ft_val_loader,
        kl_fn=ft_kl_fn,
        chi2_fn=ft_chi2_fn,
        smoothness_fn=ft_smoothness_fn,
        neg_green_fn=ft_neg_green_fn,
        neg_second_fn=ft_neg_second_fn,
        neg_fourth_fn=ft_neg_fourth_fn,
        positivity_fn=ft_positivity_fn,
        lambda_chi2=FINETUNE_LAMBDA_CHI2,
        lambda_smooth=FINETUNE_LAMBDA_SMOOTH,
        lambda_pos=FINETUNE_LAMBDA_POS,
        alpha_kl=FINETUNE_ALPHA_KL,
        eta0=FINETUNE_ETA0,
        eta2=FINETUNE_ETA2,
        eta4=FINETUNE_ETA4,
        device=DEVICE,
        out_dir=ft_out_dir,
        tag=tag,
        patience=FINETUNE_PATIENCE,
        kl_anneal_epochs=FINETUNE_KL_ANNEAL_EPOCHS,
    )

    print(f"\nFine-tuning done. Best val loss: {best_ft_loss:.4e} at epoch {best_ft_epoch+1}")

    # Load best fine-tuned model
    model.load_state_dict(
        torch.load(f"{ft_out_dir}/model/best_model_{tag}.pth", weights_only=True)
    )

    # ------------------------------------------------------------------
    # Full-dataset evaluation (deterministic: z = mu)
    # Matches run_finetune.py evaluation exactly for fair comparison.
    # ------------------------------------------------------------------
    print("\n--- Full-dataset evaluation (deterministic, z = mu) ---")
    model.eval()
    eval_loss = 0.0
    G_input_all, G_recon_all = [], []
    poles_all, residues_all = [], []

    with torch.no_grad():
        for batch in ft_train_loader:
            B = batch.shape[0]
            batch = batch.view(B, INPUT_DIM).to(DEVICE)

            mu_vae, logvar_vae, z, poles, residues, G_recon = model(batch, deterministic=True)

            loss, _, _, _, _, _, _, _ = finetune_total_loss(
                G_recon, batch,
                mu_vae, logvar_vae,
                poles, residues,
                ft_kl_fn, ft_chi2_fn, ft_smoothness_fn,
                ft_neg_green_fn, ft_neg_second_fn, ft_neg_fourth_fn,
                ft_positivity_fn,
                FINETUNE_LAMBDA_CHI2, FINETUNE_LAMBDA_SMOOTH, FINETUNE_LAMBDA_POS,
                FINETUNE_ALPHA_KL, FINETUNE_ETA0, FINETUNE_ETA2, FINETUNE_ETA4,
            )
            eval_loss += loss.item() * B

            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())

    eval_loss /= len(ft_dataset)
    print(f"Final eval loss: {eval_loss:.4e}")

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
        "inputs":              G_input_all,
        "recon":               G_recon_all,
        "poles":               poles_all,
        "residues":            residues_all,
        "inputs_avg":          G_input_all.mean(dim=0),
        "recon_avg":           G_recon_all.mean(dim=0),
        "poles_avg":           poles_all.mean(dim=0),
        "residues_avg":        residues_all.mean(dim=0),
        "A_mean":              A_mean,
        "A_std":               A_std,
        "omega_eval_grid":     omega_eval_grid,
        "beta":                BETA,
        "Ltau":                INPUT_DIM,
        "spectral_input_path": SPECTRAL_INPUT_PATH,
        "noise_var":           float(np.mean(np.diag(np.array(C)))),
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved to {ft_out_dir}/summary.pt")
    print(f"Poles mean:    {poles_all.mean(dim=0)}")
    print(f"Residues mean: {residues_all.mean(dim=0)}")

    # --- Plot fine-tuning results ---
    print("\n--- Generating fine-tuning plots ---")
    plot_finetune_eval(f"{ft_out_dir}/summary.pt")
    plot_loss_curves(f"{ft_out_dir}/losses", tag=tag,
                     save_path=f"{ft_out_dir}/plots/loss_curves_{tag}.pdf",
                     params=params)


if __name__ == "__main__":
    main()
