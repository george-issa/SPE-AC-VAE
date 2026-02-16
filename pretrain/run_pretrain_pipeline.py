"""
Main pretraining pipeline orchestrator.

Stage 1: Pretrain VAE on synthetic single-Gaussian spectral functions
         with spectral MSE + chi-squared + smoothness losses.

Stage 2 (optional): Fine-tune on target DQMC data using existing loss functions.

Usage:
    python pretrain/run_pretrain_pipeline.py
"""

import os
import sys
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model2 import VariationalAutoEncoder2  # type: ignore
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
    SpectralMSELoss,
    ChiSquaredLoss,
    SpectralSmoothnessLoss,
    SpectralPositivityLoss,
)
from pretrain.train_pretrain import train_pretrain  # type: ignore
from pretrain.train_finetune import train_finetune, finetune_total_loss  # type: ignore

# For Stage 2 fine-tuning
from data_process import GreenFunctionDataset  # type: ignore
from utils import MakeOutPath, STD, TrainTestSplit, LoadData, PrintDataSize  # type: ignore

# ==========================================================================
# CONFIGURATION
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = "/Users/georgeissa/Documents/AC"

# --- Model ---
NUM_POLES = 4
LATENT_DIM = 4 * NUM_POLES - 2
HIDDEN_DIM1 = 8 * LATENT_DIM
HIDDEN_DIM2 = LATENT_DIM
HIDDEN_DIM3 = 2 * LATENT_DIM
HIDDEN_DIM4 = 4 * LATENT_DIM
BETA = 10.0
DTAU = 0.05
INPUT_DIM = int(BETA / DTAU)  # 200
N_NODES = 256
BATCH_SIZE = 32

# --- Synthetic data generation ---
N_SAMPLES = 10000           # Total synthetic samples to generate
SIGMA_MU = 1.0              # Std of Normal for mu_n
MU_SIGMA = 3.0              # Mean of InverseGamma for sigma_n (~W/2)
SIGMA_SIGMA = 0.5           # Std of InverseGamma for sigma_n
GENERATE_FRESH = True       # If True, generate new data; if False, load from disk
USE_ON_THE_FLY = False      # If True, use on-the-fly dataset (infinite variety)
SYNTHETIC_DATA_DIR = os.path.join(MAIN_PATH, "Data", "datasets", "synthetic-gaussian-pretrain")

# --- Pretraining hyperparameters ---
PRE_EPOCHS = 100
PRE_LR = 1e-3
PRE_LAMBDA_CHI2 = 0.0       # Weight for chi-squared loss (chi2/Ltau -> 1.0 at perfect fit)
PRE_LAMBDA_SPEC = 1.0       # Weight for spectral MSE loss (raw spec ~ 1e-6 * chi2 at init)
PRE_LAMBDA_SMOOTH = 1e-3    # Weight for smoothness regularizer (raw smooth ~ chi2 at init)
PRE_ALPHA_KL = 0.0          # KL weight (0 = pure reconstruction)
PRE_PATIENCE = 15

# --- Spectral loss parameters ---
SPECTRAL_W = 6.0            # Bandwidth for spectral MSE integration
SPECTRAL_N_GL = 256         # GL nodes for spectral MSE
SMOOTHNESS_NW = 500         # Grid points for smoothness loss
SMOOTHNESS_WMIN = -8.0
SMOOTHNESS_WMAX = 8.0

# --- Stage 2: Fine-tuning (optional) ---
DO_FINETUNE = True
SPECTRAL_TYPE = "gaussian_double"
FINETUNE_EPOCHS = 200
FINETUNE_LR = 3e-3
FINETUNE_PATIENCE = 15
FINETUNE_LAMBDA_CHI2 = 1.0    # Covariance-weighted reconstruction
FINETUNE_LAMBDA_SMOOTH = 1e-3 # Smoothness on predicted A(omega)
FINETUNE_LAMBDA_POS = 1.0     # Spectral positivity penalty
FINETUNE_ALPHA_KL = 0.001     # KL divergence weight
FINETUNE_ETA0 = 1.0           # Negative Green's penalty weight
FINETUNE_ETA2 = 1.0           # Negative second derivative penalty weight
FINETUNE_T0 = 50              # Cosine annealing: period of first restart (epochs)
FINETUNE_T_MULT = 2           # Cosine annealing: T_0 multiplier after each restart
NOISE_S = 1e-05
NOISE_XI = 0.5
INPUT_ID = "inputs-7"
sID = "pretrained"

# Covariance source: path to DQMC Gbins CSV, or None to compute from synthetic data
COVARIANCE_SOURCE = os.path.join(
    MAIN_PATH, "Data", "datasets", 
    f"half-filled-{SPECTRAL_TYPE}", {INPUT_ID}, 
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)

# Ground truth spectral function (for evaluation plots)
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", 
    f"half-filled-{SPECTRAL_TYPE}", {INPUT_ID}, 
    "spectral_input.csv"
)

# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Model: NUM_POLES={NUM_POLES}, LATENT_DIM={LATENT_DIM}, INPUT_DIM={INPUT_DIM}")

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    model = VariationalAutoEncoder2(
        input_dim=INPUT_DIM,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        latent_dim=LATENT_DIM,
        hidden_dim3=HIDDEN_DIM3,
        hidden_dim4=HIDDEN_DIM4,
        num_poles=NUM_POLES,
        beta=BETA,
        N_nodes=N_NODES,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # STAGE 1: Pretraining on synthetic Gaussians
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: PRETRAINING ON SYNTHETIC GAUSSIAN DATA")
    print("=" * 80 + "\n")

    # --- Data ---
    if GENERATE_FRESH and not USE_ON_THE_FLY:
        print("Generating fresh synthetic dataset...")
        generate_and_save(
            N=N_SAMPLES,
            sigma_mu=SIGMA_MU,
            mu_sigma=MU_SIGMA,
            sigma_sigma=SIGMA_SIGMA,
            beta=BETA,
            dtau=DTAU,
            output_dir=SYNTHETIC_DATA_DIR,
            covariance_source=COVARIANCE_SOURCE,
            use_quadrature=False,  # Use GL for speed
            N_gl=256,
            omega_max=20.0,
            seed=42,
        )

    # Load covariance matrix for losses
    if COVARIANCE_SOURCE is not None and os.path.exists(COVARIANCE_SOURCE):
        print(f"Loading covariance from DQMC: {COVARIANCE_SOURCE}")
        C = load_covariance_from_dqmc(COVARIANCE_SOURCE)
    else:
        cov_path = os.path.join(SYNTHETIC_DATA_DIR, "covariance.npy")
        print(f"Loading covariance from: {cov_path}")
        C = np.load(cov_path)

    sqrt_C = cholesky_sqrt(C)

    # Create dataset
    if USE_ON_THE_FLY:
        print("Using on-the-fly synthetic dataset")
        dataset = SyntheticGaussianDataset(
            N_samples=N_SAMPLES,
            sigma_mu=SIGMA_MU,
            mu_sigma=MU_SIGMA,
            sigma_sigma=SIGMA_SIGMA,
            beta=BETA,
            dtau=DTAU,
            sqrt_C=sqrt_C,
        )
    else:
        print(f"Loading saved synthetic dataset from {SYNTHETIC_DATA_DIR}")
        dataset = SavedSyntheticDataset(SYNTHETIC_DATA_DIR)

    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}")

    # --- Loss modules ---
    spectral_mse_fn = SpectralMSELoss(W=SPECTRAL_W, N_gl=SPECTRAL_N_GL).to(DEVICE)
    chi2_fn = ChiSquaredLoss(C).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)

    # --- Output directory ---
    pre_out_dir = os.path.join(
        MAIN_PATH, "VAE_Library",
        f"pretrain_synthetic_numpoles{NUM_POLES}-{sID}"
    )
    MakeOutPath(pre_out_dir)

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=PRE_LR, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # --- Train ---
    best_val_loss, best_epoch = train_pretrain(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
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
    )

    print(f"\nPretraining done. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")

    # Load best pretrained model
    model.load_state_dict(torch.load(f"{pre_out_dir}/model/best_model_pretrain.pth", weights_only=True))

    # ------------------------------------------------------------------
    # Evaluate pretraining: reconstruct spectral functions on test data
    # ------------------------------------------------------------------
    print("\n--- Pretraining evaluation ---")
    model.eval()

    # Grab a few samples from the validation set
    eval_loader = DataLoader(val_dataset, batch_size=min(64, val_size), shuffle=False)
    with torch.no_grad():
        batch = next(iter(eval_loader))
        G_tilde, mu_targets, sigma_targets = batch
        G_tilde = G_tilde.to(DEVICE)

        mu_vae, logvar_vae, z, poles, residues, G_recon = model(G_tilde)

    # Save pretraining evaluation
    torch.save({
        "G_input": G_tilde.cpu(),
        "G_recon": G_recon.cpu(),
        "poles": poles.cpu(),
        "residues": residues.cpu(),
        "mu_targets": mu_targets,
        "sigma_targets": sigma_targets,
        "beta": BETA,
        "Ltau": INPUT_DIM,
    }, f"{pre_out_dir}/pretrain_eval.pt")
    print(f"Pretraining evaluation saved to {pre_out_dir}/pretrain_eval.pt")

    # --- Plot pretraining results ---
    print("\n--- Generating pretraining plots ---")
    plot_pretrain_eval(f"{pre_out_dir}/pretrain_eval.pt")
    plot_loss_curves(f"{pre_out_dir}/losses", tag="pretrain",
                     save_path=f"{pre_out_dir}/plots/loss_curves_pretrain.pdf")

    # ------------------------------------------------------------------
    # STAGE 2: Fine-tuning on target dataset (optional)
    # ------------------------------------------------------------------
    if not DO_FINETUNE:
        print("\nSkipping fine-tuning (DO_FINETUNE=False).")
        return

    print("\n" + "=" * 80)
    print(f"STAGE 2: FINE-TUNING ON {SPECTRAL_TYPE.upper()} DATA")
    print("=" * 80 + "\n")

    tag = "finetune"

    # Data path
    data_path = os.path.join(
        MAIN_PATH, "Data", "datasets", 
        f"half-filled-{SPECTRAL_TYPE}", {INPUT_ID}, 
        f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
    )
    
    ft_out_dir = os.path.join(
        MAIN_PATH, "VAE_Library",
        f"finetune_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_s{NOISE_S:.0e}_xi{NOISE_XI}-{sID}"
    )
    MakeOutPath(ft_out_dir)

    # Load dataset
    ft_dataset = GreenFunctionDataset(file_path=data_path)
    std = STD(ft_dataset)

    train_dataset_ft, val_dataset_ft, test_dataset_ft = TrainTestSplit(
        ft_dataset, train_ratio=0.8, val_ratio=0.1
    )
    train_loader_ft, val_loader_ft, test_loader_ft = LoadData(
        train_dataset_ft, val_dataset_ft, test_dataset_ft,
        batch_size=BATCH_SIZE, shuffle=True
    )
    PrintDataSize(ft_dataset, train_dataset_ft, val_dataset_ft, test_dataset_ft, train_loader_ft)

    # Covariance for chi2 loss (reuse C from pretraining, or reload from finetune data)
    ft_cov_path = data_path  # Use the finetune Gbins for covariance
    print(f"Loading covariance for fine-tuning from: {ft_cov_path}")
    C_ft = load_covariance_from_dqmc(ft_cov_path)

    ft_chi2_fn = ChiSquaredLoss(C_ft).to(DEVICE)
    ft_smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    ft_positivity_fn = SpectralPositivityLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)

    # Optimizer (fresh)
    ft_optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-3)
    ft_scheduler = CosineAnnealingWarmRestarts(ft_optimizer, T_0=FINETUNE_T0, T_mult=FINETUNE_T_MULT)

    # Fine-tune
    print(f"\nLoss weights: lambda_chi2={FINETUNE_LAMBDA_CHI2}, lambda_smooth={FINETUNE_LAMBDA_SMOOTH}, "
          f"lambda_pos={FINETUNE_LAMBDA_POS}, alpha_kl={FINETUNE_ALPHA_KL}, "
          f"eta0={FINETUNE_ETA0}, eta2={FINETUNE_ETA2}")
    print(f"Scheduler: CosineAnnealingWarmRestarts(T_0={FINETUNE_T0}, T_mult={FINETUNE_T_MULT})")

    best_ft_val, best_ft_epoch = train_finetune(
        model=model,
        optimizer=ft_optimizer,
        scheduler=ft_scheduler,
        num_epochs=FINETUNE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=train_loader_ft,
        val_loader=val_loader_ft,
        std=std,
        chi2_fn=ft_chi2_fn,
        smoothness_fn=ft_smoothness_fn,
        positivity_fn=ft_positivity_fn,
        lambda_chi2=FINETUNE_LAMBDA_CHI2,
        lambda_smooth=FINETUNE_LAMBDA_SMOOTH,
        lambda_pos=FINETUNE_LAMBDA_POS,
        alpha_kl=FINETUNE_ALPHA_KL,
        eta0=FINETUNE_ETA0,
        eta2=FINETUNE_ETA2,
        device=DEVICE,
        out_dir=ft_out_dir,
        tag=tag,
        patience=FINETUNE_PATIENCE,
    )

    print(f"\nFine-tuning done. Best val loss: {best_ft_val:.4e} at epoch {best_ft_epoch+1}")

    # Load best fine-tuned model
    model.load_state_dict(torch.load(f"{ft_out_dir}/model/best_model_{tag}.pth", weights_only=True))

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\n--- Test evaluation ---")
    model.eval()
    test_loss = 0.0
    G_input_all, G_recon_all = [], []
    poles_all, residues_all = [], []

    with torch.no_grad():
        for batch in test_loader_ft:
            B = batch.shape[0]
            batch = batch.view(B, INPUT_DIM).to(DEVICE)

            mu_vae, logvar_vae, z, poles, residues, G_recon = model(batch)

            loss, _, _, _, _, _, _ = finetune_total_loss(
                G_recon, batch,
                mu_vae, logvar_vae,
                poles, residues,
                std,
                ft_chi2_fn, ft_smoothness_fn, ft_positivity_fn,
                FINETUNE_LAMBDA_CHI2, FINETUNE_LAMBDA_SMOOTH, FINETUNE_LAMBDA_POS,
                FINETUNE_ALPHA_KL, FINETUNE_ETA0, FINETUNE_ETA2,
            )
            test_loss += loss.item() * B

            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())

    test_loss /= len(test_loader_ft.dataset)
    print(f"Test Loss: {test_loss:.4e}")

    G_input_all = torch.cat(G_input_all, dim=0)
    G_recon_all = torch.cat(G_recon_all, dim=0)
    poles_all = torch.cat(poles_all, dim=0)
    residues_all = torch.cat(residues_all, dim=0)

    torch.save({
        "inputs": G_input_all,
        "recon": G_recon_all,
        "poles": poles_all,
        "residues": residues_all,
        "inputs_avg": G_input_all.mean(dim=0),
        "recon_avg": G_recon_all.mean(dim=0),
        "poles_avg": poles_all.mean(dim=0),
        "residues_avg": residues_all.mean(dim=0),
        "beta": BETA,
        "Ltau": INPUT_DIM,
        "spectral_input_path": SPECTRAL_INPUT_PATH,
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved to {ft_out_dir}/summary.pt")
    print(f"Poles mean: {poles_all.mean(dim=0)}")
    print(f"Residues mean: {residues_all.mean(dim=0)}")

    # --- Plot fine-tuning results ---
    print("\n--- Generating fine-tuning plots ---")
    plot_finetune_eval(f"{ft_out_dir}/summary.pt")
    plot_loss_curves(f"{ft_out_dir}/losses", tag=tag,
                     save_path=f"{ft_out_dir}/plots/loss_curves_{tag}.pdf")


if __name__ == "__main__":
    main()
