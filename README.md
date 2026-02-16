# SPE-AC-VAE

Analytic continuation of imaginary-time Green's functions to real-frequency spectral functions using a Variational Autoencoder with a two-stage training strategy.

## Overview

Extracting real-frequency spectral functions A(ω) from imaginary-time Green's functions G(τ) is a notoriously ill-posed inverse problem in condensed matter physics. This project uses a physics-informed VAE that encodes G(τ) into a latent space and decodes it into complex poles and residues, from which both G(τ) and A(ω) can be reconstructed analytically.

## Model Architecture

- **Encoder**: Convolutional layers (Conv1d) followed by fully connected layers, mapping G(τ) → latent mean and variance
- **Latent space**: L-dimensional (L = 4 × NUM_POLES − 2)
- **Decoder**: Fully connected layers producing complex poles and residues
- **Physics layer**: Reconstructs G(τ) from poles/residues via Gauss-Legendre quadrature

The spectral function is recovered as:

$$A(\omega) = -\frac{1}{\pi} \sum_p \text{Im}\left(\frac{r_p}{\omega - p_p}\right)$$

with constraints enforcing normalization (∫A(ω)dω = 1) and positivity (A(ω) ≥ 0).

## Two-Stage Training

### Stage 1 — Pretraining on synthetic data
Train the VAE on synthetic Gaussian spectral functions with known ground truth. This teaches the model the physics mapping G(τ) → poles/residues → A(ω) in a supervised setting.

### Stage 2 — Fine-tuning on DQMC data
Transfer to real Determinantal Quantum Monte Carlo data where no ground truth spectral function is available. Uses covariance-weighted χ² loss for reconstruction, plus spectral smoothness and positivity regularizers.

## Project Structure

```
SPE_AC_VAE/
├── model2.py                  # VAE model definition
├── data_process.py            # Dataset loader
├── utils.py                   # Data utilities and I/O
├── pretrain/
│   ├── pretrain_losses.py     # Loss functions (χ², spectral MSE, smoothness, positivity)
│   ├── synthetic_data.py      # Synthetic Gaussian data generation
│   ├── train_pretrain.py      # Stage 1 training loop
│   ├── train_finetune.py      # Stage 2 training loop
│   ├── run_pretrain_pipeline.py  # Main pipeline (both stages)
│   ├── run_finetune.py        # Standalone fine-tuning script
│   ├── generate_data.py       # CLI for synthetic data generation
│   └── plot_results.py        # Visualization
└── out/                       # Trained models and results
```

## Usage

Run the full two-stage pipeline:

```bash
python pretrain/run_pretrain_pipeline.py
```

Or fine-tune a pretrained model separately:

```bash
python pretrain/run_finetune.py
```

## Example/Default Parameters

| Parameter | Value |
|-----------|-------|
| NUM_POLES | 4 |
| LATENT_DIM | 14 |
| β (inverse temperature) | 10.0 |
| Δτ | 0.05 |
| Input dim (L_τ) | 200 |
| Pretraining epochs | 100 |
| Fine-tuning epochs | 200 |

## Dependencies

- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm
