"""
VariationalAutoEncoder2 — Step 2 variant (InstanceNorm1d + tanh).

Drop-in replacement for model2.py with two targeted changes:
  1. InstanceNorm1d (affine=True) after every Conv1d layer in the encoder.
     Makes the conv stack scale-invariant: G(tau) curves from different spectral
     functions have different overall magnitudes; IN normalises each sample
     independently so the FC layers see a consistent representation.
  2. All activations replaced by leaky_relu

Everything else is identical to model2.py: same conv/FC architecture, same
decoder FC layers (z_projector → second_projector → 4 output heads), same
physical constraints in decode_poles_residues, same forward() API.

Usage: swap `from model2 import ...` → `from model2_tanh import ...`
"""

import torch
import torch.nn.functional as F
from torch import nn
from green_reconstruction2 import PoleToGaussLegendreGreens  # type: ignore


class VariationalAutoEncoder2(nn.Module):

    def __init__(self,
                 input_dim,
                 num_poles,
                 beta, N_nodes=256):

        super(VariationalAutoEncoder2, self).__init__()

        latent_dim  = 4 * num_poles - 2
        hidden_dim1 = 64 * latent_dim
        hidden_dim2 = 1  * latent_dim
        hidden_dim3 = 2  * latent_dim
        hidden_dim4 = 4  * latent_dim

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1; self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
        self.hidden_dim3 = hidden_dim3; self.hidden_dim4 = hidden_dim4
        self.num_poles = num_poles
        self.beta = beta; self.dtau = beta / input_dim

        # ------------------------------------------------------------------ #
        # Encoder: Conv1d (no bias) + InstanceNorm1d + tanh
        # ------------------------------------------------------------------ #
        self.conv1 = nn.Conv1d(1,  16,  kernel_size=9, stride=2, padding=4,
                               padding_mode='reflect', bias=False)
        self.norm1 = nn.InstanceNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32,  kernel_size=9, stride=2, padding=4,
                               padding_mode='reflect', bias=False)
        self.norm2 = nn.InstanceNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4,
                               padding_mode='reflect', bias=False)
        self.norm3 = nn.InstanceNorm1d(64)

        # Encoder: fully connected (no bias) + leaky_relu
        self.dense1 = nn.Linear(64 * (input_dim // 8), hidden_dim1, bias=False)
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)

        # Latent space (no activation on mu/logvar — raw outputs)
        self.hidden_mu     = nn.Linear(hidden_dim2, latent_dim, bias=False)
        self.hidden_logvar = nn.Linear(hidden_dim2, latent_dim, bias=False)

        # ------------------------------------------------------------------ #
        # Decoder: FC layers (no bias) + tanh, then 4 physical output heads
        # ------------------------------------------------------------------ #
        self.z_projector        = nn.Linear(latent_dim,  hidden_dim3, bias=False)
        self.second_projector   = nn.Linear(hidden_dim3, hidden_dim4, bias=False)

        # Output heads (bias=True for poles, False for residues 
        # zero-initialised for poles)
        self.poles_re_generator     = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.poles_im_generator     = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.residues_re_generator  = nn.Linear(hidden_dim4, num_poles, bias=False)
        self.residues_im_generator  = nn.Linear(hidden_dim4, num_poles, bias=False)

        # Physics-based Green's function reconstructor
        self.greens_reconstructor = PoleToGaussLegendreGreens(
            beta=beta, dtau=self.dtau, N_nodes=N_nodes
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Conv layers: xavier_uniform_ (matches Ben's make_conv_stack)
        # Linear layers: left at PyTorch default kaiming_uniform_ (matches Ben's VAE1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
        # Zero-init pole biases so poles start near the origin
        nn.init.zeros_(self.poles_re_generator.bias)
        nn.init.zeros_(self.poles_im_generator.bias)

    # ---------------------------------------------------------------------- #
    # Encoder
    # ---------------------------------------------------------------------- #

    def encode(self, x):
        B = x.shape[0]

        # Conv stack: InstanceNorm1d normalises each sample, leaky_relu bounds activations
        h = F.leaky_relu(self.norm1(self.conv1(x.unsqueeze(1))))
        h = F.leaky_relu(self.norm2(self.conv2(h)))
        h = F.leaky_relu(self.norm3(self.conv3(h)))

        h = h.view(B, -1)
        h = F.leaky_relu(self.dense1(h))
        h = F.leaky_relu(self.dense2(h))

        mu     = self.hidden_mu(h)      # raw, no activation
        logvar = self.hidden_logvar(h)  # raw, no activation

        return mu, logvar

    # ---------------------------------------------------------------------- #
    # Reparameterization
    # ---------------------------------------------------------------------- #

    def sample(self, mu, logvar):
        logvar  = torch.clamp(logvar, min=-25, max=25)  # numerical stability
        std     = torch.exp(0.5 * logvar)
        eps     = torch.randn_like(std)
        return mu + eps * std

    # ---------------------------------------------------------------------- #
    # Decoder
    # ---------------------------------------------------------------------- #

    def decode_poles_residues(self, z):
        # Decoder FC stack: tanh (matching encoder)
        h = F.leaky_relu(self.z_projector(z))
        h = F.leaky_relu(self.second_projector(h))

        epsilon = self.poles_re_generator(h)
        gamma   = self.poles_im_generator(h)
        a       = self.residues_re_generator(h)
        b       = self.residues_im_generator(h)

        # Physical constraints (applied to deterministic decoder output, same as model2)
        gamma = torch.abs(gamma)                                     # γ ≤ 0
        a     = torch.abs(a)                                         # a ≥ 0
        a     = a / (a.sum(dim=1, keepdim=True) + 1e-12)             # sum(a) = 1
        b     = b - (b.sum(dim=1, keepdim=True) / self.num_poles)    # sum(b) = 0

        poles    = torch.complex(epsilon, -gamma)
        residues = torch.complex(a, b)

        return poles, residues

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.sample(mu, logvar)
        poles, residues = self.decode_poles_residues(z)
        Gtau_reconstructed = self.greens_reconstructor(poles, residues)
        return mu, logvar, z, poles, residues, Gtau_reconstructed


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    input_dim = 200
    num_poles = 4
    beta      = 10.0
    N_nodes   = 256
    B         = 32

    model = VariationalAutoEncoder2(input_dim, num_poles, beta)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VariationalAutoEncoder3  |  P={num_poles}")
    print(f"Trainable parameters: {n_params:,}")

    x = torch.randn(B, input_dim)
    mu, logvar, z, poles, residues, G_recon = model(x)

    print(f"\nmu shape      : {mu.shape}")
    print(f"logvar shape  : {logvar.shape}")
    print(f"z shape       : {z.shape}")
    print(f"poles shape   : {poles.shape}")
    print(f"residues shape: {residues.shape}")
    print(f"G_recon shape : {G_recon.shape}")

    a     = residues.real
    b     = residues.imag
    gamma = -poles.imag

    assert (gamma >= 0).all(),  "γ constraint violated"
    assert (a >= 0).all(),      "a constraint violated"
    assert torch.allclose(a.sum(dim=1), torch.ones(B),  atol=1e-5), "a normalisation violated"
    assert torch.allclose(b.sum(dim=1), torch.zeros(B), atol=1e-5), "b zero-sum violated"

    print("\nAll physical constraints satisfied.")