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
                 beta, N_nodes=256, latent_dim=None,
                 ph_symmetric=False):
        # ph_symmetric: when True, num_poles is the *free* pole count emitted
        # by the network; decode_poles_residues concatenates each free pole
        # (eps, gamma, a, b) with its PH partner (-eps, gamma, a, -b), so the
        # spectral function is even and PoleToGaussLegendreGreens sees
        # 2*num_poles effective poles. The hidden-layer scaling, latent_dim,
        # and per-head output widths all stay tied to the *free* count.

        super(VariationalAutoEncoder2, self).__init__()

        # Hidden-layer capacity is always tied to num_poles via the original
        # scale `base = 4 * num_poles - 2`; only the latent bottleneck
        # (mu/logvar/z) can be decoupled. Pass latent_dim=None to keep the
        # legacy behaviour of latent_dim == base.
        base        = 4 * num_poles - 2
        latent_dim  = base if latent_dim is None else int(latent_dim)
        hidden_dim1 = 64 * base
        hidden_dim2 = 1  * base
        hidden_dim3 = 2  * base
        hidden_dim4 = 4  * base

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1; self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
        self.hidden_dim3 = hidden_dim3; self.hidden_dim4 = hidden_dim4
        self.ph_symmetric = bool(ph_symmetric)
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
        gamma = torch.abs(gamma)                                     # γ ≥ 0

        if self.ph_symmetric:
            # PH partner of (eps, gamma, a, b) is (-eps, gamma, a, -b). Concat
            # gives 2*num_poles effective poles with A(omega) = A(-omega) and
            # zero-sum on the imaginary residues for free, so we drop the
            # explicit `b - mean(b)` constraint and renormalize `a` so the
            # concatenated set sums to 1 (free poles sum to 1/2 each).
            a       = torch.abs(a)                                   # a ≥ 0
            a       = a / (2.0 * a.sum(dim=1, keepdim=True) + 1e-12)
            epsilon = torch.cat([epsilon, -epsilon], dim=1)
            gamma   = torch.cat([gamma,    gamma  ], dim=1)
            a       = torch.cat([a,        a      ], dim=1)
            b       = torch.cat([b,       -b      ], dim=1)
        else:
            a = torch.abs(a)                                         # a ≥ 0
            a = a / (a.sum(dim=1, keepdim=True) + 1e-12)             # sum(a) = 1
            b = b - (b.sum(dim=1, keepdim=True) / self.num_poles)    # sum(b) = 0

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

    from pretrain.pretrain_losses import spectral_from_poles  # type: ignore

    input_dim = 200
    num_poles = 4
    beta      = 10.0
    N_nodes   = 256
    B         = 32

    def _check(model, x, *, ph):
        mu, logvar, z, poles, residues, G_recon = model(x)

        a       = residues.real
        b       = residues.imag
        epsilon = poles.real
        gamma   = -poles.imag

        P_eff = 2 * num_poles if ph else num_poles
        assert poles.shape    == (B, P_eff), f"poles shape {poles.shape} != {(B, P_eff)}"
        assert residues.shape == (B, P_eff), f"residues shape {residues.shape} != {(B, P_eff)}"

        assert (gamma >= 0).all(), "γ ≥ 0 violated"
        assert (a >= 0).all(),     "a ≥ 0 violated"
        assert torch.allclose(a.sum(dim=1), torch.ones(B),  atol=1e-5), "sum(a) ≠ 1"
        assert torch.allclose(b.sum(dim=1), torch.zeros(B), atol=1e-5), "sum(b) ≠ 0"

        if ph:
            # Spectral evenness on a symmetric grid.
            omegas = torch.linspace(-5.0, 5.0, 401)
            A      = spectral_from_poles(poles, residues, omegas)
            A_flip = torch.flip(A, dims=[1])
            assert torch.allclose(A, A_flip, atol=1e-9), \
                f"A(ω) not even: max|A(ω)-A(-ω)| = {(A - A_flip).abs().max().item():.2e}"

            # Third tail-coefficient sum_p [b(eps²-γ²) - 2 a eps γ] = 0,
            # equivalently the third spectral moment vanishes.
            third = (b * (epsilon**2 - gamma**2) - 2 * a * epsilon * gamma).sum(dim=1)
            assert torch.allclose(third, torch.zeros(B), atol=1e-5), \
                f"third tail coeff ≠ 0: max|·| = {third.abs().max().item():.2e}"

    # PH off (legacy)
    torch.manual_seed(0)
    model = VariationalAutoEncoder2(input_dim, num_poles, beta)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PH=off  |  num_poles={num_poles}  effective={num_poles}")
    print(f"Trainable parameters: {n_params:,}")
    x = torch.randn(B, input_dim)
    _check(model, x, ph=False)
    print("  legacy constraints pass.")

    # PH on
    torch.manual_seed(0)
    model_ph = VariationalAutoEncoder2(input_dim, num_poles, beta, ph_symmetric=True)
    n_params_ph = sum(p.numel() for p in model_ph.parameters() if p.requires_grad)
    print(f"\nPH=on   |  num_poles={num_poles}  effective={2 * num_poles}")
    print(f"Trainable parameters: {n_params_ph:,}")
    _check(model_ph, x, ph=True)
    print("  PH symmetric: A(ω)=A(-ω), sum(a)=1, sum(b)=0, third tail coeff=0.")

    print("\nAll physical constraints satisfied.")