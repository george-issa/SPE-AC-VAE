"""
DecoderOnly — no-encoder variant of model2_leaky's decoder pipeline.

Drops the encoder and the latent space entirely. The decoder produces a
single set of poles+residues from a learned bias vector — equivalent to
the original VAE2's (z_projector -> second_projector -> 4 heads) chain
with z replaced by a constant nn.Parameter. All B samples in a batch get
the same poles/residues; the per-sample G_i input is ignored.

Forward signature matches VariationalAutoEncoder2 for drop-in compatibility
with train_finetune.py: returns (mu, logvar, z, poles, residues, G_recon)
where mu = logvar = z = zeros. KL(N(0,1) || N(0,1)) = 0, so the
KLDivergenceLoss in finetune_total_loss contributes nothing regardless of
ALPHA_KL.

Use case (per encoder_audit.py findings on LW + P=10): the trained VAE's
encoder degenerates into a constant offset and the decoder is reproducible
from a fixed z. This model strips that to its irreducible form so we can
test whether *training* needed the encoder at all (the audit only proved
inference doesn't).
"""

import torch
import torch.nn.functional as F
from torch import nn
from green_reconstruction2 import PoleToGaussLegendreGreens  # type: ignore


class DecoderOnly(nn.Module):

    def __init__(self,
                 input_dim,
                 num_poles,
                 beta, N_nodes=256,
                 ph_symmetric=False):
        super().__init__()

        # Hidden-layer scaling matches model2_leaky exactly so a side-by-side
        # comparison is fair: same per-pole capacity in second_projector and
        # the 4 output heads. The only thing missing relative to VAE2 is the
        # encoder + the z_projector layer (both excised).
        base        = 4 * num_poles - 2
        hidden_dim3 = 2 * base
        hidden_dim4 = 4 * base

        self.input_dim    = input_dim
        self.num_poles    = num_poles
        self.ph_symmetric = bool(ph_symmetric)
        self.beta         = beta
        self.dtau         = beta / input_dim
        self.hidden_dim3  = hidden_dim3
        self.hidden_dim4  = hidden_dim4

        # Replaces VAE2's z_projector(z). The original z_projector was
        # bias=False, so feeding any z just returns W*z; with z constant
        # across samples, this collapses into a single learnable vector.
        # Init non-zero: with the pole biases zero-initialised (per VAE2)
        # and second_projector / residue heads bias=False, an all-zero
        # decoder_bias propagates zeros through the whole pipeline and the
        # `a / sum(a)` normalisation divides by ~0. The encoder used to
        # break this symmetry by emitting non-zero z at random init; here
        # we have to do it explicitly. std=0.5 lands in the natural scale
        # of leaky_relu activations after one Linear layer.
        self.decoder_bias = nn.Parameter(torch.empty(hidden_dim3))
        nn.init.normal_(self.decoder_bias, mean=0.0, std=0.5)

        self.second_projector       = nn.Linear(hidden_dim3, hidden_dim4, bias=False)
        self.poles_re_generator     = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.poles_im_generator     = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.residues_re_generator  = nn.Linear(hidden_dim4, num_poles, bias=False)
        self.residues_im_generator  = nn.Linear(hidden_dim4, num_poles, bias=False)

        self.greens_reconstructor = PoleToGaussLegendreGreens(
            beta=beta, dtau=self.dtau, N_nodes=N_nodes,
        )

        nn.init.zeros_(self.poles_re_generator.bias)
        nn.init.zeros_(self.poles_im_generator.bias)

    # ---------------------------------------------------------------------- #
    # Decoder
    # ---------------------------------------------------------------------- #

    def decode_poles_residues(self, batch_size):
        h = self.decoder_bias.unsqueeze(0).expand(batch_size, -1)
        h = F.leaky_relu(h)
        h = F.leaky_relu(self.second_projector(h))

        epsilon = self.poles_re_generator(h)
        gamma   = self.poles_im_generator(h)
        a       = self.residues_re_generator(h)
        b       = self.residues_im_generator(h)

        gamma = torch.abs(gamma)

        if self.ph_symmetric:
            a       = torch.abs(a)
            a       = a / (2.0 * a.sum(dim=1, keepdim=True) + 1e-12)
            epsilon = torch.cat([epsilon, -epsilon], dim=1)
            gamma   = torch.cat([gamma,    gamma  ], dim=1)
            a       = torch.cat([a,        a      ], dim=1)
            b       = torch.cat([b,       -b      ], dim=1)
        else:
            a = torch.abs(a)
            a = a / (a.sum(dim=1, keepdim=True) + 1e-12)
            b = b - (b.sum(dim=1, keepdim=True) / self.num_poles)

        poles    = torch.complex(epsilon, -gamma)
        residues = torch.complex(a, b)
        return poles, residues

    # ---------------------------------------------------------------------- #
    # Forward (compatibility shim with VariationalAutoEncoder2)
    # ---------------------------------------------------------------------- #

    def forward(self, x, deterministic=False):
        # `deterministic` is accepted but ignored — the model is already
        # deterministic. mu/logvar/z are all zeros, so the train_finetune
        # KL term evaluates to 0 and ALPHA_KL is a no-op.
        B = x.shape[0]
        mu     = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        logvar = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        z      = mu
        poles, residues = self.decode_poles_residues(B)
        G_recon = self.greens_reconstructor(poles, residues)
        return mu, logvar, z, poles, residues, G_recon

    def encode(self, x):
        # Compatibility shim for tooling that calls encode() (e.g.
        # encoder_audit.py). There is no encoder; return the delta-posterior
        # parameters at zero. AU on a DecoderOnly run will always be 0/1.
        B = x.shape[0]
        mu     = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        logvar = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        return mu, logvar

    def sample(self, mu, logvar):
        # Compatibility shim — z is always zero.
        return torch.zeros_like(mu)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    from pretrain.pretrain_losses import spectral_from_poles  # type: ignore

    input_dim = 100
    num_poles = 10
    beta      = 10.0
    B         = 5

    for ph in (False, True):
        torch.manual_seed(0)
        model = DecoderOnly(input_dim=input_dim, num_poles=num_poles,
                            beta=beta, ph_symmetric=ph)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        P_eff = 2 * num_poles if ph else num_poles
        print(f"PH={'on' if ph else 'off'}  num_poles={num_poles}  "
              f"effective={P_eff}  trainable params={n_params:,}")

        x = torch.randn(B, input_dim)
        mu, logvar, z, poles, residues, G_recon = model(x)

        assert poles.shape    == (B, P_eff), f"poles shape {poles.shape}"
        assert residues.shape == (B, P_eff), f"residues shape {residues.shape}"
        assert G_recon.shape  == (B, input_dim), f"G_recon shape {G_recon.shape}"

        a       = residues.real
        b       = residues.imag
        epsilon = poles.real
        gamma   = -poles.imag

        assert (gamma >= 0).all(), "gamma >= 0 violated"
        assert (a >= 0).all(),     "a >= 0 violated"
        assert torch.allclose(a.sum(dim=1), torch.ones(B), atol=1e-5), "sum(a) != 1"

        # All B samples produce identical outputs (decoder ignores x)
        assert torch.allclose(poles[0], poles[-1], atol=1e-9), \
            "decoder output should be identical across batch"
        assert torch.allclose(residues[0], residues[-1], atol=1e-9), \
            "decoder output should be identical across batch"

        if ph:
            omegas = torch.linspace(-5.0, 5.0, 401)
            A      = spectral_from_poles(poles, residues, omegas)
            assert torch.allclose(A, A.flip(dims=[1]), atol=1e-9), \
                f"A(w) != A(-w) under PH-on: max diff {(A - A.flip(dims=[1])).abs().max().item():.2e}"
        else:
            assert torch.allclose(b.sum(dim=1), torch.zeros(B), atol=1e-5), "sum(b) != 0"

        # KL = 0 by shim
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        assert kl.abs().item() < 1e-9, f"KL should be 0, got {kl.item()}"

        print(f"  shapes ok | constraints ok | identical across batch | KL = {kl.item():.2e}")

    print("\nDecoderOnly smoke test passed.")
