"""
Zero-bias, symmetrically-initialized Variational Autoencoder (VAE).
All Conv1d and Linear layers have bias=False, and weights are initialized
using Xavier (Glorot) uniform initialization to ensure zero-centered symmetry.

This helps preserve balanced activations and avoids systematic bias drift.
"""

import torch
import torch.nn.functional as F
from torch import nn
from green_reconstruction2 import PoleToGaussLegendreGreens # Ben's # type: ignore
# from Green_reconstruction import GaussLegendreQuadrature    # George's # type: ignore


class VariationalAutoEncoder2(nn.Module):
    
    def __init__(self, 
                 input_dim,
                 hidden_dim1, hidden_dim2,
                 latent_dim,
                 hidden_dim3, hidden_dim4,
                 num_poles, 
                 beta, N_nodes=256):
        
        super(VariationalAutoEncoder2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1; self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
        self.hidden_dim3 = hidden_dim3; self.hidden_dim4 = hidden_dim4
        self.num_poles = num_poles
        self.beta = beta; self.dtau = beta / (input_dim)
        
        # Encoder: Conv layers (no bias)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=2, padding=4, dilation=1, padding_mode='reflect', bias=False)      # halves length
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=2, padding=4, dilation=1, padding_mode='reflect', bias=False)     # halves length again
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=2, padding=4, dilation=1, padding_mode='reflect', bias=False)    # halves length again
        
        # Encoder: fully connected
        self.dense1 = nn.Linear(128 * (input_dim // 8), hidden_dim1, bias=False) # input dimension is divided by 2^3 = 8
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        
        # Latent space
        self.hidden_mu = nn.Linear(hidden_dim2, latent_dim, bias=False)
        self.hidden_logvar = nn.Linear(hidden_dim2, latent_dim, bias=False)
        
        # Decoder
        self.z_projector = nn.Linear(latent_dim, hidden_dim3, bias=False)
        self.second_projector = nn.Linear(hidden_dim3, hidden_dim4, bias=False)
        
        # Output layers
        self.poles_re_generator = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.poles_im_generator = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.residues_re_generator = nn.Linear(hidden_dim4, num_poles, bias=True)
        self.residues_im_generator = nn.Linear(hidden_dim4, num_poles, bias=True)
        
        # Physics-based reconstruction
        self.greens_reconstructor = PoleToGaussLegendreGreens(beta=beta, dtau=self.dtau, N_nodes=N_nodes)
        
        # Initialize weights symmetrically (Xavier uniform)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
    
    def encode(self, x):
        B = x.shape[0]
        h = F.leaky_relu(self.conv1(x.unsqueeze(1)))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        
        h = h.view(B, -1)
        h = F.relu(self.dense1(h))
        h = F.relu(self.dense2(h))
        
        mu = self.hidden_mu(h)
        logvar = self.hidden_logvar(h)
        
        return mu, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_poles_residues(self, z):
        B = z.shape[0]
        
        h = F.relu(self.z_projector(z))
        h = F.relu(self.second_projector(h))
        
        epsilon = self.poles_re_generator(h)
        gamma = self.poles_im_generator(h)
        a = self.residues_re_generator(h)
        b = self.residues_im_generator(h)
        
        # Structure constraints
        gamma = - torch.abs(gamma)  # ensure negativity
        a = torch.abs(a) # ensure positivity
        
        a_sum = torch.sum(a, dim=1, keepdim=True)
        a = a / a_sum # Normalize real part of the residues
        
        b_sum = torch.sum(b, dim=1, keepdim=True)
        b = b - (b_sum / self.num_poles) # Make sure the imaginary part of residues adds up to zero
        
        poles = torch.complex(epsilon, gamma)
        residues = torch.complex(a, b)
        
        return poles, residues
    
    def forward(self, x, deterministic=False):

        mu, logvar = self.encode(x)
        z = mu if deterministic else self.sample(mu, logvar)
        poles, residues = self.decode_poles_residues(z)
        Gtau_reconstructed = self.greens_reconstructor(poles, residues)

        return mu, logvar, z, poles, residues, Gtau_reconstructed


if __name__ == "__main__":
    input_dim = 200
    hidden_dim1, hidden_dim2 = 128, 64
    latent_dim = 20
    hidden_dim3, hidden_dim4 = 64, 128
    num_poles = 5
    beta = 10.0
    N_nodes = 256

    vae_model = VariationalAutoEncoder2(input_dim, hidden_dim1, hidden_dim2,
                                        latent_dim, hidden_dim3, hidden_dim4,
                                        num_poles, beta, N_nodes)

    batch_size = 32
    random_input = torch.randn(batch_size, input_dim)

    mu, logvar, z, poles, residues, Gtau_reconstructed = vae_model(random_input)

    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print("Latent z shape:", z.shape)
    print("Poles shape:", poles.shape)
    print("Residues shape:", residues.shape)
    print("Reconstructed G(tau) shape:", Gtau_reconstructed.shape)