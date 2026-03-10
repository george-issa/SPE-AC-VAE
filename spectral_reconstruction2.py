"""
This is a class for the map taking poles and residues and giving the corresponding 
spectral function on an omega grid.
"""

import torch
import torch.nn as nn
from scipy import special
import numpy as np

class SpectralReconstructor(nn.Module):
    
    def __init__(self, Nw, wmin=-8, wmax=8, dtype=torch.float32):
        
        # Call nn.Module default init function
        super().__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map float type to complex type
        ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
        
        # Initialize all variables
        
        # Number of omega points on the grid for the spectral function
        self.Nw = Nw
        
        # min and max for the omega grid. Default is -8 and 8.
        self.register_buffer("wmin", torch.tensor(wmin, dtype=dtype))
        self.register_buffer("wmax", torch.tensor(wmax, dtype=dtype))
        
        # Initialize an omega array for all calls
        self.ws = torch.linspace(wmin, wmax, Nw, dtype=dtype, device=device)
        
    def forward(self, poles, residues):
        
        # Ensure inputs have the right shape
        if poles.ndim != 2 or residues.ndim !=2:
            raise TypeError(
                f"Poles and Residues shape is {poles.shape} and {residues.shape} "
                "but expected (batch_size, latent_dim)"
            )

        B, L = poles.shape
        
        poles = poles.unsqueeze(1)
        residues = residues.unsqueeze(1)
        
        # Create the omega array and expand it to (B, Nw, L)
        ws = torch.linspace(self.wmin, self.wmax, self.Nw).view(1, self.Nw, 1)

        Aw = torch.sum(torch.imag(residues / (ws - poles)), dim=2)
        
        return - (1.0 / torch.pi) * Aw
        
if __name__ == "__main__":
    
    sr = SpectralReconstructor(Nw=1000, wmin=-8, wmax=8)
    
    poles = torch.randn((32, 5), dtype=torch.complex64)
    residues = torch.randn((32, 5), dtype=torch.complex64)
    
    Aw = sr(poles, residues)
    
    print(f"Aw shape is {Aw.shape}")