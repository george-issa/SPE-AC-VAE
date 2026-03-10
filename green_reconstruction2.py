import torch
import torch.nn as nn
from scipy import special
import numpy as np

class PoleToGaussLegendreGreens(nn.Module):

    def __init__(self, beta, dtau, N_nodes = 256, N_iwn = 256, dtype = torch.float32):

        # Call nn.Module default _init_() method
        super().__init__()
        
        # Map float type to complex type
        ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
        
        # Tau grid
        Ltau = int(np.round(beta/dtau))
        taus = torch.linspace(0.0, beta-dtau, Ltau, dtype=dtype)
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype)) # (,)
        self.register_buffer("dtau", torch.tensor(dtau, dtype=dtype)) # (,)
        self.register_buffer("Ltau", torch.tensor(Ltau, dtype=torch.int)) # (,)
        self.register_buffer("taus", taus)  # (Ltau,)
        
        # Gauss–Legendre nodes + weights
        nodes, weights = special.roots_legendre(N_nodes)
        nodes = torch.tensor(nodes, dtype=dtype)       # (N_nodes,)
        weights = torch.tensor(weights, dtype=dtype)   # (N_nodes,)
        self.register_buffer("nodes", nodes)
        self.register_buffer("weights", weights)

        # Precompute Phi(t) = tan(pi/2 * t)
        phi = torch.tan(torch.pi/2 * nodes)  # (N_nodes,)
        self.register_buffer("phi", phi)
        
        # Matsubara frequencies for computing G(tau=0)
        iwn = 1j * (2 * np.arange(N_iwn) + 1) * np.pi / beta
        iwn = torch.tensor(iwn, dtype = ctype)
        self.register_buffer("iwn", iwn)
        coefs = torch.tensor([
            -0.5,
            -beta/4,
            -7*special.zeta(3.0)*beta**2/(4*np.pi**3),
            beta**3/48,
            31*special.zeta(5.0)*beta**4/(16*np.pi**5)
        ], dtype=dtype)
        self.register_buffer("coefs", coefs)

    def forward(self, poles, residues):
            
         # Ensure inputs are at least 2D: (batch, num_poles)
        if poles.ndim == 1:
            poles = poles.unsqueeze(0)       # (1, num_poles)
            residues = residues.unsqueeze(0) # (1, num_poles)
            
        epsilon = torch.real(poles) # (batch, num_poles)
        gamma = -torch.imag(poles) # (batch, num_poles)
        a = torch.real(residues) # (batch, num_poles)
        b = torch.imag(residues) # (batch, num_poles)
        
        # COMPUTE G(tau) FOR 0 < tau < beta

        # omegas: (batch, num_poles, N_nodes)
        omegas = epsilon[..., None] + gamma[..., None] * self.phi[None, None, :]

        # numerator: (batch, num_poles, N_nodes)
        numerator = 0.5 * (a[..., None] - b[..., None] * self.phi[None, None, :])

        # taus: (Ltau,)
        taus = self.taus
        
        # calculate arguments of exponents
        arg1 = taus[None, None, :, None] * omegas[..., None, :]
        arg2 = (taus[None, None, :, None] - self.beta) * omegas[..., None, :]
        
        # clamp the arguments of the exponents
        arg1 = torch.clamp(arg1, min=-50.0, max=50.0)
        arg2 = torch.clamp(arg2, min=-50.0, max=50.0)

        # denominator: (batch, num_poles, Ltau, N_nodes)
        denominator = torch.exp(arg1) + torch.exp(arg2)

        # integrand: (batch, num_poles, Ltau, N_nodes)
        integrand = numerator[..., None, :] / denominator
        
        # integrate: (batch, Ltau, N_nodes)
        integrand = integrand.sum(dim = 1)

        # perform gaussian quadrature outputting shape (batch, Ltau)
        G_tau = torch.matmul(integrand, self.weights).real
        
        # COMPUTE G(tau) FOR tau = 0
        
        # get matsubara frequencies (N_iwn,)
        iwn = self.iwn
        
        # compute matsubara Green's function in upper-half complex plane
        # (batches, poles, N_iwn)
        G_iwn = (a[..., None] + 1j*b[..., None] )/((epsilon[..., None] -1j*gamma[..., None] ) - iwn[None, None, :])
        
        # sum over poles
        # (batches, N_iwn)
        G_iwn = G_iwn.sum(1)
        
        
        # calculate the 1/(iwn) coefficient
        # (batches, num_poles)
        c = -(a+1j*b)
        
        # calculate the 1/(iwn) contribution to G(iwn)
        # (batches, N_iwn)
        G_iwn -= (torch.real(c[:,:,None])/iwn[None, None,:]).sum(1)
        
        # Update G(tau=0) based on 1/(iwn) constribution
        # (batches, 1)
        G_tau[:,0] = self.coefs[0] * torch.real(c).sum(1)
        
        
        # calculate the 1/(iwn)^2 coefficient
        # (batches, num_poles)
        c *= (epsilon-1j*gamma)
        
        # calculate the 1/(iwn)^2 contribution to G(iwn)
        # (batches, N_iwn)
        G_iwn -= (c[:,:,None]/iwn[None, None,:]**2).sum(1)
        
        # Update G(tau=0) based on 1/(iwn)^2 constribution
        # (batches, 1)
        G_tau[:,0] += self.coefs[1] * torch.real(c).sum(1)
        
        
        # calculate the 1/(iwn)^3 coefficient
        # (batches, num_poles)
        c *= (epsilon-1j*gamma)
        
        # calculate the 1/(iwn)^3 contribution to G(iwn)
        # (batches, N_iwn)
        G_iwn -= (c[:,:,None]/iwn[None, None,:]**3).sum(1)
        
        # Update G(tau=0) based on 1/(iwn)^3 constribution
        # (batches, 1)
        G_tau[:,0] += self.coefs[2] * torch.imag(c).sum(1)
        
        
        # calculate the 1/(iwn)^4 coefficient
        # (batches, num_poles)
        c *= (epsilon-1j*gamma)
        
        # calculate the 1/(iwn)^4 contribution to G(iwn)
        # (batches, N_iwn)
        G_iwn -= (c[:,:,None]/iwn[None, None,:]**4).sum(1)
        
        # Update G(tau=0) based on 1/(iwn)^4 constribution
        # (batches, 1)
        G_tau[:,0] += self.coefs[3] * torch.real(c).sum(1)
        
        
        # calculate the 1/(iwn)^5 coefficient
        # (batches, num_poles)
        c *= (epsilon-1j*gamma)
        
        # calculate the 1/(iwn)^5 contribution to G(iwn)
        # (batches, N_iwn)
        G_iwn -= (c[:,:,None]/iwn[None, None,:]**5).sum(1)
        
        # Update G(tau=0) based on 1/(iwn)^5 constribution
        # (batches, 1)
        G_tau[:,0] += self.coefs[4] * torch.imag(c).sum(1)
        

        # Update G(tau=0) by fourier transforming what is left of G(iwn)
        # after subtracting of 1/(iwn), ..., 1/(iwn)^5 contributions exactly
        # (batches, 1)
        G_tau[:,0] += 2*torch.real(G_iwn.sum(1)/self.beta)

        return G_tau
    
if __name__ == "__main__":
    
    beta = 10.0
    dtau = 0.05
    N_nodes = 256
    greens_reconstructor = PoleToGaussLegendreGreens(beta, dtau, N_nodes)

    batch_size = 4
    num_poles = 5
    poles = torch.randn(batch_size, num_poles) + 1j * torch.randn(batch_size, num_poles)
    residues = torch.randn(batch_size, num_poles) + 1j * torch.randn(batch_size, num_poles)

    G_tau = greens_reconstructor(poles, residues)
    print("G(tau) shape:", G_tau.shape)  # Should be (batch_size, Ltau)