# sitplus.utils.quantizer
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_pca import PCA
import numpy as np

class SimVQ(nn.Module):
    """
    Implements SimVQ, a vector quantization method that prevents codebook collapse
    by reparameterizing the codebook with a learnable linear transformation.
    
    Reference: "SimVQ: A Simple and Effective Method for Codebook Utilization"
    """
    def __init__(self, num_codes: int, embedding_dim: int, commitment_beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.beta = commitment_beta

        # 1. The Coefficient Matrix 'C'
        # This is initialized from a standard normal distribution and then FROZEN.
        # It acts as a static, well-distributed set of points.
        self.codebook = nn.Embedding(num_codes, embedding_dim)
        nn.init.normal_(self.codebook.weight)
        self.codebook.weight.requires_grad = False

        # 2. The Latent Basis 'W'
        # This is the LEARNABLE d x d linear transformation that rotates and scales
        # the entire space to match the encoder's output distribution.
        self.latent_basis = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def get_effective_codebook(self):
        """Helper to get the transformed codebook C @ W."""
        return self.latent_basis(self.codebook.weight)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: The continuous output of the encoder. Shape: (B, T, D)
        
        Returns:
            A dictionary containing:
            - 'z_q': The quantized vectors. Shape: (B, T, D)
            - 'loss': The total VQ loss (codebook + commitment).
            - 'indices': The chosen codebook indices. Shape: (B, T)
        """
        B, T, D = z_e.shape
        
        # Flatten the encoder output for efficient distance calculation
        z_e_flat = z_e.reshape(-1, D) # Shape: (B*T, D)

        # Get the effective, transformed codebook (C @ W)
        effective_codes = self.get_effective_codebook() # Shape: (K, D)

        # 2. Find the nearest neighbors
        # `torch.cdist` computes pairwise L2 distances.
        # distances shape: (B*T, K)
        distances = torch.cdist(z_e_flat, effective_codes, p=2.0)
        code_indices = torch.argmin(distances, dim=1) # Shape: (B*T)

        # 3. Look up the quantized vectors from the effective codebook
        z_q_flat = effective_codes[code_indices]
        z_q = z_q_flat.view(z_e.shape) # Reshape back to (B, T, D)

        # 4. Losses
        # The "codebook loss" is the gradient that trains W. It pulls the transformed
        # codes (C@W) towards the encoder outputs.
        loss_w = F.mse_loss(z_e.detach(), z_q)
        
        # The "commitment loss" trains the encoder. It pulls the encoder outputs (z_e)
        # towards the (detached) transformed codes.
        loss_commit = self.beta * F.mse_loss(z_e, z_q.detach())
        
        total_loss = loss_w + loss_commit

        # 5. Straight-Through Estimator (STE)
        # On the backward pass, copy the gradients from z_q directly to z_e.
        z_q = z_e + (z_q - z_e).detach()
        
        return {
            "z_q": z_q,
            "loss": total_loss,
            "indices": code_indices.view(B, T),
        }

# ==============================================================================
# --- Statistics & Validation Coda ---
# ==============================================================================
# These are pure functions designed to be called from the training loop
# to validate the behavior of the SimVQ module.

@torch.no_grad()
def compute_codebook_kurtosis(vq_module: SimVQ) -> float:
    """
    Computes the average kurtosis across all dimensions of the effective codebook.
    A Gaussian distribution has a kurtosis of ~3.0. A significant deviation
    indicates that the codebook has learned a non-Gaussian, structured distribution.
    """
    effective_codes = vq_module.get_effective_codebook().cpu()
    # Using scipy for a stable kurtosis calculation
    from scipy.stats import kurtosis
    # Fisher's definition (normal is 0) is default, so we set fisher=False for Pearson's (normal is 3)
    avg_kurt = np.mean(kurtosis(effective_codes.numpy(), axis=0, fisher=False))
    return float(avg_kurt)

@torch.no_grad()
def compute_latent_basis_rank(vq_module: SimVQ) -> int:
    """
    Computes the matrix rank of the learnable latent basis W.
    The paper shows this rank should adaptively decrease during training,
    indicating the model is finding a lower-dimensional manifold.
    """
    W = vq_module.latent_basis.weight
    rank = torch.linalg.matrix_rank(W).item()
    return int(rank)

@torch.no_grad()
def compute_and_plot_pca(vq_module: SimVQ, epoch: int, save_path: str):
    """
    Performs PCA on the effective codebook and saves a 2D scatter plot.
    
    What to look for:
    - Epoch 0: Should be a round, unstructured Gaussian cloud.
    - Later Epochs: Should stretch, rotate, and form clusters/filaments,
      showing that W has learned a data-specific transformation.
    """
    effective_codes = vq_module.get_effective_codebook()
    
    # Use standard PCA for visualization
    pca = PCA(n_components=2, device=effective_codes.device)
    codes_2d = pca.fit_transform(effective_codes).cpu()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(codes_2d[:, 0], codes_2d[:, 1], s=1, alpha=0.5)
    plt.title(f"PCA of Effective Codebook - Epoch {epoch}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()