import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse

from sitplus.utils.encoder import SitEncoder
from sitplus.utils.decoder import SitDecoder
from sitplus.utils.dwt_tokenizer import coeffs_to_raw_tokens, raw_tokens_to_coeffs

def init_weights(module):
    """Applies Kaiming normal initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class SitAutoencoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        
        # We need raw token dims, which depend on DWT. Let's compute them.
        dwt = DWTForward(J=self.config['num_dwt_levels'], wave='haar', mode='zero')
        dummy_img = torch.randn(1, 3, self.config['image_size'], self.config['image_size'])
        yl, yh = dwt(dummy_img)
        tokens_list, shapes_list = coeffs_to_raw_tokens(yl, yh, grid_size=self.config['tokens_per_scale_edge'])
        self.raw_token_dims = [t.shape[-1] for t in tokens_list]
        self.shapes_list = shapes_list

        child_module_args = {
            "d_model": self.config['d_model'],
            "n_layers": self.config['n_layers'],
            "n_heads": self.config['n_heads'],
            "d_ff": self.config.get('d_ff', self.config['d_model'] * 4), # Common default
            "rope_base": self.config.get('rope_base', 500),
            "max_seq_len": self.config.get('max_seq_len', 4096),
            "num_scales": self.config['num_scales'],
            "tokens_per_scale": self.config['tokens_per_scale'],
            "window_radius": self.config.get('window_radius', 2),
            "norm_type": self.config.get('norm_type', 'rmsnorm'),
            "use_qk_norm": self.config.get('use_qk_norm', True),
        }

        # Init components
        self.encoder = SitEncoder(
            raw_token_dims=self.raw_token_dims,
            **child_module_args
        )
        self.decoder = SitDecoder(
            raw_token_dims=self.raw_token_dims,
            **child_module_args
        )

        # --- CONDITIONAL VQ BLOCK ---
        if model_config.get("vector_quantize", False):
            self.vq_block = SimVQ(
                num_codes=model_config['num_codes'],
                embedding_dim=model_config['d_model'],
                commitment_beta=model_config.get('commitment_beta', 0.25)
            )
        else:
            self.vq_block = None
        # --- END CONDITIONAL VQ ---

        self.dwt = DWTForward(J=self.config['num_dwt_levels'], wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        # Apply Kaiming initialization
        self.apply(init_weights)

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # This returns the continuous z_e
        yl, yh = self.dwt(image_tensor)
        tokens_list, _ = coeffs_to_raw_tokens(yl, yh, grid_size=self.config['tokens_per_scale_edge'])
        latents = self.encoder(tokens_list)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # This takes the (potentially quantized) latents and decodes them to DWT coefficients.
        tokens_list_out = self.decoder(latents)
        yl_rec, yh_rec = raw_tokens_to_coeffs(tokens_list_out, self.shapes_list, grid_size=self.config['tokens_per_scale_edge'])
        recon_image = self.idwt((yl_rec, yh_rec))
        # Crop to original size
        H, W = self.config['image_size'], self.config['image_size']
        return recon_image[:, :, :H, :W]

    def forward(self, image_tensor: torch.Tensor):
        """Full autoencoding pass with optional quantization."""
        z_e = self.encode(image_tensor)
        
        if self.vq_block is not None:
            vq_output = self.vq_block(z_e)
            z_q = vq_output['z_q']
            vq_loss = vq_output['loss']
            indices = vq_output['indices']
            
            recon_image = self.decode(z_q)
            return recon_image, vq_loss, indices
        else:
            # If no VQ, pass continuous latents straight through
            recon_image = self.decode(z_e)
            return recon_image, z_e