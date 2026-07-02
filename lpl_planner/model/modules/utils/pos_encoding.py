import torch
import torch.nn as nn

class VectorizedFourierEncoding(nn.Module):
    def __init__(self, L=10):

        super().__init__()

        self.L = L
        # Precompute all frequency bands [L].
        freqs = 2.0 ** torch.arange(L, dtype=torch.float32)
        self.register_buffer('frequencies', freqs, persistent=False)
        
    def forward(self, trajectory_anchor: torch.Tensor) -> torch.Tensor:
        """
        forward function for vectorized Fourier feature encoding.
        Args:
            trajectory_anchor: torch.Tensor of shape [N, T, S], the input trajectory anchors.
        Returns:
            encoded: torch.Tensor of shape [N, T, 6L], the Fourier feature encoding.
        """
        # V: [N, T, S]
        N, T, S = trajectory_anchor.shape

        # [1, 1, 1, L]
        freqs = self.frequencies.view(1, 1, 1, self.L)

        # [N, T, S, L]
        V_expanded = trajectory_anchor.unsqueeze(-1)  # [N, T, S, 1]
        V_expanded = V_expanded.expand(N, T, S, self.L)  # [N, T, S, L]

        #  π * freq * V
        scaled = torch.pi * freqs * V_expanded  # [N, T, S, L]

        sin_enc = torch.sin(scaled)  # [N, T, S, L]
        cos_enc = torch.cos(scaled)  # [N, T, S, L]

        #  [N, T, S, L] -> [N, T, S*L]
        sin_enc = sin_enc.reshape(N, T, S * self.L)
        cos_enc = cos_enc.reshape(N, T, S * self.L)

        encoded = torch.cat([sin_enc, cos_enc], dim=-1)

        return encoded.reshape(N, T * 2 * S * self.L)
