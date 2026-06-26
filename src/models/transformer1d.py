"""1D Transformer encoder for spectrum-to-spectrum mapping.

Self-attention allows arbitrary bin-to-bin mass redistribution (non-local
"transport"), which is what the SiC→TEPC task requires. Crucially, there
are NO long-range skip connections from encoder to decoder, so the input
peak position cannot leak directly into the output — the root cause of the
double-peak artefact observed in U-Net-based models.

Architecture
------------
  input (B, 1, N)
    → per-bin linear projection to d_model
    → learnable positional encoding
    → N_layers × TransformerEncoderLayer (pre-norm, batch_first)
    → linear projection back to 1 channel
    → output head (softplus_renorm)
    → (B, 1, N)

No skip connections. The model learns the full SiC→TEPC mapping implicitly
through the attention weights, which are unconstrained and can represent
arbitrary long-range energy redistribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumTransformer(nn.Module):
    """Transformer encoder for 1D spectrum translation.

    Parameters
    ----------
    n_bins          : number of spectral bins
    d_model         : token embedding dimension
    n_layers        : number of TransformerEncoderLayers
    n_heads         : number of attention heads (must divide d_model)
    dim_feedforward : FFN hidden dimension inside each encoder layer
    dropout         : dropout probability (0 for small datasets)
    head            : output activation
    """

    def __init__(
        self,
        n_bins: int = 360,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        head: str = "softplus_renorm",
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.head = head

        # Per-bin linear projection: scalar → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Learnable positional encoding (one vector per bin)
        self.pos_enc = nn.Parameter(torch.randn(1, n_bins, d_model) * 0.02)

        # Transformer encoder (pre-norm = norm_first=True for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection: d_model → 1
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 1, N)

        Returns
        -------
        (B, 1, N)
        """
        x = x.squeeze(1)              # (B, N)
        x = x.unsqueeze(-1)           # (B, N, 1)
        x = self.input_proj(x)        # (B, N, d_model)
        x = x + self.pos_enc          # (B, N, d_model) — broadcast over batch
        x = self.encoder(x)           # (B, N, d_model)
        x = self.output_proj(x)       # (B, N, 1)
        x = x.squeeze(-1)             # (B, N)

        # Output head
        if self.head == "softplus_renorm":
            x = F.softplus(x)
            x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        elif self.head == "softmax":
            x = torch.softmax(x, dim=-1)
        elif self.head == "softplus":
            x = F.softplus(x)
        elif self.head == "relu":
            x = F.relu(x)
        else:
            raise ValueError(f"Unknown head: {self.head!r}")

        return x.unsqueeze(1)         # (B, 1, N)

    def extra_repr(self) -> str:
        return (
            f"n_bins={self.n_bins}, d_model={self.encoder.layers[0].self_attn.embed_dim}, "
            f"n_layers={len(self.encoder.layers)}, head={self.head!r}"
        )
