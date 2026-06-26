"""1D U-Net for spectrum transformation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(cfg) -> nn.Module:
    """Factory: instantiate the right architecture from cfg.model.arch.

    Supported values for cfg.model.arch:
        'unet1d'        → UNet1D (baseline)
        'unet_attn'     → AttnUNet1D (attention-gated skip connections)
        'operator'      → TransferOperator (low-rank band-limited)
        'transformer1d' → SpectrumTransformer (no skip connections)
    """
    arch = str(cfg.model.get("arch", "unet1d")).lower()

    if arch == "unet1d":
        return UNet1D(
            in_ch=cfg.model.in_ch,
            out_ch=1,
            base=cfg.model.base_channels,
            depth=cfg.model.depth,
            head=cfg.model.head,
        )
    elif arch == "unet_attn":
        from src.models.unet_attn import AttnUNet1D
        skip_gate_scale = list(cfg.model.get("skip_gate_scale", [1.0] * cfg.model.depth))
        return AttnUNet1D(
            in_ch=cfg.model.in_ch,
            out_ch=1,
            base=cfg.model.base_channels,
            depth=cfg.model.depth,
            head=cfg.model.head,
            skip_gate_scale=skip_gate_scale,
        )
    elif arch == "operator":
        from src.models.operator import TransferOperator
        return TransferOperator(
            n_bins=int(cfg.model.get("n_bins", 360)),
            rank=int(cfg.model.get("operator_rank", 24)),
            band_halfwidth=cfg.model.get("operator_band_halfwidth", 60),
            head=cfg.model.head,
        )
    elif arch == "transformer1d":
        from src.models.transformer1d import SpectrumTransformer
        return SpectrumTransformer(
            n_bins=int(cfg.model.get("n_bins", 360)),
            d_model=int(cfg.model.get("d_model", 128)),
            n_layers=int(cfg.model.get("n_layers", 4)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dim_feedforward=int(cfg.model.get("dim_feedforward", 512)),
            dropout=float(cfg.model.get("dropout", 0.0)),
            head=cfg.model.head,
        )
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose from unet1d, unet_attn, operator, transformer1d")


def _pad_or_crop(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or crop the last dimension of x to target_len."""
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    elif x.shape[-1] < target_len:
        return F.pad(x, (0, target_len - x.shape[-1]))
    return x


class ConvBlock(nn.Module):
    """Two Conv1d layers each followed by GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8) -> None:
        super().__init__()
        # Clamp groups to avoid issues with small channel counts
        g1 = min(num_groups, in_ch)
        g2 = min(num_groups, out_ch)
        # Ensure divisibility
        while g1 > 1 and in_ch % g1 != 0:
            g1 //= 2
        while g2 > 1 and out_ch % g2 != 0:
            g2 //= 2

        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g2, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g2, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet1D(nn.Module):
    """1D U-Net for 1-channel spectrum-to-spectrum mapping.

    Architecture
    ------------
    Encoder: depth down-blocks, each halving spatial dimension with MaxPool1d(2).
    Bottleneck: one ConvBlock at the lowest resolution.
    Decoder: depth up-blocks, each doubling with interpolation + skip concat.
    Head: one of
        'softmax'         normalized density (sum=1) via softmax
        'softplus_renorm' normalized density (sum=1) via softplus then divide by sum
        'softplus'        NON-normalized non-negative output (softplus); can emit
                          true zeros, used with a log-compressed target
        'relu'            NON-normalized non-negative output (relu)

    Normalized heads (softmax, softplus_renorm) must be paired with a plain
    density target; non-normalized heads (softplus, relu) must be paired with a
    log-compressed target. build_preprocessors() enforces this; see
    tests/test_leakage.py::test_preprocessor_head_match.

    Parameters
    ----------
    in_ch  : input channels (1)
    out_ch : output channels (1)
    base   : base channel count for the first encoder block
    depth  : number of encoder/decoder levels
    head   : 'softmax' | 'softplus_renorm' | 'softplus' | 'relu'
    """

    NORMALIZED_HEADS = ("softmax", "softplus_renorm")
    UNNORMALIZED_HEADS = ("softplus", "relu")

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 32,
        depth: int = 3,
        head: str = "softmax",
    ) -> None:
        super().__init__()
        self.depth = depth
        self.head = head

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        enc_channels = []
        for i in range(depth):
            out = base * (2 ** i)
            self.enc_blocks.append(ConvBlock(ch, out))
            self.pools.append(nn.MaxPool1d(2))
            enc_channels.append(out)
            ch = out

        # Bottleneck
        bottleneck_ch = base * (2 ** depth)
        self.bottleneck = ConvBlock(ch, bottleneck_ch)
        ch = bottleneck_ch

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            out = base * (2 ** i)
            # 1x1 conv to reduce channels before concat
            self.up_convs.append(nn.Conv1d(ch, out, kernel_size=1))
            self.dec_blocks.append(ConvBlock(out + skip_ch, out))
            ch = out

        # Final projection
        self.head_conv = nn.Conv1d(ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 1, L)

        Returns
        -------
        (B, 1, L) with output summing to 1 along last dim (softmax head)
        """
        L = x.shape[-1]

        # Encoder
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up_conv, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            target_len = skip.shape[-1]
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
            x = up_conv(x)
            x = _pad_or_crop(x, target_len)
            skip = _pad_or_crop(skip, target_len)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head_conv(x)
        # Match input length exactly
        x = _pad_or_crop(x, L)

        # Apply head activation
        if self.head == "softmax":
            x = torch.softmax(x, dim=-1)
        elif self.head == "softplus_renorm":
            x = F.softplus(x)
            s = x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            x = x / s
        elif self.head == "softplus":
            # Non-normalized: can approach a true zero, so a log-compressed
            # target's near-zero tail is reachable.
            x = F.softplus(x)
        elif self.head == "relu":
            x = F.relu(x)
        else:
            raise ValueError(f"Unknown head: {self.head!r}")

        return x
