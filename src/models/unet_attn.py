"""Attention-gated 1D U-Net (Direction C, v0.6).

Adds learnable attention gates to each skip connection of the standard U-Net.
The attention gate learns to suppress irrelevant spatial information from the
encoder before concatenation with the decoder features. This allows the network
to selectively reduce the leakage of the input peak position through skip
connections, which is the hypothesised cause of the secondary peak artefact.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the
Pancreas", MIDL 2018.

Each attention gate computes:
    α = σ( W_x(skip) + W_g(g) + b )
    gated_skip = α * skip
where g is the gating signal (upsampled decoder feature) and skip is the
encoder feature at the same resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ConvBlock, _pad_or_crop


class AttentionGate(nn.Module):
    """1D attention gate that modulates encoder skip features with decoder signal.

    Parameters
    ----------
    skip_ch : number of channels in the skip (encoder) feature
    gate_ch : number of channels in the gating (decoder) feature
    inter_ch: intermediate channel count for the attention computation
    """

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int | None = None) -> None:
        super().__init__()
        inter_ch = inter_ch or max(skip_ch // 2, 1)
        self.W_skip = nn.Conv1d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.W_gate = nn.Conv1d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.psi    = nn.Conv1d(inter_ch, 1, kernel_size=1)
        self.bn     = nn.BatchNorm1d(inter_ch)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Compute and apply attention gate.

        Parameters
        ----------
        skip : (B, skip_ch, L)  encoder feature at full resolution
        gate : (B, gate_ch, L') decoder gating signal (may differ in length)

        Returns
        -------
        (B, skip_ch, L) gated skip features
        """
        # Align gate to skip resolution
        gate = F.interpolate(gate, size=skip.shape[-1], mode="linear", align_corners=False)
        gate = _pad_or_crop(gate, skip.shape[-1])

        feat = F.relu(self.bn(self.W_skip(skip) + self.W_gate(gate)), inplace=True)
        alpha = torch.sigmoid(self.psi(feat))      # (B, 1, L) in [0, 1]
        return skip * alpha


class AttnUNet1D(nn.Module):
    """1D U-Net with attention gates on all skip connections.

    Parameters
    ----------
    in_ch  : input channels
    out_ch : output channels
    base   : base channel count for the first encoder block
    depth  : number of encoder/decoder levels
    head   : output activation (same options as UNet1D)
    skip_gate_scale : list of per-level gate scales. 0.0 disables the gate
                      (ablation), 1.0 uses the attention gate normally.
                      None means all gates active (default).
    """

    NORMALIZED_HEADS = ("softmax", "softplus_renorm")
    UNNORMALIZED_HEADS = ("softplus", "relu")

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 32,
        depth: int = 3,
        head: str = "softplus_renorm",
        skip_gate_scale: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.head = head
        if skip_gate_scale is None:
            skip_gate_scale = [1.0] * depth
        self.skip_gate_scale = skip_gate_scale

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        enc_channels: list[int] = []
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

        # Decoder + attention gates
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            out = base * (2 ** i)
            self.up_convs.append(nn.Conv1d(ch, out, kernel_size=1))
            self.attn_gates.append(AttentionGate(skip_ch=skip_ch, gate_ch=out))
            self.dec_blocks.append(ConvBlock(out + skip_ch, out))
            ch = out

        self.head_conv = nn.Conv1d(ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]

        # Encoder
        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with attention-gated skips
        for up_conv, attn, dec, skip, scale in zip(
            self.up_convs, self.attn_gates, self.dec_blocks,
            reversed(skips), reversed(self.skip_gate_scale)
        ):
            target_len = skip.shape[-1]
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
            x = up_conv(x)
            x = _pad_or_crop(x, target_len)
            skip = _pad_or_crop(skip, target_len)

            if scale > 0.0:
                gated_skip = attn(skip, x) * scale
            else:
                gated_skip = torch.zeros_like(skip)   # ablate this skip

            x = torch.cat([x, gated_skip], dim=1)
            x = dec(x)

        x = self.head_conv(x)
        x = _pad_or_crop(x, L)

        # Output head (same as UNet1D)
        if self.head == "softmax":
            x = torch.softmax(x, dim=-1)
        elif self.head == "softplus_renorm":
            x = F.softplus(x)
            s = x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            x = x / s
        elif self.head == "softplus":
            x = F.softplus(x)
        elif self.head == "relu":
            x = F.relu(x)
        else:
            raise ValueError(f"Unknown head: {self.head!r}")

        return x
