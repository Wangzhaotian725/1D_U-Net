"""Tests for src/model.py"""
from __future__ import annotations

import pytest
import torch

from src.model import UNet1D


def make_input(B: int = 4, L: int = 360) -> torch.Tensor:
    x = torch.rand(B, 1, L)
    return x


def test_output_shape():
    """Output shape matches input spatial size (B, 1, 360)."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="softmax")
    x = make_input(B=4, L=360)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1, 360), f"Expected (4,1,360), got {out.shape}"


def test_no_nans():
    """Forward pass produces no NaNs."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="softmax")
    x = make_input(B=4, L=360)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


def test_softmax_head_sums_to_one():
    """Softmax head: output.sum(dim=-1) ≈ 1.0 for each sample."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="softmax")
    x = make_input(B=8, L=360)
    with torch.no_grad():
        out = model(x)
    sums = out.sum(dim=-1).squeeze(1)  # (B,)
    torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=1e-5)


def test_softplus_renorm_head_positive_and_normalized():
    """softplus_renorm head: output is non-negative and normalized (sum=1)."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="softplus_renorm")
    x = make_input(B=4, L=360)
    with torch.no_grad():
        out = model(x)
    assert (out >= 0).all(), "softplus_renorm head output should be non-negative"
    sums = out.sum(dim=-1).squeeze(1)
    torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)


def test_softplus_head_positive_not_normalized():
    """softplus head (v0.4): non-negative and NON-normalized (can emit zeros)."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="softplus")
    x = make_input(B=4, L=360)
    with torch.no_grad():
        out = model(x)
    assert (out >= 0).all(), "softplus head output should be non-negative"
    # It must NOT be forced to sum to 1 (that's the whole point of the head).
    sums = out.sum(dim=-1).squeeze(1)
    assert not torch.allclose(sums, torch.ones(4), atol=1e-3)


def test_relu_head_non_negative():
    """relu head: non-negative, non-normalized, can emit exact zeros."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="relu")
    x = make_input(B=4, L=360)
    with torch.no_grad():
        out = model(x)
    assert (out >= 0).all(), "relu head output should be non-negative"


def test_different_depths():
    """Model works for depth=1, 2, 3, 4."""
    for depth in [1, 2, 3, 4]:
        model = UNet1D(in_ch=1, out_ch=1, base=8, depth=depth, head="softmax")
        x = make_input(B=2, L=360)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 360), f"Failed at depth={depth}: {out.shape}"


def test_odd_input_length():
    """Model handles odd-length inputs via pad/crop."""
    model = UNet1D(in_ch=1, out_ch=1, base=16, depth=3, head="softmax")
    x = make_input(B=2, L=361)  # odd
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 361)


def test_gradient_flows():
    """Gradients flow through model parameters."""
    model = UNet1D(in_ch=1, out_ch=1, base=8, depth=2, head="softmax")
    x = make_input(B=2, L=360)
    out = model(x)
    loss = out.mean()
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    assert len(grad_norms) > 0, "No gradients computed"
    assert all(not (g != g) for g in grad_norms), "NaN gradient found"
