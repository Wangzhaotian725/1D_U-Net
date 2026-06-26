"""Tests for src/losses.py"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.losses import SpectrumLoss, build_region_mask, build_region_weight, emd_1d, soft_peak_loss


def uniform_dist(n: int, batch: int = 1) -> torch.Tensor:
    return torch.ones(batch, n) / n


def one_hot(n: int, k: int, batch: int = 1) -> torch.Tensor:
    t = torch.zeros(batch, n)
    t[:, k] = 1.0
    return t


def test_emd_identical():
    """emd_1d(p, p) == 0."""
    n = 64
    p = torch.rand(4, n)
    p = p / p.sum(dim=-1, keepdim=True)
    result = emd_1d(p, p)
    assert result.item() == pytest.approx(0.0, abs=1e-6)


def test_emd_one_bin_shift():
    """EMD of adjacent one-hot distributions equals 1 (uniform bin_dist)."""
    n = 10
    p = one_hot(n, 3)  # mass at bin 3
    q = one_hot(n, 4)  # mass at bin 4

    # With uniform weights, EMD = 1 step (each bin contributes 1 to CDF diff)
    result = emd_1d(p, q)
    # CDF of p: 0..0, 1, 1, 1... (steps at 3)
    # CDF of q: 0..0, 0, 1, 1... (steps at 4)
    # |CDF_p - CDF_q| = 1 for bins 3..9 (7 bins with diff=1, rest 0)
    # Wait: |diff| at each bin: bins 0,1,2 -> 0; bin 3 -> 1; bins 4..9 -> 0
    # Actually CDF_p at bin 3 = 1, CDF_q at bin 3 = 0, so diff = 1
    # CDF_p at bin 4 = 1, CDF_q at bin 4 = 1, so diff = 0
    # So bins 3: |CDF_p - CDF_q| = 1, rest = 0 -> sum = 1
    assert result.item() == pytest.approx(1.0, abs=1e-6)


def test_emd_gradients_flow():
    """Gradients flow through emd_1d."""
    n = 32
    pred = torch.rand(4, n, requires_grad=True)
    pred_norm = torch.softmax(pred, dim=-1)
    target = torch.rand(4, n)
    target = target / target.sum(dim=-1, keepdim=True)

    loss = emd_1d(pred_norm, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_spectrum_loss_forward():
    """SpectrumLoss returns total loss and dict."""
    criterion = SpectrumLoss(w_mse=1.0, w_emd=1.0)
    pred = torch.rand(8, 1, 64)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = torch.rand(8, 1, 64)
    target = target / target.sum(dim=-1, keepdim=True)

    total, info = criterion(pred, target)
    assert "mse" in info
    assert "emd" in info
    assert total.item() >= 0


def test_spectrum_loss_zero_on_identical():
    """SpectrumLoss == 0 when pred == target."""
    criterion = SpectrumLoss(w_mse=1.0, w_emd=1.0)
    n = 64
    p = torch.rand(4, 1, n)
    p = p / p.sum(dim=-1, keepdim=True)

    total, info = criterion(p, p)
    assert total.item() == pytest.approx(0.0, abs=1e-6)


def test_emd_with_bin_dist():
    """emd_1d respects bin_dist weighting."""
    n = 10
    p = one_hot(n, 3)
    q = one_hot(n, 4)

    bin_dist = torch.ones(n) * 2.0  # double weight
    result_weighted = emd_1d(p, q, bin_dist=bin_dist)
    result_uniform = emd_1d(p, q)

    # Weighted should be 2x uniform (since bin_dist=2 everywhere)
    assert result_weighted.item() == pytest.approx(2.0 * result_uniform.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# soft_peak_loss tests (v0.5)
# ---------------------------------------------------------------------------

def test_soft_peak_loss_aligned():
    """soft_peak_loss(x, x) ≈ 0 for any input."""
    x = torch.rand(4, 360).softmax(dim=-1)
    loss = soft_peak_loss(x, x.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_soft_peak_loss_increases_with_shift():
    """Larger peak shift produces larger soft_peak_loss."""
    torch.manual_seed(0)
    x = torch.rand(4, 360).softmax(dim=-1)
    # Roll (shift) the target: 10 bins and 30 bins
    loss10 = soft_peak_loss(x, x.roll(10, dims=-1))
    loss30 = soft_peak_loss(x, x.roll(30, dims=-1))
    assert loss10.item() > 0
    assert loss30.item() > loss10.item()


def test_soft_peak_loss_differentiable():
    """Gradient flows through soft_peak_loss w.r.t. pred."""
    pred_logits = torch.randn(4, 360, requires_grad=True)
    pred = torch.softmax(pred_logits, dim=-1)
    target = torch.softmax(torch.randn(4, 360), dim=-1)
    loss = soft_peak_loss(pred, target)
    loss.backward()
    assert pred_logits.grad is not None
    assert not torch.isnan(pred_logits.grad).any()


# ---------------------------------------------------------------------------
# region_weight / region_mask tests (v0.5)
# ---------------------------------------------------------------------------

def _fake_energy_grid(n: int = 360) -> np.ndarray:
    """Log-spaced energy grid from 0.01 to 10000 keV, matching real data."""
    return np.logspace(np.log10(0.01), np.log10(10000), n)


def test_region_weight_values():
    """build_region_weight gives in_weight inside region and out_weight outside."""
    grid = _fake_energy_grid()
    rw = build_region_weight(grid, region_kev=[0.1, 1000.0], in_weight=1.0, out_weight=0.1)
    assert rw.shape == (360,)
    inside = (grid >= 0.1) & (grid <= 1000.0)
    assert torch.allclose(rw[torch.from_numpy(inside)], torch.tensor(1.0))
    assert torch.allclose(rw[torch.from_numpy(~inside)], torch.tensor(0.1))


def test_region_mask_binary():
    """build_region_mask is 1.0 inside and 0.0 outside the region."""
    grid = _fake_energy_grid()
    mask = build_region_mask(grid, region_kev=[0.1, 1000.0])
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    inside = torch.from_numpy((grid >= 0.1) & (grid <= 1000.0))
    assert mask[inside].all()
    assert not mask[~inside].any()


def test_spectrum_loss_region_weight_applied():
    """SpectrumLoss with region_weight gives different MSE than without."""
    n = 360
    grid = _fake_energy_grid(n)
    rw = build_region_weight(grid, [0.1, 1000.0], in_weight=1.0, out_weight=0.1)

    pred = torch.rand(4, n)
    pred = pred / pred.sum(-1, keepdim=True)
    target = torch.rand(4, n)
    target = target / target.sum(-1, keepdim=True)

    loss_uniform = SpectrumLoss(w_mse=1.0, w_emd=0.0)
    loss_weighted = SpectrumLoss(w_mse=1.0, w_emd=0.0, region_weight=rw)

    total_u, info_u = loss_uniform(pred, target)
    total_w, info_w = loss_weighted(pred, target)
    # Weighted MSE differs from uniform MSE since region_weight is non-uniform
    assert not torch.isclose(info_u["mse"], info_w["mse"]), \
        "region_weight should change the MSE value"


def test_spectrum_loss_with_peak_term():
    """SpectrumLoss.forward returns non-zero peak component when w_peak > 0."""
    n = 360
    loss = SpectrumLoss(w_mse=0.0, w_emd=0.0, w_peak=2.0)
    pred = torch.rand(4, n).softmax(dim=-1)
    target = pred.roll(15, dims=-1)  # shifted target → non-zero peak loss
    total, info = loss(pred, target)
    assert info["peak"].item() > 0
    assert total.item() > 0
