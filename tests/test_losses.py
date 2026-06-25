"""Tests for src/losses.py"""
from __future__ import annotations

import pytest
import torch

from src.losses import SpectrumLoss, emd_1d


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
