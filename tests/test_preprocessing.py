"""Tests for src/preprocessing.py"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import Preprocessor


def make_spectrum(n: int = 360, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 1.0, n)


def test_round_trip_with_log():
    """transform -> inverse_transform recovers original within 1e-5."""
    prep = Preprocessor(normalize=True, log_compress=True, log_scale=1e4)
    spec = make_spectrum()

    transformed, total = prep.transform(spec)
    recovered = prep.inverse_transform(transformed, total_counts=total)

    np.testing.assert_allclose(recovered, spec, rtol=1e-5, atol=1e-8)


def test_round_trip_no_log():
    """Round-trip without log compression."""
    prep = Preprocessor(normalize=True, log_compress=False)
    spec = make_spectrum()

    transformed, total = prep.transform(spec)
    recovered = prep.inverse_transform(transformed, total_counts=total)

    np.testing.assert_allclose(recovered, spec, rtol=1e-5, atol=1e-8)


def test_round_trip_no_normalize():
    """Round-trip without normalization."""
    prep = Preprocessor(normalize=False, log_compress=True, log_scale=1e4)
    spec = make_spectrum() * 1000.0  # large counts

    transformed, total = prep.transform(spec)
    recovered = prep.inverse_transform(transformed, total_counts=1.0)

    # Without normalization, total_counts is irrelevant
    np.testing.assert_allclose(recovered, spec, rtol=1e-5, atol=1e-6)


def test_transform_returns_float32():
    prep = Preprocessor()
    spec = make_spectrum()
    transformed, _ = prep.transform(spec)
    assert transformed.dtype == np.float32


def test_total_counts_saved():
    """Total counts saved during transform equals original sum."""
    prep = Preprocessor(normalize=True, log_compress=False)
    spec = make_spectrum() * 5000.0
    _, total = prep.transform(spec)
    np.testing.assert_allclose(total, spec.sum(), rtol=1e-10)


def test_2d_input():
    """Preprocessor handles 2D input (batch)."""
    prep = Preprocessor(normalize=True, log_compress=True, log_scale=1e4)
    spec = make_spectrum(360).reshape(1, 360)
    transformed, total = prep.transform(spec)
    assert transformed.shape == (1, 360)
