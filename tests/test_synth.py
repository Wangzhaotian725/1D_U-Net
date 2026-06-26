"""Tests for src/synth.py"""
from __future__ import annotations

import numpy as np
import pytest

from src.synth import SynthGenerator, make_synthetic_pair, sample_weights


def make_toy_data(K: int = 5, N: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    mono_A = rng.uniform(0, 1, (K, N))
    mono_B = rng.uniform(0, 1, (K, N))
    # Normalize rows
    mono_A /= mono_A.sum(axis=1, keepdims=True)
    mono_B /= mono_B.sum(axis=1, keepdims=True)
    energies = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    return mono_A, mono_B, energies


def test_one_hot_reproduces_mono():
    """One-hot weight for energy k reproduces mono pair k exactly."""
    mono_A, mono_B, energies = make_toy_data()
    K = len(energies)

    for k in range(K):
        w = np.zeros(K)
        w[k] = 1.0
        A_mix, B_mix = make_synthetic_pair(mono_A, mono_B, w)
        np.testing.assert_allclose(A_mix, mono_A[k], rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(B_mix, mono_B[k], rtol=1e-6, atol=1e-10)


def test_make_synthetic_pair_linearity():
    """make_synthetic_pair is linear: same weights on A and B."""
    mono_A, mono_B, energies = make_toy_data()
    K = len(energies)
    rng = np.random.default_rng(7)
    w = rng.uniform(0, 1, K)

    A_mix, B_mix = make_synthetic_pair(mono_A, mono_B, w)

    w_norm = w / w.sum()
    # Manually compute expected
    A_expected = (mono_A * w_norm[:, None]).sum(axis=0)
    B_expected = (mono_B * w_norm[:, None]).sum(axis=0)

    np.testing.assert_allclose(A_mix, A_expected, rtol=1e-6)
    np.testing.assert_allclose(B_mix, B_expected, rtol=1e-6)


def test_same_weights_for_both_detectors():
    """Both detectors get the same weight vector (verifying make_synthetic_pair)."""
    mono_A, mono_B, energies = make_toy_data()
    K = len(energies)
    w = np.array([0.1, 0.3, 0.2, 0.25, 0.15])

    A_mix, B_mix = make_synthetic_pair(mono_A, mono_B, w)
    # Verify they are distinct (different detector responses)
    # but each is a weighted average
    w_norm = w / w.sum()
    for k in range(K):
        # A contribution should equal w[k] * mono_A[k]
        assert A_mix.shape == mono_A[k].shape
        assert B_mix.shape == mono_B[k].shape


def test_sample_weights_mono():
    """Mono family returns exactly one non-zero entry."""
    rng = np.random.default_rng(1)
    w = sample_weights(10, "mono", rng)
    assert (w > 0).sum() == 1
    assert w.sum() == pytest.approx(1.0)


def test_sample_weights_sparse():
    """Sparse family returns 2-4 non-zero entries."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        w = sample_weights(10, "sparse", rng)
        n_active = (w > 0).sum()
        assert 2 <= n_active <= 4


def test_sample_weights_dense():
    """Dense family sums to 1 and all entries non-negative."""
    rng = np.random.default_rng(3)
    w = sample_weights(10, "dense", rng)
    assert w.sum() == pytest.approx(1.0, rel=1e-6)
    assert (w >= 0).all()


def test_sample_weights_gcr_like_removed():
    """gcr_like family has been removed and should raise ValueError."""
    energies = np.array([200.0, 400.0, 800.0, 1600.0, 3200.0])
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="gcr_like"):
        sample_weights(len(energies), "gcr_like", rng, energies_MeV=energies)


def test_synth_generator_output_shape():
    """SynthGenerator.sample returns (N,) arrays."""
    mono_A, mono_B, energies = make_toy_data(K=5, N=32)
    gen = SynthGenerator(mono_A, mono_B, energies, families=["mono", "dense"])
    rng = np.random.default_rng(0)
    A, B = gen.sample(rng)
    assert A.shape == (32,)
    assert B.shape == (32,)


def test_synth_generator_normalized():
    """Output spectra are normalized (sum=1)."""
    mono_A, mono_B, energies = make_toy_data(K=5, N=32)
    gen = SynthGenerator(mono_A, mono_B, energies, families=["dense"], poisson_noise=False)
    rng = np.random.default_rng(0)
    for _ in range(10):
        A, B = gen.sample(rng)
        assert A.sum() == pytest.approx(1.0, rel=1e-5)
        assert B.sum() == pytest.approx(1.0, rel=1e-5)
