"""Synthetic mixture generation for training."""
from __future__ import annotations

import numpy as np


def make_synthetic_pair(
    mono_A: np.ndarray,
    mono_B: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix monoenergetic spectra with given weights.

    Parameters
    ----------
    mono_A  : (K, N) detector-A spectra for K energies
    mono_B  : (K, N) detector-B spectra for K energies
    weights : (K,) non-negative mixture weights

    Returns
    -------
    (A_mix, B_mix) each (N,), normalized to sum=1
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    A_mix = (mono_A * w[:, None]).sum(axis=0)
    B_mix = (mono_B * w[:, None]).sum(axis=0)

    return A_mix, B_mix


def sample_weights(
    n_energies: int,
    family: str,
    rng: np.random.Generator,
    energies_MeV: np.ndarray | None = None,
    dirichlet_alpha_choices: list[float] | None = None,
    sparse_k_range: tuple[int, int] = (2, 4),
) -> np.ndarray:
    """Sample mixture weights for a given family.

    Parameters
    ----------
    n_energies              : number of energy bins (K)
    family                  : 'mono' | 'sparse' | 'sparse_k' | 'dense' |
                              'dirichlet_uniform' | 'loguniform'
    rng                     : numpy Generator
    energies_MeV            : (K,) energy values in MeV (unused by most families)
    dirichlet_alpha_choices : alpha values to pick from for 'dirichlet_uniform'
    sparse_k_range          : (min, max) active energies for 'sparse_k'

    Returns
    -------
    (K,) non-negative weights (not necessarily normalized)
    """
    if dirichlet_alpha_choices is None:
        dirichlet_alpha_choices = [0.3, 1.0, 3.0]

    if family == "mono":
        w = np.zeros(n_energies)
        idx = rng.integers(0, n_energies)
        w[idx] = 1.0
        return w

    elif family in ("sparse", "sparse_k"):
        # 'sparse' is kept as alias for backward compatibility
        lo, hi = sparse_k_range
        n_active = rng.integers(lo, hi + 1)  # inclusive hi
        n_active = min(n_active, n_energies)
        indices = rng.choice(n_energies, size=n_active, replace=False)
        raw = rng.uniform(0.1, 1.0, size=n_active)
        w = np.zeros(n_energies)
        w[indices] = raw
        return w

    elif family == "dense":
        # Dirichlet(alpha=1) == uniform on simplex
        w = rng.dirichlet(np.ones(n_energies))
        return w

    elif family == "dirichlet_uniform":
        alpha = float(rng.choice(dirichlet_alpha_choices))
        w = rng.dirichlet(np.full(n_energies, alpha))
        return w

    elif family == "loguniform":
        log_lo = np.log(1e-3)
        log_hi = np.log(1.0)
        log_w = rng.uniform(log_lo, log_hi, size=n_energies)
        w = np.exp(log_w)
        # normalize
        w = w / w.sum()
        return w

    else:
        raise ValueError(f"Unknown mixture family: {family!r}")


class SynthGenerator:
    """Generate synthetic (A, B) spectrum pairs for training.

    Parameters
    ----------
    mono_A                  : (K, N) monoenergetic A spectra (pre-normalized)
    mono_B                  : (K, N) monoenergetic B spectra (pre-normalized)
    energies_MeV            : (K,) energy values in MeV
    families                : list of family names to randomly select from
    poisson_noise           : whether to add Poisson noise to the mixture
    poisson_counts_range    : (min, max) total counts for Poisson sampling
    dirichlet_alpha_choices : alpha values for 'dirichlet_uniform' family
    sparse_k_range          : (min, max) active bins for 'sparse_k' family
    """

    def __init__(
        self,
        mono_A: np.ndarray,
        mono_B: np.ndarray,
        energies_MeV: np.ndarray,
        families: list[str],
        poisson_noise: bool = True,
        poisson_counts_range: tuple[int, int] = (1000, 100000),
        dirichlet_alpha_choices: list[float] | None = None,
        sparse_k_range: tuple[int, int] = (2, 4),
    ) -> None:
        self.mono_A = np.asarray(mono_A, dtype=np.float64)
        self.mono_B = np.asarray(mono_B, dtype=np.float64)
        self.energies_MeV = np.asarray(energies_MeV, dtype=np.float64)
        self.families = list(families)
        self.poisson_noise = poisson_noise
        self.poisson_counts_range = poisson_counts_range
        self.dirichlet_alpha_choices = dirichlet_alpha_choices if dirichlet_alpha_choices is not None else [0.3, 1.0, 3.0]
        self.sparse_k_range = sparse_k_range
        self.n_energies = len(energies_MeV)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample one (A_mix, B_mix) pair.

        Returns
        -------
        (A_mix, B_mix) each (N,), float64, normalized
        """
        family = rng.choice(self.families)
        weights = sample_weights(
            self.n_energies,
            family,
            rng,
            energies_MeV=self.energies_MeV,
            dirichlet_alpha_choices=self.dirichlet_alpha_choices,
            sparse_k_range=self.sparse_k_range,
        )

        A_mix, B_mix = make_synthetic_pair(self.mono_A, self.mono_B, weights)

        if self.poisson_noise:
            # Simulate Poisson noise at a random count level
            lo, hi = self.poisson_counts_range
            n_counts = int(rng.uniform(lo, hi))
            if A_mix.sum() > 0:
                A_counts = rng.poisson(A_mix * n_counts / A_mix.sum())
                A_mix = A_counts.astype(np.float64)
            if B_mix.sum() > 0:
                B_counts = rng.poisson(B_mix * n_counts / B_mix.sum())
                B_mix = B_counts.astype(np.float64)

        # Renormalize
        a_sum = A_mix.sum()
        b_sum = B_mix.sum()
        if a_sum > 0:
            A_mix = A_mix / a_sum
        if b_sum > 0:
            B_mix = B_mix / b_sum

        return A_mix, B_mix
