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
    gcr_powerlaw_index: float = -2.7,
    energies_MeV: np.ndarray | None = None,
) -> np.ndarray:
    """Sample mixture weights for a given family.

    Parameters
    ----------
    n_energies        : number of energy bins (K)
    family            : 'mono' | 'sparse' | 'dense' | 'gcr_like'
    rng               : numpy Generator
    gcr_powerlaw_index: exponent for power-law weights
    energies_MeV      : (K,) energy values (required for 'gcr_like')

    Returns
    -------
    (K,) non-negative weights (not necessarily normalized)
    """
    if family == "mono":
        w = np.zeros(n_energies)
        idx = rng.integers(0, n_energies)
        w[idx] = 1.0
        return w

    elif family == "sparse":
        n_active = rng.integers(2, 5)  # 2-4 active
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

    elif family == "gcr_like":
        if energies_MeV is None:
            # Fall back to indices as proxy for energy
            e = np.arange(1, n_energies + 1, dtype=np.float64)
        else:
            e = np.asarray(energies_MeV, dtype=np.float64)
        w = e ** gcr_powerlaw_index
        # Add some random noise to avoid identical samples
        noise = rng.uniform(0.5, 1.5, size=n_energies)
        w = w * noise
        w = np.clip(w, 0.0, None)
        return w

    else:
        raise ValueError(f"Unknown mixture family: {family!r}")


class SynthGenerator:
    """Generate synthetic (A, B) spectrum pairs for training.

    Parameters
    ----------
    mono_A            : (K, N) monoenergetic A spectra (pre-normalized)
    mono_B            : (K, N) monoenergetic B spectra (pre-normalized)
    energies_MeV      : (K,) energy values in MeV
    families          : list of family names to randomly select from
    poisson_noise     : whether to add Poisson noise to the mixture
    gcr_powerlaw_index: exponent for 'gcr_like' family
    """

    def __init__(
        self,
        mono_A: np.ndarray,
        mono_B: np.ndarray,
        energies_MeV: np.ndarray,
        families: list[str],
        poisson_noise: bool = True,
        gcr_powerlaw_index: float = -2.7,
    ) -> None:
        self.mono_A = np.asarray(mono_A, dtype=np.float64)
        self.mono_B = np.asarray(mono_B, dtype=np.float64)
        self.energies_MeV = np.asarray(energies_MeV, dtype=np.float64)
        self.families = list(families)
        self.poisson_noise = poisson_noise
        self.gcr_powerlaw_index = gcr_powerlaw_index
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
            gcr_powerlaw_index=self.gcr_powerlaw_index,
            energies_MeV=self.energies_MeV,
        )

        A_mix, B_mix = make_synthetic_pair(self.mono_A, self.mono_B, weights)

        if self.poisson_noise:
            # Simulate Poisson noise at a random count level
            n_counts = int(rng.uniform(1e3, 1e5))
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
