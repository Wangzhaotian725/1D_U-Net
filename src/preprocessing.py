"""Spectrum preprocessing: normalize + log-compress."""
from __future__ import annotations

import numpy as np


class Preprocessor:
    """Transform spectra for neural network input/output.

    Steps
    -----
    1. Normalize to density (divide by sum).
    2. Optionally log1p-compress: y = log1p(scale * y).

    Inverse applies expm1 then rescales by total_counts.
    """

    def __init__(
        self,
        normalize: bool = True,
        log_compress: bool = True,
        log_scale: float = 1e4,
    ) -> None:
        self.normalize = normalize
        self.log_compress = log_compress
        self.log_scale = log_scale

    def transform(self, spectrum: np.ndarray) -> tuple[np.ndarray, float]:
        """Transform a spectrum for model input/output.

        Parameters
        ----------
        spectrum : (..., N) array

        Returns
        -------
        (transformed, total_counts)
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        total = float(spectrum.sum())

        if self.normalize and total > 0:
            y = spectrum / total
        else:
            y = spectrum.copy()

        if self.log_compress:
            y = np.log1p(self.log_scale * y)

        return y.astype(np.float32), total

    def inverse_transform(
        self, y: np.ndarray, total_counts: float = 1.0
    ) -> np.ndarray:
        """Invert the preprocessing transform.

        Parameters
        ----------
        y            : (..., N) transformed array
        total_counts : scalar saved from transform()

        Returns
        -------
        (..., N) reconstructed spectrum
        """
        y = np.asarray(y, dtype=np.float64)

        if self.log_compress:
            y = np.expm1(y) / self.log_scale

        y = np.clip(y, 0.0, None)

        if self.normalize and total_counts > 0:
            y = y * total_counts

        return y.astype(np.float64)
