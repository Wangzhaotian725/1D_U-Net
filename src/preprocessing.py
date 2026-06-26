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


# Heads whose output is a normalized density (sum=1). Their target must be a
# plain density. Non-normalized heads (everything else) require a log-compressed
# target so prediction and target share the same numeric space. This pairing is
# the mirror of the v0.1 fatal bug and is locked by
# tests/test_leakage.py::test_preprocessor_head_match.
NORMALIZED_HEADS = ("softmax", "softplus_renorm")


def build_preprocessors(cfg) -> tuple["Preprocessor", "Preprocessor"]:
    """Build the (input, target) preprocessor pair from a config.

    The INPUT is always log-compressed (when enabled) as a representation aid for
    the 6-decade dynamic range. The TARGET space depends on the model head:

    - normalized head (softmax, softplus_renorm): target is a plain density,
      never log-compressed (the head cannot reach a log-compressed magnitude).
    - non-normalized head (softplus, relu): target IS log-compressed, so the head
      (which can emit true zeros) shares the target's space and the near-zero
      tail is learnable.

    Returns
    -------
    (input_pre, target_pre)
    """
    normalize = cfg.preprocessing.normalize_to_density
    log_scale = cfg.preprocessing.log_scale
    log_compress_input = bool(cfg.preprocessing.log_compress)

    head = getattr(cfg.model, "head", "softmax")
    target_log = log_compress_input if head not in NORMALIZED_HEADS else False

    input_pre = Preprocessor(
        normalize=normalize, log_compress=log_compress_input, log_scale=log_scale
    )
    target_pre = Preprocessor(
        normalize=normalize, log_compress=target_log, log_scale=log_scale
    )
    return input_pre, target_pre
