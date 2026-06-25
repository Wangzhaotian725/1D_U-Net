"""PyTorch datasets for synthetic spectrum mixtures."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import Preprocessor
from src.synth import SynthGenerator


class SyntheticMixtureDataset(Dataset):
    """Generates synthetic (A, B) pairs on the fly.

    Each call to __getitem__ samples fresh weights and mixes the
    monoenergetic spectra, applies the preprocessor, and returns tensors.

    Parameters
    ----------
    generator         : SynthGenerator
    preprocessor      : Preprocessor
    samples_per_epoch : number of samples (defines dataset length)
    base_seed         : base RNG seed; item index is added for reproducibility
    """

    def __init__(
        self,
        generator: SynthGenerator,
        preprocessor: Preprocessor,
        samples_per_epoch: int = 1024,
        base_seed: int = 0,
    ) -> None:
        self.generator = generator
        self.preprocessor = preprocessor
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + idx)
        A_mix, B_mix = self.generator.sample(rng)

        A_t, _ = self.preprocessor.transform(A_mix)
        B_t, _ = self.preprocessor.transform(B_mix)

        # Shape: (1, N) for Conv1d
        A_tensor = torch.from_numpy(A_t).unsqueeze(0)
        B_tensor = torch.from_numpy(B_t).unsqueeze(0)

        return {"input": A_tensor, "target": B_tensor}


class FixedMixtureSet(Dataset):
    """Fixed, seeded set for validation/test.

    Generated once at init from held-out energies.

    Parameters
    ----------
    generator         : SynthGenerator built from held-out energies
    preprocessor      : Preprocessor
    n_samples         : number of samples to generate
    seed              : fixed seed for reproducibility
    """

    def __init__(
        self,
        generator: SynthGenerator,
        preprocessor: Preprocessor,
        n_samples: int = 256,
        seed: int = 999,
    ) -> None:
        self.preprocessor = preprocessor
        rng = np.random.default_rng(seed)

        inputs = []
        targets = []
        for _ in range(n_samples):
            A_mix, B_mix = generator.sample(rng)
            A_t, _ = preprocessor.transform(A_mix)
            B_t, _ = preprocessor.transform(B_mix)
            inputs.append(A_t)
            targets.append(B_t)

        # (n_samples, 1, N)
        self.inputs = torch.from_numpy(np.stack(inputs)).unsqueeze(1)
        self.targets = torch.from_numpy(np.stack(targets)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input": self.inputs[idx], "target": self.targets[idx]}
