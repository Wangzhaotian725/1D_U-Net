"""PyTorch datasets for synthetic spectrum mixtures.

Important: input and target use *separate* preprocessors.

The model's softmax head produces a normalized probability density (sums to 1).
The training target must therefore live in the same space -- a plain normalized
density. Log-compression, if enabled, is applied ONLY to the network input
(detector A) as a representation aid for the 6-decade dynamic range; it must
NOT be applied to the target, or prediction and target end up in incompatible
numeric spaces and the loss cannot be minimized.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import Preprocessor
from src.synth import SynthGenerator


class SyntheticMixtureDataset(Dataset):
    """Generates synthetic (A, B) pairs on the fly.

    Each call to __getitem__ samples fresh weights and mixes the
    monoenergetic spectra, applies the preprocessors, and returns tensors.

    Parameters
    ----------
    generator         : SynthGenerator
    input_pre         : Preprocessor for the network input (detector A)
    target_pre        : Preprocessor for the target (detector B); must produce a
                        plain normalized density to match the softmax head
    samples_per_epoch : number of samples (defines dataset length)
    base_seed         : base RNG seed; item index is added for reproducibility
    """

    def __init__(
        self,
        generator: SynthGenerator,
        input_pre: Preprocessor,
        target_pre: Preprocessor,
        samples_per_epoch: int = 1024,
        base_seed: int = 0,
    ) -> None:
        self.generator = generator
        self.input_pre = input_pre
        self.target_pre = target_pre
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + idx)
        A_mix, B_mix = self.generator.sample(rng)

        A_t, _ = self.input_pre.transform(A_mix)
        B_t, _ = self.target_pre.transform(B_mix)

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
    input_pre         : Preprocessor for the network input (detector A)
    target_pre        : Preprocessor for the target (detector B)
    n_samples         : number of samples to generate
    seed              : fixed seed for reproducibility
    """

    def __init__(
        self,
        generator: SynthGenerator,
        input_pre: Preprocessor,
        target_pre: Preprocessor,
        n_samples: int = 256,
        seed: int = 999,
    ) -> None:
        self.input_pre = input_pre
        self.target_pre = target_pre
        rng = np.random.default_rng(seed)

        inputs = []
        targets = []
        for _ in range(n_samples):
            A_mix, B_mix = generator.sample(rng)
            A_t, _ = input_pre.transform(A_mix)
            B_t, _ = target_pre.transform(B_mix)
            inputs.append(A_t)
            targets.append(B_t)

        # (n_samples, 1, N)
        self.inputs = torch.from_numpy(np.stack(inputs)).unsqueeze(1)
        self.targets = torch.from_numpy(np.stack(targets)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input": self.inputs[idx], "target": self.targets[idx]}
