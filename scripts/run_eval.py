#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_spectrum_file
from src.dataset import FixedMixtureSet
from src.evaluate import evaluate_on_set
from src.model import UNet1D
from src.plots import plot_cdf_comparison, plot_spectrum_comparison
from src.preprocessing import Preprocessor, build_preprocessors
from src.synth import SynthGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--out-dir", default="results/eval")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    processed_dir = Path("data/processed")
    energy_grid = np.load(str(processed_dir / "energy_grid.npy"))
    data = np.load(str(processed_dir / "mono_spectra.npz"))
    mono_A = data["mono_A"]
    mono_B = data["mono_B"]
    energies_MeV = list(data["energies_MeV"])

    # Normalize
    for i in range(len(mono_A)):
        s = mono_A[i].sum()
        if s > 0:
            mono_A[i] /= s
        s = mono_B[i].sum()
        if s > 0:
            mono_B[i] /= s

    heldout = list(cfg.data.heldout_energies_MeV)
    heldout_set = set(heldout)
    held_idx = [i for i, e in enumerate(energies_MeV) if e in heldout_set]

    if not held_idx:
        print("Warning: no heldout energies found, using all energies for test set")
        held_idx = list(range(len(energies_MeV)))

    families = list(cfg.synth.mixture_families)
    test_gen = SynthGenerator(
        mono_A[held_idx],
        mono_B[held_idx],
        np.array([energies_MeV[i] for i in held_idx]),
        families=families,
        poisson_noise=cfg.synth.poisson_noise,
        gcr_powerlaw_index=cfg.synth.gcr_powerlaw_index,
    )

    input_pre, target_pre = build_preprocessors(cfg)

    test_ds = FixedMixtureSet(
        test_gen, input_pre, target_pre,
        n_samples=cfg.train.test_mixtures, seed=cfg.seed + 2,
    )

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = UNet1D(
        in_ch=cfg.model.in_ch,
        out_ch=1,
        base=cfg.model.base_channels,
        depth=cfg.model.depth,
        head=cfg.model.head,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate_on_set(model, test_ds, target_pre, energy_grid, cfg)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    with open(str(out_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir}/metrics.json")

    # Plot first sample
    sample = test_ds[0]
    x_np = sample["input"].squeeze().numpy()
    y_np = sample["target"].squeeze().numpy()

    with torch.no_grad():
        pred = model(sample["input"].unsqueeze(0)).squeeze().numpy()

    plot_spectrum_comparison(
        energy_grid, x_np, y_np, pred,
        title="Test sample spectrum comparison",
        out_path=str(out_dir / "spectrum_comparison.png"),
    )
    plot_cdf_comparison(
        energy_grid, y_np, pred,
        title="Test sample CDF comparison",
        out_path=str(out_dir / "cdf_comparison.png"),
    )
    print(f"Saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
