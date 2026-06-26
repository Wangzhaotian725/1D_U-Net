#!/usr/bin/env python3
"""Evaluate trained model on GCR spectrum."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate_gcr
from src.model import UNet1D
from src.plots import plot_cdf_comparison, plot_spectrum_comparison
from src.preprocessing import Preprocessor, build_preprocessors


def save_plot_data(out_dir: Path, energy, sic, tepc_true, tepc_pred) -> None:
    """Save the raw data used to generate the two GCR plots as CSV files.

    gcr_spectrum_comparison.csv  — four columns used in the log-log overlay plot
    gcr_cdf_comparison.csv       — three columns used in the CDF plot
    """
    cdf_true = np.cumsum(tepc_true)
    cdf_pred = np.cumsum(tepc_pred)
    if cdf_true[-1] > 0:
        cdf_true = cdf_true / cdf_true[-1]
    if cdf_pred[-1] > 0:
        cdf_pred = cdf_pred / cdf_pred[-1]

    spectrum_df = pd.DataFrame({
        "energy_keV":       energy,
        "SiC_input":        sic,
        "TEPC_true":        tepc_true,
        "TEPC_predicted":   tepc_pred,
    })
    spectrum_path = out_dir / "gcr_spectrum_comparison.csv"
    spectrum_df.to_csv(str(spectrum_path), index=False)
    print(f"Saved spectrum data to {spectrum_path}")

    cdf_df = pd.DataFrame({
        "energy_keV":       energy,
        "CDF_true":         cdf_true,
        "CDF_predicted":    cdf_pred,
    })
    cdf_path = out_dir / "gcr_cdf_comparison.csv"
    cdf_df.to_csv(str(cdf_path), index=False)
    print(f"Saved CDF data to {cdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on GCR spectrum")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--out-dir", default="results/gcr")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical grid
    energy_grid = np.load("data/processed/energy_grid.npy")

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

    input_pre, target_pre = build_preprocessors(cfg)

    results = evaluate_gcr(model, cfg, input_pre, energy_grid, target_pre=target_pre)

    # Separate arrays from scalar metrics
    arrays = {k: v for k, v in results.items() if k.startswith("_")}
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}

    print("GCR metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    with open(str(out_dir / "gcr_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir}/gcr_metrics.json")

    # Extract arrays
    energy    = arrays["_energy_grid"]
    sic       = arrays["_sic"]
    tepc_true = arrays["_tepc_true"]
    tepc_pred = arrays["_tepc_pred"]

    # Save CSV data for both plots
    save_plot_data(out_dir, energy, sic, tepc_true, tepc_pred)

    # Generate plots
    plot_spectrum_comparison(
        energy, sic, tepc_true, tepc_pred,
        title="GCR spectrum: SiC -> TEPC",
        out_path=str(out_dir / "gcr_spectrum_comparison.png"),
    )
    plot_cdf_comparison(
        energy, tepc_true, tepc_pred,
        title="GCR TEPC CDF comparison",
        out_path=str(out_dir / "gcr_cdf_comparison.png"),
    )
    print(f"Saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
