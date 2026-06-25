#!/usr/bin/env python3
"""Prepare processed data files from raw Excel spectra."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_spectrum_file


def main() -> None:
    raw_glob = "data/raw/*MeV.xlsx"
    files = sorted(glob.glob(raw_glob))
    if not files:
        print(f"ERROR: No files found matching {raw_glob}")
        sys.exit(1)

    print(f"Found {len(files)} raw files:")
    for f in files:
        print(f"  {f}")

    all_A = []
    all_B = []
    all_E = []
    energy_grid = None

    for f in files:
        spec = load_spectrum_file(f, skiprows=1)
        e_MeV = spec["energy_MeV"]
        if e_MeV is None:
            print(f"  SKIP {f}: could not parse energy from filename")
            continue

        if energy_grid is None:
            energy_grid = spec["energy"]
            print(f"\nCanonical energy grid: {len(energy_grid)} bins, "
                  f"{energy_grid[0]:.4f}–{energy_grid[-1]:.2f} keV")
        else:
            if not np.allclose(spec["energy"], energy_grid, rtol=1e-4):
                print(f"  WARNING: {f} has different energy grid!")

        all_A.append(spec["SiC_SBD"])
        all_B.append(spec["TEPC"])
        all_E.append(e_MeV)
        sic_sum = spec["SiC_SBD"].sum()
        tepc_sum = spec["TEPC"].sum()
        print(f"  {Path(f).name:20s}  {e_MeV:7.0f} MeV  "
              f"SiC sum={sic_sum:.3e}  TEPC sum={tepc_sum:.3e}")

    if energy_grid is None:
        print("ERROR: No valid files loaded.")
        sys.exit(1)

    # Sort by energy
    order = np.argsort(all_E)
    mono_A = np.stack(all_A)[order]
    mono_B = np.stack(all_B)[order]
    energies_MeV = np.array(all_E)[order]

    print(f"\nParsed energies (MeV): {list(energies_MeV)}")

    # Save
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "energy_grid.npy"), energy_grid)
    np.savez(
        str(out_dir / "mono_spectra.npz"),
        mono_A=mono_A,
        mono_B=mono_B,
        energies_MeV=energies_MeV,
    )

    print(f"\nSaved:")
    print(f"  {out_dir}/energy_grid.npy  shape={energy_grid.shape}")
    print(f"  {out_dir}/mono_spectra.npz  mono_A={mono_A.shape}, mono_B={mono_B.shape}")


if __name__ == "__main__":
    main()
