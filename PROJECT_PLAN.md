# Detector Spectrum Translation — 1D U-Net Baseline

> **Goal for Claude Code**: Scaffold a complete, runnable Python project on GitHub that
> trains a **1D U-Net** to predict one detector's energy-deposition spectrum from
> another's, using a **combined MSE + EMD (Earth Mover's Distance) loss**. This document
> is the full specification. Build the repository exactly as described, then run the
> smoke test in the final section to confirm it works end-to-end.

---

## 1. Problem statement

We have paired energy-deposition spectra from two simulated detectors (`SiC_SBD` and
`TEPC`) observing the same radiation field. Each spectrum is a histogram of counts over a
**shared, fixed, log-spaced energy grid**. We want to learn a deterministic mapping:

```
spectrum_A (input)  ->  spectrum_B (target)
```

so that, given a new detector-A spectrum, we can predict the corresponding detector-B
spectrum. This baseline treats the whole spectrum as a 1D signal and maps it through a
1D U-Net (sequence-to-sequence, same length in = length out).

**Direction**: Make the mapping direction configurable. Default `SiC_SBD -> TEPC`, but a
single config flag must allow `TEPC -> SiC_SBD`. Do not hard-code the direction.

### 1.1 Training data vs. the real deployment target — read this first

This is the defining feature of the project and shapes the whole design:

- **Training data**: **11 mono-energetic proton spectra** at 200, 300, 400, 500, 600, 800,
  1000, 2000, 4000, 7000, and 10000 MeV. Each is a file with the same
  `keV / SiC_SBD / TEPC` layout on the same energy grid. Each gives **one A->B pair**, so we
  have **11 real pairs**.
- **Real deployment target**: a **Galactic Cosmic Ray (GCR) spectrum** (`GCR_spectrum`),
  which is a **broad continuum** — physically a fluence-weighted superposition over a wide
  energy range, not a single peak. The end goal is: *given detector-A's GCR energy-
  deposition spectrum, predict detector-B's GCR energy-deposition spectrum.*

**The core generalization risk**: the model is trained on 11 **narrow, peaked** spectra
but must be applied to a **broad, smooth** spectrum. A naive model can overfit the shape
"sharp peak in, sharp peak out" and fail on the continuum. Section 6 addresses this
directly with a physically exact data-synthesis strategy. Treat solving this as a
first-class objective, not an afterthought.

### 1.2 Why these spectra are hard to relate analytically

- The spectra follow a **Gaussian-Vavilov-Landau** family of shapes (asymmetric, heavy
  high-energy tail) with no clean closed form, which is why we use a learned model.
- The two detectors have **different response functions**: in the 200 MeV sample the
  SiC_SBD peak sits near 27 keV while the TEPC peak sits near 8.6 keV, and TEPC is broader
  with a longer tail. The peak positions and widths also **shift with primary energy**
  across the 11 files. The network must learn a non-trivial, energy-dependent reshaping.

---

## 2. Data specification

### 2.1 The files

Place all training files under `data/raw/` and the deployment target under
`data/deploy/`. Confirmed structure (verified on the 200 MeV sample; all files share it):

- **Sheet**: `Sheet1`.
- **Row 1** is a header band; the real column headers (`keV`, `SiC_SBD`, `TEPC`) sit on the
  second row. **Read with `skiprows=1` and assign names explicitly** — do not trust auto-
  detected headers. The first spreadsheet column is an empty index column.
- Columns after cleaning:
  - `keV` — bin energy (bin centers), **360 log-spaced points** from `0.01` keV to
    `~9623` keV (~6 decades; successive-bin ratio ~ 1.039).
  - `SiC_SBD` — detector-A counts (integer-valued).
  - `TEPC` — detector-B counts (integer-valued).
- After `skiprows=1`, coerce all three columns to numeric and drop fully-empty rows; you
  should be left with **360 rows**.

**File naming / energy parsing**: the 11 training files encode primary energy in the
filename (e.g. `200MeV.xlsx`, `10000MeV.xlsx`). Parse the energy from the filename with a
robust regex and **carry it as metadata** for each pair — it is needed for the synthetic
GCR mixing in Section 6 and is a candidate auxiliary input later (Section 13).

**The GCR file differs from the training files — handle explicitly, do not assume
identical layout** (verified from `GCR_spectrum.xlsx`):

| Property            | 11 training files          | `GCR_spectrum.xlsx`                 |
| ------------------- | -------------------------- | ----------------------------------- |
| Column headers      | `keV`, `SiC_SBD`, `TEPC`   | `E(keV)`, `TEPC`, `SiC`             |
| Column **order**    | energy, SiC, TEPC          | energy, **TEPC, SiC** (A/B swapped) |
| Detector-A name     | `SiC_SBD`                  | `SiC` (same detector, short name)   |
| Number of bins      | 360                        | **420**                             |
| Energy start        | 0.01000 keV                | 0.01019 keV (~0.8% offset)          |
| Energy max          | 9,623 keV                  | **98,099 keV** (one extra decade)   |
| Geometric bin step  | x1.039 (log-step 0.016667) | x1.039 (identical step)             |

The loader must **map columns by meaning, not position**: detect detector-A as
`SiC_SBD`-or-`SiC` and detector-B as `TEPC`, and the energy column as `keV`-or-`E(keV)`,
case/whitespace-insensitive. Provide a small alias map so both layouts load into the same
canonical `{energy, SiC_SBD, TEPC}` dict. **Both detector columns are present in the GCR
file, so the real GCR A->B pair has ground truth** — the headline GCR metric in Section 8
is computable.

### 2.2 Loader contract (`src/data.py`)

```python
def load_spectrum_file(path: str) -> dict:
    """Return {'energy': (N,), 'SiC_SBD': (N,), 'TEPC': (N,), 'energy_MeV': float|None}.
    - Reads sheet with skiprows=1; coerce numeric; dropna(how='all').
    - Maps columns BY ALIAS, not position: energy in {'kev','e(kev)'},
      detector-A in {'sic_sbd','sic'}, detector-B in {'tepc'} (case/space-insensitive).
    - Parses primary energy (MeV) from filename; None for GCR_spectrum.
    - For training files: asserts energy grid == canonical grid (see 2.3).
    - For the GCR file (different grid): returns its native grid; caller resamples to
      canonical via resample_to_canonical() before inference.
    - Does NOT normalize here.
    """
```

### 2.3 Energy grid — canonical training grid + GCR resampling

The 11 training files share the **same** 360-bin energy axis. Treat this as the
**canonical grid**: persist it once to `data/processed/energy_grid.npy`, assert every
*training* file conforms to it on load, and use it everywhere (training, EMD bin
distances, plotting, GCR mixing).

**The GCR file is on a different grid** (420 bins, ~0.8% energy offset, extends one decade
higher) but with the **identical geometric step** (x1.039). Because the step matches and
the offset is tiny, resample the GCR spectrum onto the canonical 360-bin grid:

- Interpolate **in log-energy** (the natural axis here). Interpolating the **normalized
  density** is cleanest; if interpolating counts, re-normalize after.
- The GCR mass sits at 3-10 keV (SiC peak ~9.8 keV, TEPC peak ~3.5 keV), far below the
  canonical grid's 9,623 keV ceiling, so **truncating the GCR's extra high-energy decade
  loses negligible mass** — but log the discarded fraction as a sanity check and warn if it
  exceeds a small threshold (e.g. 1e-4).
- Implement `resample_to_canonical(energy, values, canonical_grid)` and unit-test it
  (interpolating a grid onto itself is the identity; total mass is preserved within
  tolerance).

Never resample the *training* files — they already match the canonical grid. Resampling is
GCR-only.

---

## 3. Preprocessing (`src/preprocessing.py`)

Three transforms, in this order, each individually unit-tested:

1. **Normalize to a probability density.** Divide each spectrum by its total counts so it
   sums to 1. This is the physically meaningful representation and makes EMD well-defined.
   **Store the original total counts** so predictions can be rescaled to absolute counts.

2. **Log-compression of the y-axis (recommended, configurable).** Counts span many orders
   of magnitude. Apply `y -> log1p(scale * y)` (config `log_scale`, e.g. `1e4`) so the
   network sees the tail, not just the peak. Provide the exact inverse `expm1(y)/scale`.
   Toggle via `preprocessing.log_compress`.

3. The bins are already log-spaced; the network operates on the **bin index** (0..359), so
   no resampling is needed. Document that "position along the sequence" = "log energy",
   which matters for the EMD term (Section 5.2).

Provide a `Preprocessor` class with `.transform()` / `.inverse_transform()` so tests can
assert round-trip identity to floating-point tolerance.

---

## 4. Model: 1D U-Net (`src/model.py`)

A compact 1D U-Net sized for short (360-bin) signals:

- **Input**: `(batch, 1, 360)` (detector-A density). *(Optionally `(batch, 2, 360)` later
  if a primary-energy channel is added — see Section 13. Keep `in_ch` configurable.)*
- **Output**: `(batch, 1, 360)` (detector-B density).
- **Encoder**: 3-4 down blocks, each `Conv1d -> Norm -> act -> Conv1d -> Norm -> act` then
  downsample. Widths e.g. `32 -> 64 -> 128 -> 256`.
- **Bottleneck**: one conv block at deepest width.
- **Decoder**: mirror with upsampling; **concatenate skip connections** from matching
  encoder levels (preserves peak position and tail detail).
- **Downsampling**: `MaxPool1d(2)` or strided conv. **Length 360** = 2^3 * 45 -> three /2
  steps give 45 (fine). For a 4th level, pad/crop so concat shapes line up; implement and
  unit-test a center-crop/pad helper guaranteeing `len(out) == len(in) == 360`.
- **Norm/activation**: `GroupNorm` + `GELU` (stable at small batch sizes).
- **Output head**: **prefer softmax across the energy axis** so the prediction is a proper
  non-negative, normalized density — this keeps the EMD term clean. Alternative
  `softplus + renormalize`. Make it a config option.

Pure PyTorch. Expose `UNet1D(in_ch=1, out_ch=1, base=32, depth=3, head='softmax')`.

---

## 5. Loss function: MSE + EMD (`src/losses.py`)

```
L = w_mse * MSE(pred, target) + w_emd * EMD(pred, target)
```

config `w_mse`, `w_emd` (start `1.0`/`1.0`, then tune).

### 5.1 MSE term
Mean-squared error between predicted and target spectra **in the transformed space** the
network outputs. Pins down per-bin amplitude.

### 5.2 EMD term (the important one)
MSE alone is bin-local and tends to **smear peaks and flatten the Landau tail**. EMD
(1-Wasserstein) compares *distributions*, penalizing mass at the wrong energy. For **1D
distributions on an ordered axis** it has a closed form — no OT solver:

```
EMD(p, q) = sum_i | CDF_p(i) - CDF_q(i) | * d_i
```

`CDF` = cumulative sum along energy; `d_i` = adjacent-bin distance. Requirements:

- Compute on **normalized, non-negative** spectra (softmax head makes this automatic;
  otherwise renormalize inside the loss).
- Configurable bin distance `emd_space`:
  - `"index"` -> `d_i = 1` (uniform; distance in bins / log-energy steps). **Default.**
  - `"log_energy"` -> `d_i = delta(log10 E)` (~ constant since bins are geometric).
  - `"linear_energy"` -> `d_i = delta E` keV. **Caution**: 6-decade span lets the tail
    dominate; expose but do not default.
- Fully **differentiable** (cumsum + abs + weighted sum). Verify with a gradient test.

Provide `emd_1d(pred, target, bin_dist)` and `SpectrumLoss(nn.Module)` returning
`(total, {'mse':..., 'emd':...})` for per-component logging.

---

## 6. The data strategy — mono-energetic training, GCR deployment

**This section is the heart of the project.** We have only 11 real pairs and must
generalize to a broad GCR continuum. The solution exploits a physical fact:

> **Detector response is linear in particle fluence.** A detector's energy-deposition
> spectrum from a mixed field equals the fluence-weighted sum of its responses to the
> individual mono-energetic components. This holds **independently for each detector**, so
> a weighted sum of detector-A mono spectra and the *same* weighted sum of detector-B mono
> spectra remain a valid, physically consistent A->B pair.

This gives us an essentially unlimited, **physically exact** supply of broad training
spectra that resemble GCR.

### 6.1 Synthetic broad-spectrum generation (`src/synth.py`)

Generate training pairs by mixing the 11 mono-energetic spectra:

- Draw a random non-negative weight vector `w` over the (training) energies (e.g.
  Dirichlet, or log-uniform magnitudes then normalize). Build:
  - `A_mix = sum_k w_k * A_k` (detector-A), `B_mix = sum_k w_k * B_k` (detector-B), using
    the **same** `w`. Counts add linearly **before** normalization; normalize the resulting
    pair afterward.
- **Sample weights to span from near-mono to broad-continuum**, including:
  - GCR-like smooth power-law / modulated shapes over the energies (use realistic GCR
    differential-flux weightings — e.g. a power law in energy with a solar-modulation-like
    rolloff — as one family of weight vectors so synthetic spectra resemble the true
    target);
  - random sparse mixtures (2-4 active energies) and dense mixtures (all energies);
  - the pure mono spectra themselves (one-hot weights) so the endpoints stay in-domain.
- Optionally add **Poisson resampling** per bin (`Poisson(lambda=counts)`) to mimic
  simulation statistics, applied independently per detector.

**Why this is the right call**: it directly closes the train/deploy gap. The model sees
broad, GCR-shaped inputs during training that are *guaranteed physically consistent* with
their detector-B targets, instead of only sharp peaks.

### 6.2 Train / validation / test split — avoid leakage and prove generalization

- **Hold out whole mono-energies, not just mixtures.** Reserve **2-3 of the 11 energies**
  (e.g. 600, 2000, 7000 MeV — spread across the range) as a **held-out set used neither as
  mixing components in training nor individually in training**. Build a **validation/test
  set of synthetic mixtures from the held-out energies only**, plus the held-out pure mono
  spectra themselves. This tests interpolation to unseen primary energies — the closest
  proxy we have for generalizing to the real GCR field.
- Make the held-out energy list and the train/val/test mixture counts config options.
- Keep a clear separation in code between "real pairs", "synthetic training mixtures", and
  "held-out evaluation".

### 6.3 The real GCR spectrum

`GCR_spectrum.xlsx` is the **final deployment input**, used at inference only. **Confirmed:
it contains both detector columns** (`SiC` = detector-A, `TEPC` = detector-B), so there is
**ground truth** for the prediction.

- First **resample the GCR spectrum onto the canonical 360-bin grid** (Section 2.3), using
  the alias-mapped columns.
- Use detector-A (`SiC`) as input, predict detector-B (`TEPC`), and report the full metric
  suite against the true `TEPC` GCR column. **This is the decisive, real-world test — make
  it the headline result.**
- Never use the GCR file in training or for synthetic mixing.

---

## 7. Training pipeline (`src/train.py`)

- **Config-driven** via `configs/baseline.yaml` (OmegaConf or YAML + dataclass). Every knob
  above lives here.
- **Reproducibility**: set/log all seeds; record resolved config + git commit into the run
  dir.
- **Dataset**: an iterable/`Dataset` that yields synthetic mixtures on the fly from the
  training-energy pool (Section 6.1), with `samples_per_epoch` configurable;
  validation/test use a **fixed, seeded** set of held-out mixtures so metrics are
  comparable across runs.
- **Optimizer/schedule**: AdamW + cosine (or plateau) LR; gradient clipping; optional AMP.
- **Logging**: TensorBoard/CSV of `total/mse/emd` for train and val each epoch.
- **Checkpointing**: best-by-val-total -> `runs/<ts>/best.pt`, plus `last.pt`.
- **`--fast-dev-run`**: 2 batches x 1 epoch end-to-end for CI/smoke test.

---

## 8. Evaluation (`src/evaluate.py`)

Report on the held-out set **and** on the real GCR pair (if ground truth present), in the
**original count space** (invert preprocessing first):

- **EMD / 1-Wasserstein** (primary — what we optimize, best reflects spectral-shape
  fidelity).
- **Peak-position error**: argmax-energy difference and 50%-cumulative-mass (median) energy
  difference, in keV.
- **Tail fidelity**: predicted vs. true integrated counts above a high-energy threshold
  (e.g. > 90th-percentile energy) — checks the Landau tail survives.
- **Per-bin MSE / MAE** (secondary).
- **Total-counts recovery** after rescaling by stored normalization.
- A dedicated **`evaluate_gcr()`** that runs the full pipeline on `GCR_spectrum.xlsx` and
  reports the above against the true detector-B GCR spectrum when available. **This is the
  headline number.**

### 8.1 Plots (`src/plots.py`)
Per evaluated pair: input A, true B, predicted B overlaid on **log-log axes**; the residual
(pred - true); and the CDFs of true vs. predicted B (makes EMD visual). Produce these for
the held-out mono energies and, prominently, for the GCR case.

---

## 9. Repository layout

```
detector-spectrum-unet/
|-- README.md
|-- LICENSE                   # MIT unless user says otherwise
|-- pyproject.toml            # pin torch, numpy, pandas, openpyxl, pyyaml/omegaconf,
|                             #   matplotlib, tensorboard, pytest
|-- .gitignore                # venvs, runs/, data/processed/, __pycache__, *.pt
|-- configs/
|   `-- baseline.yaml
|-- data/
|   |-- raw/                  # the 11 *MeV.xlsx training files
|   |-- deploy/               # GCR_spectrum.xlsx (inference only)
|   `-- processed/            # energy_grid.npy, cached mono arrays, fixed eval mixtures
|-- src/
|   |-- __init__.py
|   |-- data.py               # load_spectrum_file (alias cols), resample_to_canonical
|   |-- preprocessing.py      # Preprocessor (transform + inverse)
|   |-- synth.py              # weighted mixing of mono spectra -> synthetic pairs (CORE)
|   |-- dataset.py            # SyntheticMixtureDataset (train), FixedMixtureSet (val/test)
|   |-- model.py              # UNet1D
|   |-- losses.py             # emd_1d, SpectrumLoss (MSE+EMD)
|   |-- train.py
|   |-- evaluate.py           # held-out metrics + evaluate_gcr()
|   `-- plots.py
|-- scripts/
|   |-- prepare_data.py       # xlsx -> processed arrays + energy_grid.npy
|   |-- run_train.py
|   |-- run_eval.py
|   `-- run_gcr.py            # apply trained model to GCR_spectrum.xlsx
|-- tests/
|   |-- test_data.py          # alias col mapping; canonical-grid assert; resample identity
|   |-- test_preprocessing.py # transform/inverse round-trip
|   |-- test_synth.py         # linearity: mix(A,w) & mix(B,w) consistent; one-hot == mono
|   |-- test_losses.py        # emd_1d correctness on toy dists + differentiability
|   `-- test_model.py         # in/out shape == (B,1,360), no NaNs, softmax sums to 1
`-- .github/
    `-- workflows/ci.yml      # ruff + pytest + fast-dev-run on push
```

---

## 10. Config file (`configs/baseline.yaml`) — required keys

```yaml
seed: 42
data:
  raw_glob: "data/raw/*MeV.xlsx"
  gcr_file: "data/deploy/GCR_spectrum.xlsx"
  sheet_skiprows: 1
  direction: "SiC_SBD->TEPC"          # or "TEPC->SiC_SBD"
  heldout_energies_MeV: [600, 2000, 7000]   # excluded from training & training-mixtures
preprocessing:
  normalize_to_density: true
  log_compress: true
  log_scale: 1.0e4
synth:
  weight_prior: dirichlet              # dirichlet | loguniform | gcr_powerlaw
  gcr_powerlaw_index: -2.7             # used when weight_prior includes gcr-like family
  mixture_families: [mono, sparse, dense, gcr_like]
  poisson_noise: true
model:
  arch: unet1d
  in_ch: 1
  base_channels: 32
  depth: 3
  norm: group
  activation: gelu
  head: softmax                        # softmax | softplus_renorm
loss:
  w_mse: 1.0
  w_emd: 1.0
  emd_space: index                     # index | log_energy | linear_energy
train:
  samples_per_epoch: 1024
  val_mixtures: 256                    # fixed/seeded, from held-out energies only
  test_mixtures: 256
  epochs: 300
  batch_size: 64
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine
  grad_clip: 1.0
  amp: true
eval:
  tail_percentile: 90
```

---

## 11. README must include

- Problem framing: detector spectrum-to-spectrum translation; trained on 11 mono-energetic
  proton spectra (200-10000 MeV); **deployed on a broad GCR spectrum**.
- The **linearity-of-response** insight and how synthetic mixing (`src/synth.py`) bridges
  the mono->GCR gap. Make this the centerpiece of the README.
- Install + quickstart:
  ```bash
  pip install -e .
  python scripts/prepare_data.py
  python scripts/run_train.py --config configs/baseline.yaml
  python scripts/run_eval.py  --config configs/baseline.yaml --ckpt runs/<ts>/best.pt
  python scripts/run_gcr.py   --config configs/baseline.yaml --ckpt runs/<ts>/best.pt
  ```
- A **caveat box**: the mono->GCR generalization relies on response linearity holding in
  the simulation; the held-out-energy evaluation is a proxy, and the **true GCR pair is the
  decisive test**. With only 11 real anchor energies, validate the GCR result carefully
  before trusting it.
- "What to tune first": `w_emd` vs `w_mse`; `synth.weight_prior` / `mixture_families`;
  `log_compress` on/off; held-out energy choice; model `depth`.

---

## 12. Acceptance / smoke test (run before declaring done)

1. `pip install -e .` succeeds.
2. `pytest -q` passes — in particular:
   - `emd_1d` = 0 for identical dists, known value for a one-bin shift, gradients flow;
   - preprocessing transform->inverse round-trips < 1e-6;
   - **`test_data`**: alias column mapping loads both layouts into canonical form;
     `resample_to_canonical` is identity on the canonical grid and preserves total mass;
   - **`test_synth`**: a one-hot weight reproduces the corresponding mono pair exactly, and
     mixing is linear and applied with the *same* weights to both detectors;
   - model output `(B,1,360)`, no NaNs, softmax head sums to 1.
3. `python scripts/prepare_data.py` loads all 11 files, writes `energy_grid.npy` (len 360),
   and reports parsed energies `[200,...,10000]`.
4. `run_train.py --fast-dev-run` completes, writes a checkpoint, logs decreasing total loss.
5. `run_eval.py` produces metric JSON + overlay/CDF plots for held-out energies.
6. `run_gcr.py` loads `GCR_spectrum.xlsx` (alias columns), **resamples it onto the
   canonical 360-bin grid**, runs the model on detector-A (`SiC`), and reports the full
   metric suite against the true detector-B (`TEPC`) GCR spectrum, plus the overlay/CDF
   plot. This is the headline result.

When all pass, commit and push with conventional messages; tag `v0.1.0-baseline`.

---

## 13. Next iterations (explicit non-goals for this baseline)

- **Primary-energy / mixture-weight conditioning**: feed the known energy (mono) or weight
  vector (synthetic) as an auxiliary input channel or FiLM conditioning. Keep `in_ch`
  configurable so this drops in later.
- No GAN / diffusion / Transformer yet — those come **after** the U-Net + metric harness is
  trustworthy.
- No absolute-calibration physics beyond normalize/denormalize round-trips.
- If response linearity is found **not** to hold in the simulation (check via held-out-
  energy EMD), revisit the synthetic-mixing assumption before scaling the approach.
