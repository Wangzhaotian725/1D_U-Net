# Experiment Summary: v0.6 — Skip-Connection Leakage Hypothesis

**Date**: 2026-06-26
**Branch**: `claude/experiment-v06`
**Status**: FAILED — worst GCR result in project history

---

## 1. Version History (GCR Metrics)

| Version | Architecture | EMD ↓ | MSE ↓ | MAE ↓ | Peak Err (bins) ↓ | Notes |
|---------|-------------|-------|-------|-------|-------------------|-------|
| v0.1 | UNet1D | — | — | — | — | Log-head bug |
| v0.2 | UNet1D | — | — | — | — | GCR-contaminated synth |
| v0.3 | UNet1D | — | — | — | — | Overfit |
| v0.4 | UNet1D (softmax) | 5.30 | 3.73e-5 | 2.01e-3 | **8** | Best to date (at v0.5 start) |
| v0.5 | UNet1D (softplus_renorm) | 6.32 | — | — | 6 (held-out) | EMD regressed with 16 energies |
| **v0.6A** | TransferOperator | ~25.8 (selection) | — | — | ~17–22 (held-out) | Failed: still bimodal, overfit |
| **v0.6B** | SpectrumTransformer | **21.83** | **1.67e-4** | **4.54e-3** | **24** | **Catastrophic failure** |

v0.6B is the worst GCR result recorded. Every metric is 3–4× worse than v0.4.

---

## 2. The v0.6 Hypothesis (Stated)

The double-peak artefact observed in v0.4/v0.5 predictions was hypothesised to be caused by U-Net skip connections leaking the input SiC peak position into the output. Evidence:

- v0.4/v0.5: input SiC peak at bin 179 (~9.6 keV); secondary predicted TEPC peak at bin 175 (~8.3 keV) — only 4 bins apart
- True TEPC peak at bin 152 (~3.4 keV)

The hypothesis was mechanistically plausible. The proposed fix: remove skip connections entirely, using architectures that cannot leak spatially.

**The hypothesis was partially correct about the mechanism but catastrophically wrong about the remedy.**

---

## 3. What Happened in v0.6

### 3A. TransferOperator (17,280 parameters)

- Band-limited (±60 bins), column-normalised transfer matrix
- Held-out best selection_score: 25.79 (peak_err ~17–22, SPR ~0.65–0.99)
- **Result**: Persistent bimodal output. A column-normalised linear operator applied to multimodal mixture inputs can still produce bimodal outputs — the constraint only ensures mass conservation, not unimodality. Severely underparameterised for the task.

### 3B. SpectrumTransformer (839,553 parameters)

- 4-layer Transformer encoder, no skip connections, learnable positional encoding (n_bins=360)
- Held-out metrics appeared promising: selection_score 5.81, peak_err 3.84, SPR ~0.55
- **GCR result**: EMD=21.83, peak_pos_error=24 — catastrophic

**GCR failure modes visible in the plots:**

1. **Flat low-energy plateau**: Predicted TEPC shows ~10⁻⁵ density from 10⁻² to 1 keV. The true TEPC is essentially zero in this region. The Transformer distributes mass uniformly across low-energy bins it never learned to suppress.

2. **Peak shifted to higher energy**: Predicted peak around 10–50 keV; true TEPC peak is around 3–5 keV. The CDF comparison shows the predicted distribution rising much later than the true. This is the OPPOSITE direction from the v0.4/v0.5 double-peak artefact.

3. **Noisy / jagged high-energy tail**: The attention mechanism produces irregular, spiky output at high energies. Without spatial inductive bias, the model has no preference for smooth outputs.

4. **Massive train/val gap**: Train loss 0.75, val loss 30 at convergence. The Transformer severely overfit the mixture training distribution (839K params on ~16 monoenergetic basis spectra).

---

## 4. Root Cause Analysis

### 4.1 The Spatial Inductive Bias Was the Feature, Not the Bug

The U-Net skip connections that were hypothesised to cause the double-peak artefact also provide something critical: **spatial locality**. When a U-Net decoder receives skip features from a given encoder depth, it naturally attends to the energy region where the input had structure. This allows the model to generalise — a GCR input with a smooth, broadband spectrum maps to a TEPC output that is also centred in the right energy region.

The SpectrumTransformer has no such prior. Its self-attention is fully position-agnostic for inputs it has not seen. The GCR spectrum is structurally unlike any training mixture (it is a smooth power-law-like input spanning decades), so the learned attention patterns generalise poorly. The model effectively predicts a "default" output pattern rather than one informed by the GCR input structure.

### 4.2 The Double-Peak Was a Secondary Problem

Retrospectively:
- v0.4 EMD=5.30, peak_err=8 bins, double-peak visible but primary peak near correct position (~bin 152)
- v0.6B EMD=21.83, peak_err=24 bins, no double-peak but primary peak completely wrong (~bin 176+)

**Eliminating the double-peak artefact at the cost of 4× worse EMD and 3× worse peak error is not a win.** The v0.4 U-Net prediction, despite the secondary peak, captured the primary TEPC distribution far more accurately. The double-peak was a cosmetic artefact; the v0.6B Transformer prediction has a fundamentally wrong spectral shape.

### 4.3 Out-of-Distribution Generalisation

The training data consists of synthetic mixtures of 16 monoenergetic SiC spectra. Each monoenergetic SiC spectrum has a distinct, roughly Gaussian shape. Their mixtures are also relatively structured. The GCR SiC spectrum is a broad, smooth, power-law-dominated distribution that is unlike any training mixture.

- **U-Net**: skip connections provide a strong implicit regulariser — "put spectral mass near where the input has mass." This helps OOD generalisation.
- **Transformer**: no such regulariser. The model must learn the input→output mapping entirely from data. With only 16 basis functions, it overfit to the specific shapes of mixtures of those bases.

### 4.4 Positional Encoding Mismatch

The SpectrumTransformer uses a learned positional encoding of fixed length n_bins=360. The GCR spectrum was presumably evaluated at 360 bins (matching training), so this is not the proximate cause. However, the learned positional encoding encodes *where in the training distribution* each bin tends to appear, which may not align with the GCR spectrum's energy allocation.

---

## 5. Lessons Learned

| Lesson | Implication |
|--------|-------------|
| Skip connections provide essential spatial inductive bias for OOD generalisation | Do not remove them; instead, selectively suppress their content |
| The double-peak artefact was an additive problem, not a multiplicative one | Suppress it with regularisation rather than architectural surgery |
| Model capacity ≫ effective training diversity → overfitting | 839K params with 16 basis spectra was too many params |
| Held-out synthetic metrics ≠ GCR performance | The synthetic held-out set is still drawn from the mixture distribution; GCR is OOD |
| SPR on held-out synthetic is an imperfect proxy for SPR on GCR | A unimodal held-out prediction does not guarantee unimodal GCR prediction |

---

## 6. Recommended Direction for v0.7

The correct resolution to the double-peak problem is:

**Keep the U-Net skip connections but suppress their content selectively.**

### Option A: AttnUNet1D with strong unimodal loss (preferred)

```yaml
model:
  arch: unet_attn
  skip_gate_scale: [1.0, 1.0, 1.0]
loss:
  w_unimodal: 2.0   # much stronger than v0.6's 0.5
```

The attention gates learn to zero out the part of each skip that carries the input peak position, while retaining the broad spectral shape information. The `unimodal_loss` explicitly penalises the secondary peak during training.

### Option B: UNet1D + stronger unimodal loss + skip dropout

Apply spatial dropout (e.g., `nn.Dropout1d(p=0.2)`) to each skip connection at training time. This forces the decoder to learn without full skip information, reducing reliance on the input peak position in the skip, while retaining the spatial inductive bias at test time.

### Option C: UNet1D with skip gate ablation

Use `AttnUNet1D` with `skip_gate_scale=[0.0, 1.0, 1.0]` — ablate only the shallowest skip (which carries the most local/positional information). Deeper skips carry more semantic/shape information and are less likely to cause peak leakage.

### Suggested v0.7 Config Changes

```yaml
model:
  arch: unet_attn
  base_channels: 32
  depth: 3
  head: softplus_renorm
  skip_gate_scale: [1.0, 1.0, 1.0]   # all attention gates active

loss:
  w_unimodal: 2.0      # 4× stronger than v0.6
  w_peak: 3.0          # up from 2.0
  w_emd: 2.0
  w_mse: 0.5

train:
  lambda_secondary_peak_ratio: 1.0   # up from 0.5
```

---

## 7. Anti-Leakage Compliance Audit

All v0.6 decisions remained compliant:
- Training data: 16 monoenergetic SiC/TEPC pairs only
- Heldout set: {100, 600, 2000, 7000} MeV — disjoint from training
- Selection score computed entirely from held-out synthetic metrics
- GCR spectrum evaluated once, after `frozen_config.md` was committed
- `GCR_spectrum` not referenced in `src/train.py` (verified by test)

**The failure was architectural, not a leakage violation.**

---

## 8. Conclusion

v0.6 tested the hypothesis that skip-connection removal would fix the double-peak artefact. This hypothesis was falsified: both the TransferOperator and SpectrumTransformer performed worse than the baseline UNet1D on every GCR metric. The spatial inductive bias provided by U-Net skip connections is essential for out-of-distribution generalisation to the GCR deployment spectrum.

The path forward is not to remove skip connections but to regularise them: attention gates (AttnUNet1D) combined with a stronger unimodal training penalty represent the most principled next step.
