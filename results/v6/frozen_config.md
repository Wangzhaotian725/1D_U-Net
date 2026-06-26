# v0.6 Pre-registered Configuration (frozen before GCR evaluation)

**DATE**: 2026-06-26
**BRANCH**: claude/experiment-v06
**CONFIG**: configs/experiment_v6.yaml

## Hypothesis

The double-peak artefact in v0.4/v0.5 U-Net predictions is caused by skip-connection
leakage: the shallow encoder skip connections pass the input peak position (bin 179,
~9.6 keV SiC) directly into the decoder, creating a spurious secondary peak close to
the input peak (observed at bin 175, ~8.3 keV in v0.4/v0.5 predictions). The true TEPC
peak is at bin 152 (~3.4 keV), far from the spurious artefact.

## Architectural Directions Tested

**Direction A (primary): TransferOperator** (`arch: operator`)
- Low-rank (rank=24) + band-limited (±60 bins) + column-normalised transfer matrix
- No skip connections → cannot leak input peak position to output
- Physical motivation: TEPC response is a linear, energy-dependent operator on fluence

**Direction B (ablation): AttnUNet1D** (`arch: unet_attn`)
- Standard U-Net with learnable attention gates (Oktay et al. 2018) on skip connections
- `skip_gate_scale` list enables per-level gate ablation

**Direction C (ablation): SpectrumTransformer** (`arch: transformer1d`)
- 4-layer Transformer encoder (d_model=128, n_heads=8), no skip connections
- Self-attention allows arbitrary non-local energy redistribution

## Selection Score (v0.6)

```
selection_score = peak_err + 0.3 * region_emd + 0.5 * secondary_peak_ratio
```

All metrics computed on held-out energies only (100, 600, 2000, 7000 MeV).
`secondary_peak_ratio` = second-tallest / tallest peak height (0 = unimodal, 1 = equal peaks).
No deployment-spectrum information is used in any metric.

## Anti-Leakage Constraints

- 部署谱（`GCR_spectrum`）来源与物理成因视为"未知"，不假定为银河宇宙线或任何特定来源。
- 训练、合成、超参选择、版本间方向决策，均不得使用部署谱的任何信息（掩码/盲测纪律）。
- This file must be committed BEFORE `run_gcr.py` is executed.

## Frozen Hyperparameters

| Parameter | Value |
|-----------|-------|
| arch | operator |
| operator_rank | 24 |
| operator_band_halfwidth | 60 |
| head | softplus_renorm |
| w_unimodal | 0.5 |
| w_emd | 2.0 |
| w_peak | 2.0 |
| w_mse | 0.5 |
| lambda_region | 0.3 |
| lambda_secondary_peak_ratio | 0.5 |
| lr | 2e-4 |
| epochs | 600 |
| early_stop_patience | 80 |

## GCR Evaluation (to be filled after run_gcr.py)

- [ ] GCR EMD: TBD
- [ ] GCR peak position error (bins): TBD
- [ ] Secondary peak ratio: TBD
- [ ] Double-peak artefact resolved: TBD
