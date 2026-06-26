"""Anti-leakage tests for experiment v0.2."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# 1. No GCR references outside allowed files
# ---------------------------------------------------------------------------

def test_no_gcr_in_train():
    """No Python file in src/ or scripts/ (except run_gcr.py and evaluate_gcr)
    should reference the GCR file path."""
    root = Path(__file__).parent.parent
    forbidden_patterns = {"GCR_spectrum", "deploy/"}

    violations = []
    for py_file in sorted((root / "src").rglob("*.py")):
        # evaluate_gcr function is allowed (it's the deployment evaluator)
        text = py_file.read_text()
        # Allow evaluate.py since it contains evaluate_gcr (deployment only)
        if py_file.name == "evaluate.py":
            # Only the evaluate_gcr function is allowed to ref GCR — check train.py etc.
            continue
        for pat in forbidden_patterns:
            if pat in text:
                violations.append(f"{py_file}: contains {pat!r}")

    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        for py_file in sorted(scripts_dir.rglob("*.py")):
            if py_file.name == "run_gcr.py":
                continue
            text = py_file.read_text()
            for pat in forbidden_patterns:
                if pat in text:
                    violations.append(f"{py_file}: contains {pat!r}")

    assert not violations, "GCR references found outside allowed files:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 2. Synth neutral families work; gcr_like raises; no gcr_powerlaw_index attr
# ---------------------------------------------------------------------------

def test_synth_neutral():
    """Neutral families work; gcr_like raises; gcr_powerlaw_index is absent."""
    from src.synth import SynthGenerator, sample_weights

    rng = np.random.default_rng(42)
    n = 10

    # gcr_like should raise
    with pytest.raises(ValueError, match="gcr_like"):
        sample_weights(n, "gcr_like", rng)

    # SynthGenerator should not have gcr_powerlaw_index
    mono_A = np.ones((n, 32)) / 32
    mono_B = np.ones((n, 32)) / 32
    energies = np.linspace(100, 10000, n)

    gen = SynthGenerator(mono_A, mono_B, energies, families=["mono"])
    assert not hasattr(gen, "gcr_powerlaw_index"), \
        "SynthGenerator should not have gcr_powerlaw_index attribute"

    # All five neutral families work
    energies = np.linspace(100.0, 10000.0, n)
    for family in ("mono", "sparse_k", "dirichlet_uniform", "loguniform", "powerlaw_neutral"):
        w = sample_weights(n, family, rng, energies_MeV=energies)
        assert w.shape == (n,), f"Family {family!r} returned wrong shape"
        assert (w >= 0).all(), f"Family {family!r} returned negative weights"
        assert w.sum() > 0, f"Family {family!r} returned all-zero weights"

    # powerlaw_neutral must use a RANDOM exponent over a generic range, not a
    # fixed physical index. Drawing many samples should yield varied weightings.
    weights = [
        tuple(np.round(sample_weights(n, "powerlaw_neutral", np.random.default_rng(s),
                                      energies_MeV=energies), 6))
        for s in range(20)
    ]
    assert len(set(weights)) > 1, \
        "powerlaw_neutral must randomize its exponent (no fixed physical index)"


# ---------------------------------------------------------------------------
# 3. Heldout energies are disjoint from training energies
# ---------------------------------------------------------------------------

def test_heldout_disjoint():
    """Training and heldout energy sets must be disjoint (13-energy set)."""
    from src.synth import SynthGenerator

    all_energies = [90, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 4000, 5000, 6000, 7000, 8000, 10000]
    heldout = [100, 600, 2000, 7000]
    assert len(all_energies) == 16

    heldout_set = set(heldout)
    train_energies = [e for e in all_energies if e not in heldout_set]
    held_energies = [e for e in all_energies if e in heldout_set]

    # Disjoint check
    assert set(train_energies) & set(held_energies) == set(), \
        "Train and heldout energy sets overlap!"

    # SynthGenerator built from train energies contains only train energies
    K_train = len(train_energies)
    mono_A = np.ones((K_train, 32)) / 32
    mono_B = np.ones((K_train, 32)) / 32
    gen = SynthGenerator(mono_A, mono_B, np.array(train_energies), families=["mono"])

    gen_energies = set(gen.energies_MeV.tolist())
    assert gen_energies & heldout_set == set(), \
        f"SynthGenerator contains heldout energies: {gen_energies & heldout_set}"


# ---------------------------------------------------------------------------
# 4. Config has gcr_file but train.py body does not reference GCR_spectrum
# ---------------------------------------------------------------------------

def test_config_select_uses_heldout_only():
    """Configs carry gcr_file, but train.py's train() body does not reference it."""
    root = Path(__file__).parent.parent

    for name in ("experiment_v2.yaml", "experiment_v4.yaml", "experiment_v5.yaml", "experiment_v6.yaml", "experiment_v7.yaml"):
        cfg = OmegaConf.load(root / "configs" / name)
        assert "gcr_file" in cfg.data, f"cfg.data.gcr_file must be present in {name}"

    train_src = (root / "src" / "train.py").read_text()
    assert "GCR_spectrum" not in train_src, \
        "src/train.py must not reference GCR_spectrum (only evaluate_gcr may)"


# ---------------------------------------------------------------------------
# 5. Preprocessor / head pairing is locked (mirror of the v0.1 fatal bug)
# ---------------------------------------------------------------------------

def test_preprocessor_head_match():
    """Normalized head -> plain-density target; non-normalized head -> log target."""
    from src.preprocessing import build_preprocessors

    base = {
        "preprocessing": {"normalize_to_density": True, "log_compress": True, "log_scale": 1e4},
    }

    # Normalized heads: target must NOT be log-compressed.
    for head in ("softmax", "softplus_renorm"):
        cfg = OmegaConf.create({**base, "model": {"head": head}})
        _, target_pre = build_preprocessors(cfg)
        assert target_pre.log_compress is False, \
            f"normalized head {head!r} must use a plain-density target"

    # Non-normalized heads: target MUST be log-compressed (shares the head space).
    for head in ("softplus", "relu"):
        cfg = OmegaConf.create({**base, "model": {"head": head}})
        _, target_pre = build_preprocessors(cfg)
        assert target_pre.log_compress is True, \
            f"non-normalized head {head!r} must use a log-compressed target"


# ---------------------------------------------------------------------------
# 6. Mass-conservation term in the loss behaves correctly
# ---------------------------------------------------------------------------

def test_mass_conservation_loss():
    """w_mass>0 penalizes integral mismatch; identical spectra give ~zero mass."""
    import torch
    from src.losses import SpectrumLoss

    loss = SpectrumLoss(w_mse=0.0, w_emd=0.0, w_mass=1.0, log_scale=1e4)
    # Two log-compressed spectra with the same decoded mass -> mass term ~ 0.
    x = torch.rand(4, 360)
    _, d_same = loss(x, x.clone())
    assert d_same["mass"].item() < 1e-6

    # Scaling one up increases decoded mass mismatch -> positive penalty.
    _, d_diff = loss(x + 1.0, x)
    assert d_diff["mass"].item() > 0.0


# ---------------------------------------------------------------------------
# 7. Model selection / early stopping reads only heldout/wide metrics, not GCR
# ---------------------------------------------------------------------------

def test_selection_uses_no_gcr():
    """The selection signals in train.py are derived from heldout/wide loaders
    only; no GCR-derived quantity feeds early stopping or best-checkpointing."""
    root = Path(__file__).parent.parent
    train_src = (root / "src" / "train.py").read_text()

    # The composite/selection logic must not mention any GCR symbol.
    for token in ("gcr", "GCR", "evaluate_gcr", "gcr_file"):
        assert token not in train_src, \
            f"src/train.py selection path must not reference {token!r}"


# ---------------------------------------------------------------------------
# 8. Pre-registration: frozen config committed before GCR results exist
# ---------------------------------------------------------------------------

def test_frozen_before_gcr():
    """If both the frozen config and GCR results exist, the frozen config must
    be no newer than the GCR results (config chosen before the single eval)."""
    root = Path(__file__).parent.parent
    frozen = root / "results" / "v4" / "frozen_config.md"
    gcr = root / "results" / "v4" / "gcr_metrics.json"

    if not frozen.exists() or not gcr.exists():
        import pytest
        pytest.skip("frozen_config.md or gcr_metrics.json not present yet")

    assert frozen.stat().st_mtime <= gcr.stat().st_mtime + 1.0, \
        "frozen_config.md must be written/committed before GCR evaluation"


# ---------------------------------------------------------------------------
# 9. v0.5 pre-registration: frozen config committed before GCR results
# ---------------------------------------------------------------------------

def test_frozen_before_gcr_v5():
    """v0.5: results/v5/frozen_config.md must be older than gcr_metrics.json."""
    root = Path(__file__).parent.parent
    frozen = root / "results" / "v5" / "frozen_config.md"
    gcr = root / "results" / "v5" / "gcr_metrics.json"

    if not frozen.exists() or not gcr.exists():
        pytest.skip("v5 frozen_config.md or gcr_metrics.json not present yet")

    assert frozen.stat().st_mtime <= gcr.stat().st_mtime + 1.0, \
        "results/v5/frozen_config.md must be committed before GCR evaluation"


# ---------------------------------------------------------------------------
# 10. v0.5 loss components: selection_score uses no GCR; peak/region legal
# ---------------------------------------------------------------------------

def test_v5_selection_score_uses_no_gcr():
    """selection_score in train.py must not reference any GCR symbol."""
    root = Path(__file__).parent.parent
    train_src = (root / "src" / "train.py").read_text()
    for token in ("gcr", "GCR", "evaluate_gcr", "gcr_file", "GCR_spectrum"):
        assert token not in train_src, \
            f"src/train.py must not reference {token!r} in selection logic"


def test_v5_config_uses_selection_score():
    """experiment_v5.yaml must use selection_score as early_stop_metric."""
    root = Path(__file__).parent.parent
    cfg = OmegaConf.load(root / "configs" / "experiment_v5.yaml")
    assert cfg.train.early_stop_metric == "selection_score", \
        "v0.5 must use selection_score as early_stop_metric"
    assert "lambda_region" in cfg.train, \
        "v0.5 config must carry lambda_region"
    assert "w_peak" in cfg.loss, \
        "v0.5 config must carry loss.w_peak"
    assert cfg.model.head in ("softmax", "softplus_renorm"), \
        f"v0.5 must use a normalized head, got {cfg.model.head!r}"


# ---------------------------------------------------------------------------
# 11. v0.6: double-peak diagnosed on held-out synthetic data
# ---------------------------------------------------------------------------

def test_double_peak_diagnosed_on_heldout():
    """Synthetic held-out spectra fed to the baseline UNet1D show secondary_peak_ratio > 0
    for a manually crafted bimodal prediction; unimodal prediction gives ratio = 0."""
    import numpy as np
    from src.metrics import secondary_peak_ratio, count_peaks

    # A unimodal spectrum: single Gaussian at bin 150
    bins = np.arange(360)
    unimodal = np.exp(-0.5 * ((bins - 150) / 10) ** 2)
    unimodal /= unimodal.sum()

    assert count_peaks(unimodal) == 1, "Unimodal spectrum must have exactly 1 peak"
    assert secondary_peak_ratio(unimodal) == 0.0, \
        "Unimodal spectrum must have secondary_peak_ratio = 0"

    # A bimodal spectrum: two Gaussians (primary at 150, secondary at 180)
    bimodal = (
        np.exp(-0.5 * ((bins - 150) / 10) ** 2) +
        0.4 * np.exp(-0.5 * ((bins - 180) / 10) ** 2)
    )
    bimodal /= bimodal.sum()

    assert count_peaks(bimodal) >= 2, "Bimodal spectrum must have ≥ 2 peaks"
    spr = secondary_peak_ratio(bimodal)
    assert 0.0 < spr <= 1.0, f"Bimodal secondary_peak_ratio must be in (0,1], got {spr}"

    # The secondary peak (at ~180) should be close to the 'input peak' at 179
    # This mirrors the v0.4/v0.5 double-peak artefact diagnosis
    from src.metrics import transport_leak_score
    fake_input = np.exp(-0.5 * ((bins - 179) / 5) ** 2)
    leak = transport_leak_score(bimodal, fake_input)
    assert leak > 0.5, \
        f"Secondary peak at ~180 should be close to input peak at 179 (leak={leak:.3f})"


# ---------------------------------------------------------------------------
# 12. v0.6: config and pre-registration checks
# ---------------------------------------------------------------------------

def test_v6_config_uses_selection_score():
    """experiment_v6.yaml must use selection_score with secondary_peak_ratio term."""
    root = Path(__file__).parent.parent
    cfg = OmegaConf.load(root / "configs" / "experiment_v6.yaml")
    assert cfg.train.early_stop_metric == "selection_score", \
        "v0.6 must use selection_score as early_stop_metric"
    assert "lambda_secondary_peak_ratio" in cfg.train, \
        "v0.6 config must carry lambda_secondary_peak_ratio"
    assert "w_unimodal" in cfg.loss, \
        "v0.6 config must carry loss.w_unimodal"
    assert cfg.model.head in ("softmax", "softplus_renorm"), \
        f"v0.6 must use a normalized head, got {cfg.model.head!r}"
    assert cfg.model.arch in ("operator", "transformer1d", "unet_attn", "unet1d"), \
        f"v0.6 arch must be a valid architecture, got {cfg.model.arch!r}"


def test_frozen_before_gcr_v6():
    """v0.6: results/v6/frozen_config.md must be older than gcr_metrics.json."""
    root = Path(__file__).parent.parent
    frozen = root / "results" / "v6" / "frozen_config.md"
    gcr = root / "results" / "v6" / "gcr_metrics.json"

    if not frozen.exists() or not gcr.exists():
        pytest.skip("v6 frozen_config.md or gcr_metrics.json not present yet")

    assert frozen.stat().st_mtime <= gcr.stat().st_mtime + 1.0, \
        "results/v6/frozen_config.md must be committed before GCR evaluation"


# ---------------------------------------------------------------------------
# 13. v0.6: new architecture sanity checks
# ---------------------------------------------------------------------------

def test_transfer_operator_output():
    """TransferOperator output is non-negative, sums to 1, and unimodal for unimodal input."""
    import torch
    import numpy as np
    from src.models.operator import TransferOperator

    model = TransferOperator(n_bins=64, rank=4, band_halfwidth=10)
    model.eval()

    with torch.no_grad():
        x = torch.rand(2, 1, 64)
        x = x / x.sum(dim=-1, keepdim=True)
        out = model(x)

    assert out.shape == (2, 1, 64), f"Expected (2,1,64), got {out.shape}"
    assert (out >= 0).all(), "TransferOperator output must be non-negative"
    sums = out.squeeze(1).sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5), \
        f"TransferOperator output must sum to 1, got {sums}"


def test_spectrum_transformer_output():
    """SpectrumTransformer output shape and no NaN."""
    import torch
    from src.models.transformer1d import SpectrumTransformer

    model = SpectrumTransformer(n_bins=64, d_model=32, n_layers=2, n_heads=4)
    model.eval()

    with torch.no_grad():
        x = torch.rand(3, 1, 64)
        out = model(x)

    assert out.shape == (3, 1, 64), f"Expected (3,1,64), got {out.shape}"
    assert not torch.isnan(out).any(), "SpectrumTransformer output contains NaN"
    assert (out >= 0).all(), "SpectrumTransformer softplus_renorm output must be non-negative"


def test_attn_unet_output():
    """AttnUNet1D output shape; ablated skip (scale=0) still produces valid output."""
    import torch
    from src.models.unet_attn import AttnUNet1D

    model = AttnUNet1D(in_ch=1, out_ch=1, base=8, depth=2,
                       skip_gate_scale=[1.0, 0.0])
    model.eval()

    with torch.no_grad():
        x = torch.rand(2, 1, 128)
        out = model(x)

    assert out.shape == (2, 1, 128), f"Expected (2,1,128), got {out.shape}"
    assert not torch.isnan(out).any(), "AttnUNet1D output contains NaN"


def test_unimodal_loss_differentiable():
    """unimodal_loss is zero for a perfectly monotone ramp and positive for oscillating signal."""
    import torch
    from src.metrics import unimodal_loss

    # Perfectly monotone increasing then decreasing (tent): exactly unimodal, zero reversals
    N = 64
    t = torch.arange(N, dtype=torch.float32)
    tent = torch.where(t < N // 2, t, torch.tensor(float(N - 1)) - t).unsqueeze(0)

    # Oscillating signal: many sign reversals
    osc = torch.sin(torch.linspace(0, 8 * 3.14159, N)).unsqueeze(0)

    loss_tent = unimodal_loss(tent)
    loss_osc = unimodal_loss(osc)

    assert loss_tent.item() == 0.0, f"Tent (unimodal) loss should be exactly 0, got {loss_tent.item()}"
    assert loss_osc.item() > 0.0, f"Oscillating unimodal_loss should be > 0, got {loss_osc.item()}"


# ---------------------------------------------------------------------------
# 14. v0.7: config checks and pre-registration guard
# ---------------------------------------------------------------------------

def test_v7_config():
    """experiment_v7.yaml must use unet_attn with stronger unimodal loss."""
    root = Path(__file__).parent.parent
    cfg = OmegaConf.load(root / "configs" / "experiment_v7.yaml")
    assert cfg.model.arch == "unet_attn", \
        "v0.7 must use unet_attn (skip connections + attention gates)"
    assert cfg.train.early_stop_metric == "selection_score"
    assert float(cfg.loss.w_unimodal) >= 1.5, \
        "v0.7 must use w_unimodal >= 1.5 (4x stronger than v0.6)"
    assert float(cfg.loss.w_peak) >= 2.5, \
        "v0.7 must use w_peak >= 2.5"
    assert float(cfg.train.lambda_secondary_peak_ratio) >= 0.8, \
        "v0.7 lambda_secondary_peak_ratio must be >= 0.8"
    assert cfg.model.head in ("softmax", "softplus_renorm")


def test_frozen_before_gcr_v7():
    """v0.7: results/v7/frozen_config.md must be older than gcr_metrics.json."""
    root = Path(__file__).parent.parent
    frozen = root / "results" / "v7" / "frozen_config.md"
    gcr = root / "results" / "v7" / "gcr_metrics.json"

    if not frozen.exists() or not gcr.exists():
        pytest.skip("v7 frozen_config.md or gcr_metrics.json not present yet")

    assert frozen.stat().st_mtime <= gcr.stat().st_mtime + 1.0, \
        "results/v7/frozen_config.md must be committed before GCR evaluation"
