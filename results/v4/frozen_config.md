# v0.4 预注册冻结配置（Pre-registration）

> **铁律**：本文件必须在运行 `scripts/run_gcr.py`（部署谱单次评估）**之前**提交。
> git 历史据此证明：最终配置的选择发生在看部署谱之前，杜绝事后合理化。
> 由 `tests/test_leakage.py::test_frozen_before_gcr` 锁定时间顺序。

---

## 一、最终选定配置

- 配置文件：`configs/experiment_v4.yaml`
- 输出头：`softplus`（非归一化，可输出真零）
- 模型：depth=3，base_channels=32（回退 v0.3 的过拟合大模型）
- 合成族：`[mono, sparse_k, dirichlet_uniform, loguniform, powerlaw_neutral]`（全部中性）
- EMD 空间：`index`
- 损失：w_mse=0.5，w_emd=2.0，w_mass=1.0
- 留出能量（加部署谱之前冻结）：`[100, 600, 2000, 7000]`
- 选择指标：`composite_wide = 1.0·wide_val_emd + 0.1·heldout_mono_peak_error`

## 二、在留出 / 宽谱验证集上的指标（运行后填写）

> 以下数值**只能来自留出集与宽谱验证集**，严禁填入任何部署谱派生量。

| 指标 | 值 |
|------|-----|
| 最优 composite_wide | _（待填）_ |
| wide_val_emd | _（待填）_ |
| extreme_val_emd | _（待填）_ |
| heldout_mono_peak_error (bin) | _（待填）_ |
| 收敛 epoch | _（待填）_ |

## 三、留出集稳健性检查（第四步，运行后填写）

对比两组留出设计，确认最优配置稳定：

| 留出组 | composite_wide | wide_val_emd | peak_err |
|--------|---------------|--------------|----------|
| `[100, 600, 2000, 7000]`（主） | _（待填）_ | _（待填）_ | _（待填）_ |
| 备选组（如 `[200, 800, 4000]`） | _（待填）_ | _（待填）_ | _（待填）_ |

## 四、冻结声明

- [ ] 上述配置在查看部署谱指标之前确定
- [ ] 本文件已提交到 git（提交时间早于 `results/v4/gcr_metrics.json`）
- [ ] 仅在本文件提交后，才运行 `scripts/run_gcr.py` 一次

> 提交本文件后，运行：
> ```bash
> python scripts/run_gcr.py --config configs/experiment_v4.yaml \
>     --ckpt checkpoints/best.pt --out-dir results/v4
> ```
> 该结果**不得**触发回到第一/二/三步的任何调整；若要再改，只能据留出集设计 v0.5。
