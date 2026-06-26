# v0.5 预注册冻结配置（Pre-registration）

> **铁律**：本文件必须在运行 `scripts/run_gcr.py`（部署谱单次评估）**之前**提交。
> git 历史据此证明：最终配置的选择发生在看部署谱之前，杜绝事后合理化。
> 由 `tests/test_leakage.py::test_frozen_before_gcr_v5` 锁定时间顺序。

---

## 一、最终选定配置

- 配置文件：`configs/experiment_v5.yaml`
- 输出头：`softplus_renorm`（归一化，回退 v0.4 的非归一化头）
- 模型：depth=3，base_channels=32（小模型，防过拟合）
- 合成族：`[mono, sparse_k, dirichlet_uniform, loguniform, powerlaw_neutral]`（全部中性）
- EMD 空间：`index`
- **新增损失项**：
  - `w_peak: 2.0`（软 argmax 峰位对齐损失，v0.5 第一优先）
  - `region_kev: [0.1, 1000]`（MSE 区间加权；高能平底区权重 0.1）
- 选择指标：`selection_score = peak_err + 0.3 · region_emd[0.1-1000 keV]`
- 留出能量（沿用 v0.4）：`[100, 600, 2000, 7000]`

## 二、本轮优化优先级（用户指定）

1. **第一优先**：峰位对齐（peak position alignment）
2. **第二优先**：0.1 keV – 1000 keV 区间谱线整体吻合
3. **不追求**：高能尾部（>1000 keV）快速归零

## 三、在留出 / 区间验证集上的指标（运行后填写）

> 以下数值**只能来自留出集与区间验证集**，严禁填入任何部署谱派生量。

| 指标 | 值 |
|------|-----|
| 最优 selection_score | _（待填）_ |
| heldout_mono_peak_error (bin) | _（待填）_ |
| region_emd [0.1-1000 keV] | _（待填）_ |
| wide_val_emd | _（待填）_ |
| 收敛 epoch | _（待填）_ |

## 四、留出集稳健性检查（运行后填写）

| 留出组 | selection_score | peak_err | region_emd |
|--------|----------------|---------|------------|
| `[100, 600, 2000, 7000]`（主） | _（待填）_ | _（待填）_ | _（待填）_ |
| 备选组（如 `[200, 800, 4000]`） | _（待填）_ | _（待填）_ | _（待填）_ |

## 五、冻结声明

- [ ] 上述配置在查看部署谱指标之前确定
- [ ] 本文件已提交到 git（提交时间早于 `results/v5/gcr_metrics.json`）
- [ ] 仅在本文件提交后，才运行 `scripts/run_gcr.py` 一次

> 提交本文件后，运行：
> ```bash
> python scripts/run_gcr.py --config configs/experiment_v5.yaml \
>     --ckpt checkpoints/best.pt --out-dir results/v5
> ```
> 该结果**不得**触发回到第一/二/三步的任何调整；若要再改，只能据留出集设计 v0.6。
