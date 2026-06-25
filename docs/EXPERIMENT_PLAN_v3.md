# 实验计划 v0.3：深度扩展 + 中性幂律族 + EMD 空间回退

> 基于 v0.1/v0.2 对比分析，针对 GCR 泛化能力的系统性改进  
> 分支：`claude/experiment-v03`  
> 日期：2026-06-25

---

## 一、核心问题与对策

| 问题 | 根因 | v0.3 对策 |
|------|------|----------|
| 峰位误差 2→6 bin 退步 | `log_energy` EMD 放大高能距离，偏移低能 | 回退到 `emd_space: index` |
| 高能平底 ~1e-4 | 归一化输出头无法输出真正的零 | 增加模型容量（depth=4, base=64），让模型更精准地分配质量 |
| CDF 低能偏移 | 中性先验对低/高能对称采样，不覆盖 GCR 峰区 | 新增 `powerlaw_neutral` 族（随机幂律，不假定 GCR 来源） |
| 验证 EMD ↓ 98% 但 GCR EMD ↓ 13% | val_emd 与 GCR 泛化脱钩 | 增大每轮样本数（2048）、放宽早停（patience=80） |

---

## 二、配置变更

### 2.1 模型架构（优先级最高）

```yaml
model:
  head: softplus_renorm   # 保持（允许近零输出）
  depth: 4                # 3 → 4（感受野覆盖全部 360 bin）
  base_channels: 64       # 32 → 64（参数量约 4 倍，~250 万）
```

**原理**：depth=3 时，编码器最深层分辨率为 360/8=45 bin，无法捕捉全局谱形。
depth=4 时最深层分辨率为 360/16=22 bin，感受野覆盖整个输入，有助于建模宽谱的全局结构。

### 2.2 EMD 距离空间（紧急修复）

```yaml
loss:
  emd_space: index        # log_energy → index
  w_emd: 2.0              # 保持
  w_mse: 0.5              # 保持
```

**原理**：v0.2 改用 `log_energy` 后峰位误差从 2 退步到 6 bin。
`index` 空间（均匀 bin 距离）在 v0.1 中取得了最佳峰位精度，应先回退验证。

### 2.3 新增合成族：`powerlaw_neutral`

```yaml
synth:
  mixture_families: [mono, sparse_k, dirichlet_uniform, loguniform, powerlaw_neutral]
  powerlaw_alpha_range: [-3.0, 0.0]   # 随机幂律指数范围，不预设 GCR 值
  sparse_k_range: [2, 8]              # [2,4] → [2,8]，覆盖更宽的峰
  dirichlet_alpha_choices: [0.1, 0.3, 1.0, 3.0]  # 增加 0.1（更稀疏）
```

**`powerlaw_neutral` 设计**：
- 对 K 个能量的权重按 `w_i ∝ E_i^α` 采样，`α ~ Uniform(-3, 0)`
- `α=0` → 均匀权重；`α=-2.7` → 类 GCR；`α=-0.5` → 轻微软谱
- **关键**：α 从 [-3, 0] 均匀采样，不假定任何特定物理分布，仅覆盖"幂律宽谱"形状空间
- 这使模型接触到从硬谱到软谱的完整连续体，而不是把 GCR 指数硬编码为 -2.7

### 2.4 训练超参数

```yaml
train:
  samples_per_epoch: 2048   # 1024 → 2048
  epochs: 600               # 500 → 600（depth=4 收敛更慢）
  early_stop_patience: 80   # 50 → 80（给更深网络更多探索空间）
  lr: 2.0e-4                # 3e-4 → 2e-4（更深网络用更小初始学习率）
  batch_size: 64            # 保持
```

---

## 三、需要新增的代码

### 3.1 `src/synth.py`：新增 `powerlaw_neutral` 族

```python
elif family == "powerlaw_neutral":
    alpha = rng.uniform(*self.powerlaw_alpha_range)  # 从 [-3, 0] 均匀采样
    w = energies ** alpha
    w = w / w.sum()
```

### 3.2 `configs/experiment_v3.yaml`：完整配置文件

新建 `configs/experiment_v3.yaml`（详见第四节）。

### 3.3 `tests/test_synth.py`：新增 `powerlaw_neutral` 测试

验证：
- `powerlaw_neutral` 家族可以正常采样
- α 参数从配置范围均匀采样
- 输出权重之和为 1

---

## 四、完整 `experiment_v3.yaml`

```yaml
seed: 42
data:
  raw_glob: "data/raw/*MeV.xlsx"
  gcr_file: "data/deploy/GCR_spectrum.xlsx"
  sheet_skiprows: 1
  direction: "SiC_SBD->TEPC"
  heldout_energies_MeV: [600, 2000, 7000]

preprocessing:
  normalize_to_density: true
  log_compress: true
  log_scale: 1.0e4

synth:
  mixture_families: [mono, sparse_k, dirichlet_uniform, loguniform, powerlaw_neutral]
  dirichlet_alpha_choices: [0.1, 0.3, 1.0, 3.0]
  sparse_k_range: [2, 8]
  powerlaw_alpha_range: [-3.0, 0.0]
  poisson_noise: true
  poisson_counts_range: [1000, 100000]

model:
  arch: unet1d
  in_ch: 1
  base_channels: 64
  depth: 4
  norm: group
  activation: gelu
  head: softplus_renorm

loss:
  w_mse: 0.5
  w_emd: 2.0
  emd_space: index

train:
  samples_per_epoch: 2048
  val_mixtures: 256
  test_mixtures: 256
  epochs: 600
  early_stop_metric: val_emd
  early_stop_patience: 80
  batch_size: 64
  optimizer: adamw
  lr: 2.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine
  grad_clip: 1.0
  amp: true

eval:
  tail_percentile: 90
```

---

## 五、预期改进

| 指标 | v0.2 结果 | v0.3 预期 |
|------|----------|----------|
| GCR EMD | 6.38 | < 5.0 |
| 峰位误差 | 6 bin | ≤ 3 bin |
| 高能平底 | ~1e-4 | ~1e-5（depth=4 更精准聚焦） |
| CDF 低能偏移 | 明显 | 改善（powerlaw_neutral 覆盖更宽形状空间） |

---

## 六、消融实验设计（可选）

为定位各变更的贡献，可运行以下对照组：

| 实验 | 变更 | 目的 |
|------|------|------|
| v3-ablation-A | 仅 `emd_space: index`（其他同 v0.2） | 验证 EMD 空间是峰位退步的根因 |
| v3-ablation-B | 仅 depth=4, base=64（其他同 v0.2） | 验证更深网络对 GCR EMD 的贡献 |
| v3-ablation-C | 完整 v0.3 | 综合效果 |

---

## 七、反泄漏约束（继承 v0.2）

- `powerlaw_neutral` 的 α 范围 [-3, 0] 不预设 GCR 指数，仅覆盖物理上合理的幂律区间
- GCR_spectrum.xlsx 仍仅在 `scripts/run_gcr.py` + `src/evaluate.py` 中被引用
- 所有超参数（depth, base, emd_space, powerlaw_alpha_range 等）根据**持留集验证指标**选定，
  GCR 测试集保持严格盲测

---

## 八、运行命令

```bash
# 训练
python scripts/run_train.py --config configs/experiment_v3.yaml

# 快速验证（2 batch × 1 epoch）
python scripts/run_train.py --config configs/experiment_v3.yaml --fast-dev-run

# GCR 最终评估（训练完成后执行一次）
python scripts/run_gcr.py --config configs/experiment_v3.yaml \
    --ckpt checkpoints/best.pt --out-dir results/gcr_v3

# 测试套件
pytest -q tests/
```
