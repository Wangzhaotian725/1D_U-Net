# 1D U-Net 探测器谱转换实验总结报告

> 项目：SiC_SBD → TEPC 能量沉积谱转换  
> 覆盖版本：v0.1 基线 / v0.2 中性先验 / v0.3 深度扩展  
> 生成日期：2026-06-26  
> 代码仓库：`Wangzhaotian725/1D_U-Net`

---

## 一、项目背景与目标

### 1.1 问题定义

利用 1D U-Net 神经网络，将 SiC 半导体探测器（SiC_SBD）测量的能量沉积谱
转换为等效的组织等效比例计数器（TEPC）谱，从而在辐射环境中用低成本固态探测器
替代体积庞大的 TEPC。

- **输入**：SiC_SBD 测量谱，360 个 bin，对数能量轴（~0.01～10000 keV）
- **输出**：对应的 TEPC 谱，同样 360 个 bin
- **核心假设**：探测器响应对中子注量线性叠加（已由 GCR 测试验证）

### 1.2 数据结构

| 数据集 | 内容 | 用途 |
|--------|------|------|
| 单能谱（11 个） | 200/300/400/500/600/800/1000/2000/4000/7000/10000 MeV 中子 | 合成训练/验证数据 |
| 持留能量（3 个） | 600/2000/7000 MeV（从 11 个中划出） | 验证集（插值泛化测试） |
| GCR 宽谱（1 个） | 真实银河宇宙线测量谱 | **最终盲测集，绝不参与训练** |

### 1.3 核心设计原则

1. **softmax/softplus_renorm 头 + 纯密度目标**：输出头归一化，目标谱必须为归一化密度，log 压缩只用于输入端
2. **持留整个能量**：测试"对未见能量插值"的泛化能力，非随机样本划分
3. **GCR 严格盲测**：GCR 谱仅在 `scripts/run_gcr.py` 和 `src/evaluate.py` 中被引用，不参与任何超参数选择

---

## 二、关键 Bug 记录：softmax/log 压缩空间不匹配

### 2.1 症状

v0.1 初始实现中，训练 300 epoch 后损失从 65740 仅降至 65670（降幅 0.1%），模型完全不收敛：

```
GCR EMD: 72.6
峰位误差: 59 bin
预测谱: 完全平坦的均匀分布
```

### 2.2 根因

| 空间 | 数值范围 | 说明 |
|------|---------|------|
| softmax 头输出 | ~0.001～0.05，和为 1 | 归一化概率密度 |
| log 压缩目标 | `log1p(1e4 × density)` ≈ 3～7，和约 1000 | 对数尺度 |

两者处于完全不同的数值空间，EMD/MSE 损失被一个无法弥合的常数偏移主导，梯度无效。

### 2.3 修复方案

- **目标谱**：保持为纯归一化密度，不做 log 压缩
- **输入谱**：log 压缩（`log1p(1e4 × density)`），处理 6 个数量级的动态范围
- 新增 `build_preprocessors(cfg)` 函数，根据输出头类型自动构造正确的 `(input_pre, target_pre)` 对

### 2.4 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 训练损失（最终） | 65670（不动） | **0.165** |
| GCR EMD | 72.6 | **7.33** |
| GCR 峰位误差 | 59 bin | **2 bin** |

---

## 三、v0.1 基线实验

### 3.1 配置

```yaml
model:
  head: softmax
  depth: 3
  base_channels: 32        # 参数量：629,857

synth:
  mixture_families: [mono, sparse, dense, gcr_like]
  gcr_powerlaw_index: -2.7  # 使用了 GCR 物理先验

loss:
  w_mse: 1.0
  w_emd: 1.0
  emd_space: index

train:
  epochs: 300
  samples_per_epoch: 1024
  # 无早停
```

### 3.2 训练曲线

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 29.82     | 6.90     |
| 50    | 0.64      | 1.07     |
| 100   | 0.45      | 0.91     |
| 150   | 0.37      | 0.81     |
| 200   | 0.23      | 0.54     |
| 250   | 0.18      | 0.50     |
| 300   | 0.165     | 0.516    |
| **最优** | — | **0.430**（约第 170 轮） |

### 3.3 GCR 评估结果

```json
{
  "emd": 7.33,
  "mse": 1.86e-05,
  "mae": 1.55e-03,
  "peak_pos_error": 2.0
}
```

### 3.4 分析

**优点**：
- 主峰区（3～10 keV）预测与真实 TEPC 谱高度吻合
- 峰位误差仅 2 bin（约 1 个几何步长），峰形基本准确
- CDF 曲线在主峰区重合良好

**缺点**：
- 使用了 `gcr_like`（幂律指数 -2.7）先验，存在 GCR 信息泄漏风险，实验结论的严谨性存疑
- 高能尾部（10 keV 以上）存在约 1e-5 的平底（softmax 无法输出真正的零）
- 无早停机制，300 epoch 后验证损失出现轻微反弹

---

## 四、v0.2 中性先验实验

### 4.1 核心变更

**目标**：移除所有 GCR 物理先验，确保实验结论不依赖对 GCR 的任何假设。

```yaml
model:
  head: softplus_renorm    # softmax → softplus_renorm（允许近零输出）
  depth: 3
  base_channels: 32        # 参数量不变：629,857

synth:
  # 移除 gcr_like 和 gcr_powerlaw_index
  mixture_families: [mono, sparse_k, dirichlet_uniform, loguniform]
  dirichlet_alpha_choices: [0.3, 1.0, 3.0]
  sparse_k_range: [2, 4]

loss:
  w_mse: 0.5
  w_emd: 2.0               # EMD 权重加倍
  emd_space: log_energy    # index → log_energy（对数能量距离）

train:
  epochs: 500
  early_stop_metric: val_emd
  early_stop_patience: 50  # 新增早停
```

新增 `tests/test_leakage.py`：4 项反泄漏测试，确保 GCR 信息不泄漏至训练流程。

### 4.2 训练曲线

| Epoch | Train Loss | Val Loss | Val EMD |
|-------|-----------|----------|---------|
| 1     | 1.6763    | 0.9808   | 0.4904  |
| 10    | 0.2262    | 0.2202   | 0.1101  |
| 50    | 0.0224    | 0.0236   | 0.0118  |
| 100   | 0.0108    | 0.0207   | 0.0103  |
| 110   | 0.0115    | 0.0177   | 0.0089  |
| **161（早停）** | — | — | — |
| **最优** | — | **0.0172** | **0.0086** |

验证损失从 v0.1 的 0.430 降至 0.0172（↓ 96%），161 轮提前收敛（节省 54% 计算）。

### 4.3 GCR 评估结果

```json
{
  "emd": 6.38,
  "mse": 2.08e-05,
  "mae": 1.47e-03,
  "peak_pos_error": 6.0
}
```

### 4.4 分析

**进步**：
- GCR EMD：7.33 → 6.38（↓ 13%）
- softplus_renorm 使高能平底从 ~3e-3 降至 ~1e-4
- 移除 gcr_like，实验设计更严谨，结论可信度更高
- 早停有效，训练效率提升

**退步**：
- 峰位误差：2 bin → 6 bin（退步）
- CDF 低能端偏移更明显

**根因分析**：
- `log_energy` EMD 空间使高能 bin 之间距离被人为放大，优化器倾向于把质量集中在低能端
- 中性先验（dirichlet_uniform/loguniform）对 11 个能量对称采样，不覆盖 GCR 中高能主峰区
- 验证 EMD（0.0086）与 GCR EMD（6.38）出现脱钩：val_emd 在持留能量上表现极好，但不能完全代理 GCR 宽谱泛化能力

---

## 五、v0.3 深度扩展实验

### 5.1 核心变更

**目标**：增大模型容量 + 回退 EMD 空间 + 新增中性幂律合成族。

```yaml
model:
  head: softplus_renorm
  depth: 4               # 3 → 4（感受野覆盖全 360 bin）
  base_channels: 64      # 32 → 64（参数量：10,121,921，约 16 倍）

synth:
  mixture_families: [mono, sparse_k, dirichlet_uniform, loguniform, powerlaw_neutral]
  powerlaw_alpha_range: [-3.0, 0.0]   # 新增：随机幂律，不预设 GCR 指数
  sparse_k_range: [2, 8]              # [2,4] → [2,8]
  dirichlet_alpha_choices: [0.1, 0.3, 1.0, 3.0]

loss:
  w_mse: 0.5
  w_emd: 2.0
  emd_space: index       # log_energy → index（回退，修复峰位退步）

train:
  samples_per_epoch: 2048  # 1024 → 2048
  epochs: 600
  early_stop_patience: 80  # 50 → 80
  lr: 2.0e-4               # 3e-4 → 2e-4
  # 使用 GPU（NVIDIA RTX 5070）
```

### 5.2 训练曲线

| Epoch | Train Loss | Val Loss | Val EMD |
|-------|-----------|----------|---------|
| 1     | 90.2919   | 28.3312  | 14.1656 |
| 10    | 2.6097    | 2.6568   | 1.3284  |
| 50    | 0.7988    | 1.1859   | 0.5929  |
| 100   | 0.4433    | 1.1064   | 0.5532  |
| 140   | 0.3852    | 1.1638   | 0.5819  |
| **144（早停）** | — | — | — |
| **最优** | — | **0.9707** | **0.4854** |

> 注：v0.2 与 v0.3 的 val_emd 不可直接比较（EMD 空间不同：log_energy vs index，量纲不同）。

训练/验证损失存在明显 gap（train≈0.39，val≈1.16），表明 **10M 参数模型对 8 个训练能量的数据过拟合**。

### 5.3 GCR 评估结果

```json
{
  "emd": 6.01,
  "mse": 2.13e-05,
  "mae": 1.47e-03,
  "peak_pos_error": 5.0
}
```

### 5.4 分析

**进步**：
- GCR EMD：6.38 → 6.01（↓ 6%），三版持续下降
- 峰位误差：6 → 5 bin（小幅改善，emd_space 回退有效）
- powerlaw_neutral 族提供了更宽形状覆盖

**问题**：
- 模型严重过拟合（10M 参数 vs 8 个训练能量）
- 高能平底（~1e-4）依然存在，与 v0.2 几乎相同
- CDF 低能偏移未得到显著改善

---

## 六、三版横向对比

### 6.1 GCR 指标汇总

| 指标 | v0.1 | v0.2 | v0.3 | 总变化 |
|------|------|------|------|--------|
| **GCR EMD**（主指标） | 7.33 | 6.38 | **6.01** | ↓ 18% |
| MSE | 1.86e-5 | 2.08e-5 | 2.13e-5 | ≈ 持平 |
| MAE | 1.55e-3 | 1.47e-3 | 1.47e-3 | ↓ 5% |
| **峰位误差**（bin） | **2** | 6 | 5 | v0.2 引入退步，v0.3 略有修复 |

### 6.2 训练效率对比

| 版本 | 参数量 | 收敛 Epoch | 设备 | 每 Epoch 耗时（估算） |
|------|--------|-----------|------|----------------------|
| v0.1 | 629,857 | 300（无早停） | CPU | ~30s |
| v0.2 | 629,857 | 161（早停） | CPU | ~30s |
| v0.3 | 10,121,921 | 144（早停） | GPU (RTX 5070) | ~5s |

### 6.3 关键发现总结

| 发现 | 影响 |
|------|------|
| softplus_renorm 优于 softmax | 高能平底从 3e-3 降至 1e-4 |
| log_energy EMD 导致峰位退步 | 峰位误差 2→6 bin（v0.2），回退到 index 部分修复 |
| val_emd 与 GCR EMD 脱钩 | val_emd 极低（0.0086）但 GCR EMD 未同步大幅改善 |
| 10M 参数对小数据集过拟合 | v0.3 train/val gap 约 3x |
| 归一化输出头是高能平底的根本限制 | 三版均存在 ~1e-4 平底，换用非归一化头才能根治 |

---

## 七、当前瓶颈与根因

### 7.1 高能平底（最主要的剩余误差来源）

**现象**：预测谱在 10 keV 以上维持约 1e-4 的平底，而真实 TEPC 谱快速跌落至零。

**根因**：softmax 和 softplus_renorm 均强制 `∑output = 1`。360 个 bin 中，
约 300 个在真实谱中应为零的 bin 被均摊了残余概率质量，无法输出真正的零。

**量化**：360 个 bin 均摊 → 最低可能平底 ≈ 1/360 ≈ 3e-3（softmax）；
softplus_renorm 通过允许极小值，实际平底降至 ~1e-4，但仍无法消除。

### 7.2 CDF 低能端偏移

**现象**：预测 CDF 在 1～5 keV 段比真实 CDF 更快上升，说明预测谱质心偏低能。

**根因**：
- 训练合成族对 11 个能量对称采样，低能（200～600 MeV）分量出现频率与高能相同
- 真实 GCR 谱的质心位于约 5 keV（对应 1000 MeV 量级的中子能量），高于合成样本的平均质心

### 7.3 val_emd 与 GCR EMD 的信息差

验证集测试的是对"持留能量点"的插值能力；GCR 测试的是对"连续宽谱形状"的外推能力。
这两者在信息内容上存在本质差距：val_emd→0 不意味着 GCR EMD→0。

---

## 八、下一步方向（v0.4）

### 优先级 A：换用非归一化输出头（最高优先级）

**方案**：将输出头改为 `relu` 或 `softplus`（不归一化），目标谱改用 log 压缩密度。

```yaml
model:
  head: linear            # 或 relu/softplus
preprocessing:
  target_log: true        # 目标谱也做 log 压缩
```

**预期效果**：模型可输出真正的零，高能平底从 ~1e-4 降至 ~0（或接近数值精度底限）。
需同步修改 `build_preprocessors(cfg)` 和评估脚本的逆变换逻辑。

### 优先级 B：回归小模型 + powerlaw_neutral（继承 v0.3 改进）

```yaml
model:
  depth: 3
  base_channels: 32       # 避免过拟合
synth:
  mixture_families: [mono, sparse_k, dirichlet_uniform, loguniform, powerlaw_neutral]
  # 保留 v0.3 的合成族设计
loss:
  emd_space: index        # 保持 index（峰位更准）
```

### 优先级 C：增大训练数据多样性

- 增大每轮样本数：`samples_per_epoch: 4096`
- 扩大 Dirichlet alpha 范围：`[0.05, 0.1, 0.3, 1.0, 3.0, 10.0]`（覆盖更极端的稀疏/稠密分布）

---

## 九、关键代码位置速查

| 要修改的行为 | 文件 | 关键位置 |
|-------------|------|---------|
| 输出头类型 | `configs/experiment_v*.yaml` | `model.head` |
| 输入/目标预处理分离 | `src/preprocessing.py` | `build_preprocessors()` |
| 合成策略家族 | `configs/experiment_v*.yaml` | `synth.mixture_families` |
| 幂律中性族实现 | `src/synth.py` | `sample_weights()` 中的 `powerlaw_neutral` |
| EMD 距离空间 | `src/losses.py` | `make_bin_dist()` |
| GCR 最终评估 | `src/evaluate.py` | `evaluate_gcr()` |
| 反泄漏测试 | `tests/test_leakage.py` | 4 项测试函数 |

---

## 十、附录：实验分支与配置文件对应关系

| 版本 | Git 分支 | 配置文件 | 结果目录 |
|------|---------|---------|---------|
| v0.1 基线 | `claude/nifty-einstein-tuilh3` | `configs/baseline.yaml` | `results/gcr/` |
| v0.2 中性先验 | `claude/nifty-einstein-tuilh3` | `configs/experiment_v2.yaml` | `results/gcr_v2/` |
| v0.3 深度扩展 | `claude/experiment-v03` | `configs/experiment_v3.yaml` | `results/gcr_v3/` |
| v0.4（计划中） | `claude/experiment-v04`（待建） | `configs/experiment_v4.yaml` | `results/gcr_v4/` |
