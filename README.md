# 探测器能谱转换 — 1D U-Net 基线

> 利用 **1D U-Net** 神经网络，从 SiC 探测器的能量沉积谱预测 TEPC 探测器的能量沉积谱。
> 训练数据为 11 组单能质子谱（200 MeV ~ 10000 MeV），最终部署目标为真实的 **银河宇宙射线（GCR）宽谱**。

---

## 项目背景

两种探测器（SiC_SBD 和 TEPC）在相同辐射场中会产生不同的能量沉积谱。两者的谱形均属
Gaussian-Vavilov-Landau 族（非对称、重尾），没有简洁的解析关系，因此采用神经网络学习
从探测器 A（SiC）到探测器 B（TEPC）的映射。

### 核心挑战：从单能谱泛化到 GCR 宽谱

模型只用 11 条**窄峰谱**训练，却要在**平滑宽谱**的 GCR 上推理。
解决方案依赖一个物理事实：

> **探测器响应对粒子注量具有线性叠加性。**
> 混合辐射场的谱 = 各单能成分谱的注量加权和，且对两个探测器独立成立。

因此，同一组随机权重作用于 SiC 各单能谱和 TEPC 各单能谱，
得到的混合对依然是**物理上严格一致的** A→B 配对（`src/synth.py`）。
通过这种方式可以无限生成接近 GCR 形状的宽谱训练样本，从根本上弥合训练/部署差距。

---

## 目录结构

```
1D_U-Net/
├── README.md
├── LICENSE
├── pyproject.toml              # 项目依赖与打包配置
├── configs/
│   └── baseline.yaml           # 所有超参数的统一配置文件
├── data/
│   ├── raw/                    # 11 个单能训练文件（*MeV.xlsx）
│   ├── deploy/                 # GCR_spectrum.xlsx（仅推理，不参与训练）
│   └── processed/              # 脚本生成：energy_grid.npy、mono_spectra.npz
├── src/                        # 核心源码
│   ├── data.py                 # 数据加载（列名别名映射 + GCR 重采样）
│   ├── preprocessing.py        # 归一化 + 对数压缩预处理
│   ├── synth.py                # 合成宽谱生成（加权线性叠加）
│   ├── dataset.py              # PyTorch Dataset 封装
│   ├── model.py                # 1D U-Net 模型
│   ├── losses.py               # MSE + EMD 组合损失
│   ├── train.py                # 训练主循环
│   ├── evaluate.py             # 评估指标计算
│   └── plots.py                # 可视化工具
├── scripts/
│   ├── prepare_data.py         # 原始 xlsx → 处理后的 npy/npz
│   ├── run_train.py            # 启动训练
│   ├── run_eval.py             # 在留出集上评估
│   └── run_gcr.py              # 在 GCR 真实谱上推理并报告指标
└── tests/
    ├── test_data.py            # 列名别名、重采样正确性测试
    ├── test_preprocessing.py   # 预处理正逆变换一致性测试
    ├── test_synth.py           # 合成混合线性性测试
    ├── test_losses.py          # EMD 正确性与梯度回传测试
    └── test_model.py           # 模型输出形状、NaN、softmax 归一化测试
```

---

## 各文件详解

### `configs/baseline.yaml`
所有可调参数的唯一入口。关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `data.direction` | `SiC_SBD->TEPC` | 映射方向，可改为反向 |
| `data.heldout_energies_MeV` | `[600, 2000, 7000]` | 留出能量，不参与训练 |
| `preprocessing.log_compress` | `true` | 对数压缩 y 轴 |
| `synth.mixture_families` | `[mono, sparse, dense, gcr_like]` | 合成策略 |
| `loss.w_mse / w_emd` | `1.0 / 1.0` | 损失权重 |
| `train.epochs` | `300` | 训练轮数 |
| `model.depth` | `3` | U-Net 深度（下采样层数） |

---

### `src/data.py` — 数据加载

- **`load_spectrum_file(path, skiprows)`**：按**列名别名**（大小写/空格不敏感）读取 xlsx，
  不按列位置，因此同时支持训练文件（`keV / SiC_SBD / TEPC`）和 GCR 文件（`E(keV) / SiC / TEPC`）。
  从文件名解析一次束流能量（MeV）作为元数据。
- **`resample_to_canonical(energy, values, canonical_grid)`**：在**对数能量轴**上插值，
  将 GCR 文件的 420 bin 谱重采样到 360 bin 标准网格，并警告丢失质量超过 0.01% 的情况。

> 注意：训练文件用 `skiprows=1`（第一行为干扰行，真表头在第二行）；
> GCR 文件用 `skiprows=0`（表头在第一行）。

---

### `src/preprocessing.py` — 预处理

`Preprocessor` 类按顺序执行两步变换：

1. **密度归一化**：将计数谱除以总计数，使之和为 1（物理意义上的概率密度）。
2. **对数压缩**（可选）：`y → log1p(scale × y)`，让网络同时关注峰区和长尾。

提供精确逆变换 `.inverse_transform()`，用于将模型输出还原到物理计数空间。

> **重要：输入与目标使用不同的预处理。**
> 模型输出头是 softmax，产生的是归一化密度（和为 1）。因此**目标谱必须也是纯归一化密度**，
> 绝不能再做对数压缩——否则预测和目标处于不可对齐的数值空间，损失无法下降。
> 对数压缩**只作用于网络输入**（探测器 A），仅作为 6 个数量级动态范围的表示技巧。
> `build_preprocessors(cfg)` 会根据输出头自动构造正确的 `(input_pre, target_pre)` 对。

---

### `src/synth.py` — 合成宽谱生成

- **`make_synthetic_pair(mono_A, mono_B, weights)`**：用相同权重对 A、B 的单能谱做加权和，
  保证物理一致性。
- **`sample_weights(n_energies, family, rng)`**：按策略采样权重：
  - `mono`：独热向量（退化为单能谱，保持端点在域内）
  - `sparse`：2~4 个活跃能量的随机组合
  - `dense`：Dirichlet(α=1) 均匀宽谱
  - `gcr_like`：幂律 E^(-2.7)，逼近真实 GCR 注量权重
- **`SynthGenerator`**：组合以上策略，可选 Poisson 统计噪声。

---

### `src/dataset.py` — PyTorch Dataset

- **`SyntheticMixtureDataset`**：每次 `__getitem__` 实时采样新权重并混合，
  每个 epoch 产生 `samples_per_epoch` 对新样本，防止过拟合。
- **`FixedMixtureSet`**：在初始化时用固定种子生成固定的验证/测试集，
  保证不同 run 之间指标可比。

---

### `src/model.py` — 1D U-Net

```
输入 (B, 1, 360)
  → 编码器 × depth 层（Conv1d → GroupNorm → GELU → MaxPool1d(2)）
  → 瓶颈层
  → 解码器 × depth 层（插值上采样 → 跳跃连接拼接 → Conv1d）
  → 1×1 投影
  → softmax（输出为归一化概率密度）
输出 (B, 1, 360)
```

- **跳跃连接**保留峰位和尾部细节。
- **GroupNorm + GELU** 在小批量下稳定训练。
- **softmax 输出头**保证预测始终非负且归一化，使 EMD 计算无需额外处理。

---

### `src/losses.py` — 组合损失

```
L = w_mse × MSE(pred, target) + w_emd × EMD(pred, target)
```

**EMD（1-Wasserstein 距离）** 的闭合形式（仅适用于 1D 有序轴）：

```
EMD(p, q) = Σᵢ |CDF_p(i) − CDF_q(i)| × dᵢ
```

通过 `cumsum + abs + sum` 实现，完全可微分，直接惩罚谱形的质心偏移，
避免 MSE 单独使用时产生的峰位模糊和尾部压平问题。

---

### `src/train.py` — 训练主循环

- 优化器：AdamW + 余弦退火学习率调度
- 混合精度（AMP，GPU 可用时自动启用）
- 梯度裁剪（`grad_clip=1.0`）
- 每 epoch 在验证集上评估，保存最优检查点 `checkpoints/best.pt`
- TensorBoard 记录 `train/val` 的 `total / mse / emd` 三分量损失

---

### `src/evaluate.py` — 评估

在留出集和 GCR 真实谱上计算：

| 指标 | 说明 |
|------|------|
| **EMD** | 主指标，谱形保真度（与训练损失一致） |
| **峰位误差** | 预测峰 vs 真实峰的 bin 偏差 |
| **尾部保真度** | 高能尾（>P90）区域的 Pearson 相关系数 |
| **MSE / MAE** | 逐 bin 精度 |

`evaluate_gcr()` 是决定性测试：加载 `GCR_spectrum.xlsx`，重采样，以 SiC 为输入预测 TEPC，
与文件中真实 TEPC 列对比——这是模型是否真正泛化到宽谱的最终答案。

---

### `src/plots.py` — 可视化

- **`plot_spectrum_comparison`**：对数-对数坐标下叠加显示输入（SiC）、真实 TEPC、预测 TEPC。
- **`plot_cdf_comparison`**：CDF 对比图，使 EMD 距离可视化（两条曲线之间的面积即为 EMD）。

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/Wangzhaotian725/1D_U-Net.git
cd 1D_U-Net

# 安装依赖（建议使用虚拟环境）
pip install -e .
```

**依赖项**：`torch >= 2.0`、`numpy`、`pandas`、`openpyxl`、`omegaconf`、
`matplotlib`、`tensorboard`、`scipy`、`pytest`

---

## 运行步骤

### 第 1 步：准备数据

将训练文件放入 `data/raw/`，GCR 文件放入 `data/deploy/`，然后运行：

```bash
python scripts/prepare_data.py
```

输出示例：
```
找到 11 个原始文件
规范能量网格：360 个 bin，0.0100 ~ 9623.00 keV
解析到的能量 (MeV): [200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 2000.0, 4000.0, 7000.0, 10000.0]
已保存：data/processed/energy_grid.npy  data/processed/mono_spectra.npz
```

### 第 2 步：快速验证管线（可选）

```bash
python scripts/run_train.py --config configs/baseline.yaml --fast-dev-run
```

仅跑 2 个 batch × 1 epoch，用于验证整条管线无报错。

### 第 3 步：完整训练

```bash
python scripts/run_train.py --config configs/baseline.yaml
```

训练过程中可用 TensorBoard 实时查看损失曲线：

```bash
tensorboard --logdir runs/
```

### 第 4 步：在留出集上评估

```bash
python scripts/run_eval.py --config configs/baseline.yaml --ckpt checkpoints/best.pt
```

输出 `results/eval/metrics.json` 及谱形对比图、CDF 对比图。

### 第 5 步：在真实 GCR 谱上推理（决定性测试）

```bash
python scripts/run_gcr.py --config configs/baseline.yaml --ckpt checkpoints/best.pt
```

输出 `results/gcr/gcr_metrics.json`、`gcr_spectrum_comparison.png`、`gcr_cdf_comparison.png`。
**这是头号结果**——模型对真实 GCR 宽谱的预测与真实 TEPC 谱的对比。

### 第 6 步：运行单元测试

```bash
pytest -q
```

全部 33 个测试应通过。

---

## 数据格式说明

| 属性 | 训练文件（11 个） | GCR_spectrum.xlsx |
|------|------------------|-------------------|
| 列名 | `keV`, `SiC_SBD`, `TEPC` | `E(keV)`, `SiC`, `TEPC` |
| 列顺序 | 能量, SiC, TEPC | 能量, TEPC, SiC（A/B 互换）|
| Bin 数 | 360 | 420 |
| 能量范围 | 0.01 ~ 9623 keV | 0.01 ~ 98099 keV |
| 表头位置 | 第 2 行（skiprows=1）| 第 1 行（skiprows=0）|

代码通过**列名别名映射**自动处理上述差异，不依赖列位置。

---

## 主要调参建议

| 参数 | 位置 | 效果 |
|------|------|------|
| `loss.w_emd` vs `loss.w_mse` | `baseline.yaml` | 增大 w_emd 改善谱形，增大 w_mse 改善逐 bin 精度 |
| `synth.mixture_families` | `baseline.yaml` | 加入 `gcr_like` 使训练分布更接近 GCR |
| `preprocessing.log_compress` | `baseline.yaml` | 关闭后网络更关注峰区，忽视长尾 |
| `model.depth` | `baseline.yaml` | 增加深度提升感受野，但需更多训练数据 |
| `data.heldout_energies_MeV` | `baseline.yaml` | 调整留出能量以测试不同插值难度 |

---

## 注意事项

> **泛化风险提示**：模型在 11 个单能谱上训练，部署在 GCR 宽谱上。
> 这依赖**探测器响应线性叠加**在仿真中成立的假设。
> 留出能量集上的 EMD 是代理指标，GCR 真实谱对比才是最终验证。
> 在充分信任 GCR 预测结果之前，请仔细核查 `run_gcr.py` 的输出图像。

---

## 许可证

MIT License
