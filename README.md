# 探测器能谱转换 — 1D U-Net

> 利用 **1D U-Net** 神经网络，从 SiC 探测器的能量沉积谱预测 TEPC 探测器的能量沉积谱。
> 训练数据为 13 组单能质子谱（90 MeV ~ 10000 MeV），部署目标为一条独立的宽谱
> `GCR_spectrum`（文件沿用历史命名，**其形状成因记为"未知"**）。

---

## ⚠️ v0.4 重要更正与无泄露约束（凌驾一切性能目标）

本节更正早期文档中的若干表述，并声明本项目的两条最高约束。后续所有版本以此为准。

1. **部署谱来源未知**：`GCR_spectrum` **不假定**为银河宇宙线或任何特定物理来源。
   早期文档中"银河宇宙射线""真实测量谱""线性假设已由该谱验证""其质心约 5 keV"等
   表述**一律作废**——它们要么是未经证实的物理假定，要么是读取了部署谱并用其指导设计（属泄露）。

2. **训练/合成/调参绝不使用部署谱信息**：训练数据、合成分布、超参选择、版本间方向决策，
   均不得使用 `GCR_spectrum` 的任何信息。部署谱在所有配置冻结后，由 `run_gcr.py`
   **跑且仅跑一次**，结果不得回头触发任何调整（见 `docs/EXPERIMENT_PLAN_v3.md` 之后的 v0.4 计划）。

3. **新增 90/100 MeV 单能谱的合法性**：这是**纯训练数据扩展**——目的是扩大能量覆盖、
   提升对未见能量的插值鲁棒性。**不得**论证为"更接近部署谱低能端"（那将构成对部署谱的观察）。

4. **合成族保持中性**：`powerlaw_neutral` 的指数在通用随机范围 `[-3, 0]` 内采样，
   覆盖各种单调形状，**不锚定**任何观测到的目标谱指数（由 `tests/test_leakage.py` 锁定）。

5. **对"完美吻合"的现实校准**：目标谱与单能谱均为带泊松涨落的有限统计仿真，
   "完全吻合"在统计意义上不可达。合理目标是"剩余误差降至统计涨落地板、且无系统性偏移"。
   只有 13 个能量锚点，对完全未知来源的宽谱，泛化能力存在不可消除的不确定性——
   **本项目承诺"在可证明无泄露的前提下给出最优且诚实的预测"，而非"完美吻合"。**

> 关于 v0.4 的留出集重设计、非归一化输出头、宽谱验证集、预注册冻结流程，
> 详见 `docs/EXPERIMENT_SUMMARY_v1_v2_v3.md` 与上传的 v0.4 实验计划。

---

## 项目背景

两种探测器（SiC_SBD 和 TEPC）在相同辐射场中会产生不同的能量沉积谱。两者的谱形均属
Gaussian-Vavilov-Landau 族（非对称、重尾），没有简洁的解析关系，因此采用神经网络学习
从探测器 A（SiC）到探测器 B（TEPC）的映射。

### 核心挑战：从单能谱泛化到宽谱

模型只用 13 条**窄峰谱**训练，却要在**平滑宽谱**的部署谱上推理。
解决方案依赖一个**工作假设**（探测器响应对注量线性叠加）：

> **假设：探测器响应对粒子注量近似线性叠加。**
> 混合辐射场的谱 ≈ 各单能成分谱的注量加权和，且对两个探测器独立成立。
> 此假设是合成方法的前提，**不**以部署谱来"验证"它。

因此，同一组随机权重作用于 SiC 各单能谱和 TEPC 各单能谱，
得到的混合对在该假设下构成一致的 A→B 配对（`src/synth.py`）。
通过中性随机权重（dirichlet / loguniform / powerlaw_neutral）可生成覆盖
**宽谱形状空间**的训练样本，缩小训练/部署的形状差距——但权重分布**不参照**部署谱形状。

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

> **重要：输入与目标使用不同的预处理，且必须与输出头配套。**
> - **归一化输出头**（`softmax` / `softplus_renorm`）产生归一化密度（和为 1），
>   因此**目标谱必须是纯归一化密度**，绝不能再做对数压缩。
> - **非归一化输出头**（v0.4 的 `softplus` / `relu`）可输出真正的零，
>   因此**目标谱必须做对数压缩**，使预测与目标处于同一数值空间。
> 对数压缩始终作用于网络输入（探测器 A），作为 6 个数量级动态范围的表示技巧。
> `build_preprocessors(cfg)` 按输出头自动构造正确的 `(input_pre, target_pre)` 对，
> 该配套关系由 `tests/test_leakage.py::test_preprocessor_head_match` 锁定
> （这是 v0.1 致命 bug 的镜像，务必保持）。

---

### `src/synth.py` — 合成宽谱生成

- **`make_synthetic_pair(mono_A, mono_B, weights)`**：用相同权重对 A、B 的单能谱做加权和，
  保证物理一致性。
- **`sample_weights(n_energies, family, rng)`**：按**中性**策略采样权重
  （均不参照部署谱形状）：
  - `mono`：独热向量（退化为单能谱，保持端点在域内）
  - `sparse_k`：`sparse_k_range` 个活跃能量的随机组合（`sparse` 为兼容别名）
  - `dirichlet_uniform`：对称 Dirichlet，α 从 `dirichlet_alpha_choices` 随机选取
  - `loguniform`：各能量独立的对数均匀权重
  - `powerlaw_neutral`：幂律 `E^α`，**指数 α 在通用随机范围 `[-3, 0]` 内采样**，
    覆盖从硬谱到软谱的连续体；**非固定物理指数、不锚定任何观测谱**
  - 注：早期的 `gcr_like`（固定 E^-2.7）已移除，调用会抛 `ValueError`
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

全部测试应通过（含 `tests/test_leakage.py` 的反泄露套件）。

---

## 数据格式说明

| 属性 | 训练文件（13 个） | GCR_spectrum.xlsx（部署，来源未知） |
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
| `synth.mixture_families` | `experiment_v4.yaml` | 增减中性合成族（均不参照部署谱） |
| `model.head` | `experiment_v4.yaml` | 归一化头 vs 非归一化头（影响目标空间，见预处理说明） |
| `preprocessing.log_compress` | `experiment_v4.yaml` | 关闭后网络更关注峰区，忽视长尾 |
| `model.depth` | `experiment_v4.yaml` | 增加深度提升感受野，但需更多训练数据 |
| `data.heldout_energies_MeV` | `experiment_v4.yaml` | 调整留出能量以测试不同插值难度 |

---

## 注意事项

> **泛化风险提示**：模型在 13 个单能谱上训练，部署在一条来源未知的宽谱上。
> 这依赖**探测器响应近似线性叠加**这一**工作假设**（不以部署谱来验证它）。
> 留出集 / 宽谱验证集上的 `composite_wide` 是合法的选择指标，但它**只是代理**——
> 留出能量再怎么组合，也无法完全等价于真正未知来源的目标谱。
> 部署谱评估（`run_gcr.py`）在所有配置冻结后**只跑一次**，其结果不得回头触发调参。

---

## 许可证

MIT License
