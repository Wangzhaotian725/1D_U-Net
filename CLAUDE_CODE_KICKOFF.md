# Claude Code 启动指令

> 配套文档：`PROJECT_PLAN.md`（完整规格，逐节实现）。本文件只说明如何起步。

## 任务

按 `PROJECT_PLAN.md` 搭建一个完整、可运行的 Python 项目：用 **1D U-Net** + **MSE+EMD 组合损失**，
学习从探测器 A（SiC）的能量沉积谱预测探测器 B（TEPC）的能量沉积谱。`PROJECT_PLAN.md` 是唯一事实来源，
按其中各节实现，不要自行更改架构决策。

## 仓库

- 名称：`detector-spectrum-unet`
- 许可证：MIT（除非另有说明）
- 初始化 git，完成并通过冒烟测试后提交并打标签 `v0.1.0-baseline`

## 数据放置（我会提供这些文件）

```
data/raw/      <- 11 个训练文件：200MeV.xlsx, 300MeV.xlsx, 400MeV.xlsx, 500MeV.xlsx,
                  600MeV.xlsx, 800MeV.xlsx, 1000MeV.xlsx, 2000MeV.xlsx, 4000MeV.xlsx,
                  7000MeV.xlsx, 10000MeV.xlsx
data/deploy/   <- GCR_spectrum.xlsx（部署目标，仅推理用，绝不参与训练）
```

**注意两个易错点**（详见 PROJECT_PLAN 第 2 节）：
1. 所有 xlsx 第一行是干扰行，真表头在第二行 —— 用 `skiprows=1`。
2. GCR 文件与训练文件**列名、列顺序、网格都不同**：按**列名别名**映射（A=`sic_sbd`/`sic`，
   B=`tepc`，能量=`kev`/`e(kev)`），并把 GCR 谱**重采样到 360 道标准网格**。绝不按列位置读取。

## 执行顺序

1. **先读** `PROJECT_PLAN.md` 全文，再开始写代码。
2. 按第 9 节的目录结构搭建骨架（`src/`、`scripts/`、`tests/`、`configs/`、CI）。
3. 实现各模块，**优先把 `tests/` 写出来并让其通过**，尤其是：
   - `test_data`（别名映射 + 重采样恒等/质量守恒）
   - `test_synth`（响应线性：one-hot 还原单能谱；同一权重作用于 A、B）
   - `test_losses`（EMD 在相同分布为 0、单道平移有已知值、梯度可回传）
   - `test_model`（输出形状 `(B,1,360)`、无 NaN、softmax 头归一）
4. 跑通第 12 节的冒烟测试全部 6 步后再提交：

```bash
pip install -e .
pytest -q
python scripts/prepare_data.py
python scripts/run_train.py --config configs/baseline.yaml --fast-dev-run   # 先快速验证管线
python scripts/run_train.py --config configs/baseline.yaml                  # 完整训练
python scripts/run_eval.py  --config configs/baseline.yaml --ckpt runs/<ts>/best.pt
python scripts/run_gcr.py   --config configs/baseline.yaml --ckpt runs/<ts>/best.pt
```

## 完成判据

第 12 节六步全部通过；`run_gcr.py` 输出对真实 TEPC GCR 谱的全套指标（EMD、峰位误差、尾部保真度）
和叠加/CDF 图 —— 这是头号结果。一切通过后提交并打 `v0.1.0-baseline` 标签。

## 边界

- 本期只做 U-Net baseline。**不要**引入 GAN / 扩散 / Transformer（留作下一期，见第 13 节）。
- 若留出能量的 EMD 显示"响应线性"在仿真中不成立，先停下来在 README 标注此发现，再讨论是否调整合成策略。
