# v0.7 预注册配置（在GCR评估前冻结）

**日期**: 2026-06-26
**分支**: claude/experiment-v07
**配置文件**: configs/experiment_v7.yaml

## 核心假设（来自v0.6总结）

v0.6证伪了"完全去除skip连接可消除双峰伪影"的假设。正确的认识是：

- Skip连接提供了空间感应偏置（spatial inductive bias），对GCR等域外样本的泛化至关重要
- 双峰伪影是skip连接泄漏的叠加效应，应通过**正则化抑制**而非**架构切除**来解决
- v0.4 U-Net（EMD=5.3, peak_err=8）远优于v0.6 Transformer（EMD=21.8, peak_err=24）

## v0.7 核心变更

| 项目 | v0.5/v0.6 | v0.7 | 变更原因 |
|------|-----------|------|----------|
| 架构 | UNet1D / Transformer | **AttnUNet1D** | 保留skip空间偏置，同时用注意力门控抑制峰位泄漏 |
| w_unimodal | 0.5 | **2.0** | 4倍增强，更强力惩罚双峰输出 |
| w_peak | 2.0 | **3.0** | 加强峰位对准约束 |
| lambda_spr | 0.5 | **1.0** | SPR与peak_err等权，更重视消双峰 |
| skip_gate_scale | — | [1.0, 1.0, 1.0] | 三级skip全部启用注意力门 |

## 选择分数（v0.7）

```
selection_score = peak_err + 0.3 × region_emd + 1.0 × secondary_peak_ratio
```

所有指标仅在 held-out 能量点（100, 600, 2000, 7000 MeV）上计算，不使用任何部署谱信息。

## 反泄漏约束（继续执行）

- 部署谱（`GCR_spectrum`）来源与物理成因视为"未知"，不假定为银河宇宙线或任何特定来源。
- 训练、合成、超参选择、版本间方向决策，均不得使用部署谱的任何信息（掩码/盲测纪律）。
- 本文件必须在执行 `run_gcr.py` **之前**提交。

## 冻结超参数

| 参数 | 值 |
|------|----|
| arch | unet_attn |
| skip_gate_scale | [1.0, 1.0, 1.0] |
| head | softplus_renorm |
| base_channels | 32 |
| depth | 3 |
| w_unimodal | 2.0 |
| w_peak | 3.0 |
| w_emd | 2.0 |
| w_mse | 0.5 |
| lambda_region | 0.3 |
| lambda_secondary_peak_ratio | 1.0 |
| lr | 2e-4 |
| epochs | 600 |
| early_stop_patience | 80 |

## GCR评估结果（运行 run_gcr.py 后填写）

- [ ] GCR EMD：待填
- [ ] GCR peak_pos_error（bins）：待填
- [ ] Secondary peak ratio：待填
- [ ] 双峰伪影是否消除：待填
- [ ] 与v0.4基准对比：待填
