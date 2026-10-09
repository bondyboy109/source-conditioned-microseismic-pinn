# 新低秩源条件算子：实验与成本汇总

本汇总只对应 source_conditioned_low_rank.py；不混用之前 MoE/PINN 的数字。未修改论文正文。

## 前向拟合与清洁定位

16 个开发事件的平均相对 L2 从 1.0493 降至 0.7567，改善 27.88%。这是开发集结果，不是独立测试性能。

清洁定位尝试 8 个事件，混合目标能运行 7 个，1 个因有效到时不足而无法运行。下表只比较同一批 7 个事件。

| 样本 | 波形平均 eloc (m) | 混合平均 eloc (m) | 波形中位 eloc (m) | 混合中位 eloc (m) | 改善事件 |
|---|---:|---:|---:|---:|---:|
| 同一批清洁事件 | 833.4 | 506.1 | 658.3 | 394.5 | 6/7 |

## 噪声：配对比较与不可用率

不可用率指有效到时不足而未能执行混合搜索，不等同于定位误差超过某阈值。误差只在双方均可运行的相同事件上计算；因此它仍是条件误差，不包含未运行事件。

| SNR (dB) | 配对数/尝试数 | 混合不可用率 | 波形平均 eloc (m) | 混合平均 eloc (m) | 混合改善数 |
|---:|---:|---:|---:|---:|---:|
| 20 | 18/24 | 25.0% | 826.8 | 781.1 | 7/18 |
| 10 | 15/24 | 37.5% | 857.8 | 771.7 | 8/15 |
| 5 | 8/24 | 66.7% | 951.0 | 1011.0 | 4/8 |

## 跨地质模型：误差与搜索成本

4 个已有合成事件的事后零样本审计。各方法使用相同接收器和 FMM 场；搜索耗时不含建场、模型加载和共享记录准备。神经两分支共享的粗搜耗时分别计入假设独立运行的搜索成本，不能把两列相加当作脚本总耗时。

| 方法 | 中位 eloc (m) | 平均 eloc (m) | 平均搜索耗时 (s/事件) |
|---|---:|---:|---:|
| 波形 | 1126.9 | 946.8 | 5.0232 |
| 波形 + FMM | 163.8 | 208.7 | 4.6458 |
| 纯 FMM | 86.2 | 105.5 | 0.0754 |

25 个接收器的 FMM 场重新建场：36.71 s（CPU）。与原缓存最大差值 0.0 s，复算定位误差最大变化 0。
模型参数量 70,160；联合评估峰值 CUDA 已分配显存 328.6 MiB（不是整机显存占用，也不是逐分支峰值）。
已有 Overthrust 开发运行记录：FMM 建场 37.02 s，低秩模型训练 19.51 s。跨地质审计仅重建 FMM 场，不重新训练网络。

本次混合搜索约为纯 FMM 的 61.6 倍耗时，且定位误差更高。两者均需 FMM 场，混合方法还需模型训练；因此本比较没有有利的有限盈亏平衡事件数。与连续精度匹配 FD 的盈亏平衡仍未测量。单次、固定顺序的计时只能描述本次运行，不能给出稳定速度提升或统计显著性。

环境：{'python': '3.10.21', 'numpy': '2.2.6', 'torch': '2.5.1+cu121', 'cuda': '12.1', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'platform': 'Windows-10-10.0.26200-SP0', 'gpu': 'NVIDIA GeForce RTX 4060 Laptop GPU'}

## 迁移学习

source_to_target and target_to_target each use 1500 first-stage plus 1500 target-stage updates. target_only_1500 uses only 1500 total updates. Only the two staged arms match the total update budget; the shorter target-only arm is a target-stage-budget control.

源域→目标域平均定位误差 565.1 m；同预算的目标域→目标域为 465.2 m。当前结果不支持源域预训练提高定位精度。较短 target-only 的 503.0 m 不能作为匹配总成本的对照。

## 当前可写结论与下一步

可写：显式源条件低秩接口改善开发集波形拟合；到时约束在清洁数据及一个跨地质审计中改善波形定位。不能写：优于 FMM、具有计算加速优势、源域迁移已提高定位精度或噪声下普遍改善。

到时拾取对照显示，固定阈值虽然提高了“返回拾取”的数量，却造成大量误拾取和更大的定位误差；MAD 稳健拾取的可用率较低但定位误差更小。因此当前结果应解释为可靠性和可用率之间的权衡，而不是简单的台站数量问题。软权重开发筛查显示，低于 0.5 的置信度门槛几乎不剔除台站，0.5 门槛又会在 10 dB 下保留不足 6 个台站；这不足以支持修改正式定位目标。当前确认实验继续使用预注册的 MAD 拾取。当前重复使用的事件只可作开发/事后分析。论文文本压缩留到实验结束。

## 输入记录

- `source_conditioned_low_rank_development_results.json`；SHA256：`8ab4e31656441e3da1c436edee7f0520d50362cb34749bb77e94ea1ff9c7a4f4`
- `source_conditioned_low_rank_localization_results.json`；SHA256：`391a97c9be37124a59fcef55e28fbaa0c94d451df11384cc087a1406a5258379`
- `source_transfer_low_rank_results.json`；SHA256：`ce66eec063aa055d6633fce302c6071a7dda75de389672192257ba07ae3fa646`
- `source_conditioned_low_rank_noise_results.json`；SHA256：`5300d256f68649850bdc6e7815a13d9fb4a21f121b45ec330411abc3d216bdf6`
- `source_conditioned_low_rank_cross_geology_results.json`；SHA256：`bbaf98aa1323d2ac3b0bbac8eb3f35a5c85e0391b8edb2fab4d0bf33b2c974d4`
- `source_conditioned_low_rank_cross_geology_cost_results.json`；SHA256：`456d875eff4a906a2c53156291c4f6813ff5268a75fa46d1eacd43ea79fdb37e`
