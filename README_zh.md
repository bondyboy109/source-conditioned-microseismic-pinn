# 走时约束 MoE 微地震定位：最新版

本包对应当前论文，主目录展示 **waveform-only 与 hybrid** 两条神经分支。
当前方法由接收器互易程函场、可微三线性插值、源条件特征、三个联合专家与区域门控组成。
最终网络使用接收器波形监督；hybrid 在波形目标中加入到时约束。

## 包含内容

- 33个训练、16个开发、8个确认和4个独立地质观测文件及相对路径清单。
- 最终模型权重；代码、协议和数据 SHA-256；最新九张论文图。
- 只运行两条神经分支的当前版本回放入口、独立新训练入口。
- 完整研究记录单独存入 archive/supporting-research-records.zip，原始数值不改写。

## 使用步骤

激活已安装PyTorch的环境，在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/verify_package.py
```

从[数据附件页面](https://github.com/bondyboy109/source-conditioned-microseismic-pinn/releases/tag/research-artifacts-2026-10-09)
下载 receiver_fmm_fields_stride2.npy.zip 和 independent_layered_undulating.npy.zip。
假设下载到 E:\moe-assets：

```powershell
python scripts/install_release_assets.py --asset-dir E:\moe-assets
python scripts/verify_package.py --require-assets
python scripts/replay_confirmation.py --check-inputs
python scripts/replay_confirmation.py --events 1
python scripts/replay_confirmation.py
```

新训练使用不同输出目录，默认3000次更新：

```powershell
python scripts/train_moe.py --check-inputs
python scripts/train_moe.py --output outputs/new_training/moe_run1
```

重新绘制当前单栏图七、图八：`python scripts/render_neural_figures.py`。
图件输出到新的 `outputs/figure_reproduction/` 目录。

## 结果与归属

8个确认源全部保留：waveform-only定位中位误差2400.2 m，hybrid为362.9 m，
两条神经分支之间减少84.9%，所有8个事件的定位误差均改善。
确认记录为25台×1001点，dt=0.002 s，0–2 s；失败门槛与500 m精度门槛分开统计。
当前方法在5 dB下有11/24门槛失败，40%台站缺失下有7/24；这些限制仍保留。

回放属于已检查事件的当前实现复跑，并不是新的独立确认。
原历史冻结脚本add81f…没有恢复；现有174a5…实现明确标记为current。
历史协议和完整原始结果留在独立归档，不为展示重新修改。
迁移/单独M=5载波集成属于另行审计，未校准集成区间不能用于验证最终MoE的不确定性。
图六空间快照来自FD参考计算，最终MoE仅输出接收器波形。

外部Overthrust速度数组不随本包分发，重新生成FD或建立缓存前需按原许可获取。
使用已存缓存做定位和训练无需该外部数组。大附件与既有Release哈希一致。
软件许可证尚未选择，仓库可见性和访问权限由原仓库设置决定。
