# 微震定位：可微 FMM 对齐的混合专家接口

作者：Ruyun Tian、Xiaopeng Wu。论文题名：A Differentiable FMM-Aligned Mixture-of-Experts Interface for Microseismic Localization。

当前主方法由接收记录监督的源条件波形网络、互易 FMM 到时缓存、可微三线性源坐标插值、三个联合训练专家及区域门控、波形与到时联合目标组成。它已不是纯 PDE 残差训练的 PINN；旧代码保留用于审计。

最终实验为 33 个训练源、16 个开发源、8 个确认源，5 Hz、已知速度。确认观测为 25×1001，dt=0.002 s、0–2 s。定位误差中位数：waveform-only 2400.2 m、hybrid 362.9 m、FMM 276.8 m；小于 500 m 的事件分别 0/8、5/8、7/8；三个分支的流程失败均为 0/8。Hybrid 优于波形单项，但没有证明优于 FMM 或更快，也没有证明正向迁移增益。

## 操作

在仓库根目录、已有 PyTorch 环境中逐条运行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

从 [Release 附件](https://github.com/bondyboy109/source-conditioned-microseismic-pinn/releases/tag/research-artifacts-2026-10-09) 下载三个 ZIP 到同一文件夹，然后将下例路径换成实际下载文件夹：

```powershell
python scripts/install_release_assets.py --asset-dir "E:\downloaded-assets"
python scripts/verify_package.py --require-assets
python scripts/replay_confirmation.py --check-inputs
python scripts/replay_confirmation.py --events 1
python scripts/replay_confirmation.py
```

重放结果写到 outputs/reproduction，不覆盖论文归档。最终重放依赖接收器 FMM 缓存，不需要原始 Overthrust 数组。另两份附件用于迁移/地质审计。大型缓存不放入普通 Git 提交。

## 必须保留的说明

历史 add81f… 定位源码未恢复，现有代码为 174a5…；当前重放不是新的独立确认，也不代表历史逐字节复现。协议与结果保持原始字节，差异见 [当前审计](docs/PACKAGE_AUDIT.md)。旧 reliability 结果已标 legacy，使用 v2。

M=5 属于单独 carrier 集成；nominal 95% ensemble-dispersion interval 未校准，只反映随机初始化离散性，不是最终三专家 MoE 的 UQ 验证。

[英文说明](README.md) 给出结构与参数；[实验索引](experiments/INDEX.md) 区分 legacy diagnostic、development audit、independent confirmation。仓库保留原有私有可见性，访问者需授权。未擅自添加软件许可证，原始 Overthrust 数据按原条款取得。
