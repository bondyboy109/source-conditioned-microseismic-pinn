# 源条件 PINN 微震震源成像

本仓库实现八输入源条件算子：

```text
u_theta(x, y, z, t; xs, ys, zs, ts)
```

震源位置与起震时刻显式进入网络，因此冻结网络后仍可对
`xs、ys、zs、ts` 四个参数求梯度。

## 文件结构

- `source_conditioned_pinn.py`：网络、物理损失和检查点接口。
- `train_source_conditioned_pinn.py`：频率递进训练。
- `invert_source_conditioned.py`：四维粗搜和有界局部优化。
- `data/README.md`：速度模型和观测数据格式。
- `legacy/`：旧四输入方法，仅用于失败审计和负对照。
- `outputs/`：本地模型和结果目录，Git 不上传其中内容。

## 安装

推荐 Python 3.10。已有 PyTorch 环境时执行：

```powershell
python -m pip install -r requirements.txt
```

## 放置速度模型

根据 SEG/EAGE 的许可获取 Overthrust 数据，并准备：

```text
data\velocity\vp_model.npy
```

仓库不包含原始 Overthrust 文件、训练数组或模型检查点。

## 最小测试

```powershell
python .\train_source_conditioned_pinn.py `
  --velocity .\data\velocity\vp_model.npy `
  --frequencies 5 --epochs 1 --n-phys 16 --n-ic 8 `
  --n-bc-per-face 2 --save-every 1 --output .\outputs\smoke
```

## 正式训练

```powershell
python .\train_source_conditioned_pinn.py `
  --velocity .\data\velocity\vp_model.npy `
  --frequencies 5 10 15 20 30 40 60 `
  --epochs 2000 --n-phys 4096 --n-ic 2048 `
  --n-bc-per-face 341 --save-every 200 `
  --output .\outputs\source_conditioned
```

8 GB 显存可能需要减小批量。论文必须记录实际运行参数。

## 反演

```powershell
python .\invert_source_conditioned.py `
  .\outputs\source_conditioned\conditioned_60Hz.pth `
  .\data\observations\event_01.npz `
  --output .\outputs\source_conditioned\event_01.json
```

观测数据必须来自独立正演或真实记录，不能使用旧均速模板替代。
