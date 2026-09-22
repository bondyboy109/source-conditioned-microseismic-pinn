# 上传到 GitHub

当前文件夹已经初始化为 Git 仓库，并排除了虚拟环境、原始 Overthrust
文件、大型 NumPy 数组、模型检查点和运行输出。

## 1. 在 GitHub 网站创建空仓库

登录 GitHub，点击 **New repository**，建议填写：

```text
Repository name: source-conditioned-microseismic-pinn
Description: Source-conditioned frequency-progressive PINN for 3-D microseismic source imaging
Visibility: Private（检查完成后再改 Public）
```

不要勾选自动创建 README、`.gitignore` 或 LICENSE，因为本地已经有这些文件。

## 2. 复制仓库地址

HTTPS 地址形式如下：

```text
https://github.com/你的用户名/source-conditioned-microseismic-pinn.git
```

## 3. 在 PowerShell 上传

```powershell
Set-Location "E:\source-conditioned-microseismic-pinn"

git remote add origin https://github.com/你的用户名/source-conditioned-microseismic-pinn.git

git push -u origin main
```

GitHub 若要求登录，应使用浏览器授权或 Personal Access Token，不能把密码或
Token 写入代码、README、命令脚本或提交记录。

## 4. 上传后检查

GitHub 页面中应看到源码和说明文件，但不应看到：

- `.venv/`
- `overthrust.vites`
- `vp_model.npy`
- 训练/验证/测试数组
- `.pth` 检查点
- `outputs/` 中的运行结果
- 个人电脑绝对路径

大型数据可在论文接受后根据许可证放到 Zenodo、Figshare 或其他科研数据仓库，
再把公开链接写入 `data/README.md`。

## 5. 后续更新

修改代码后使用：

```powershell
git status
git add 文件名
git commit -m "Describe the change"
git push
```

不要使用 `git add` 上传真实数据、模型或含身份信息的日志。
