# 御郁先锋 Windows CPU 便携版

完整程序位于 [`Vanguard-Windows-CPU`](Vanguard-Windows-CPU) 文件夹。模型权重和较大的运行库使用 Git LFS 存储。

在 Windows 安装 Git for Windows（包含 Git LFS）后运行：

```powershell
git lfs install
git clone https://github.com/Kai-Forge/mindguardvanguard.git
cd mindguardvanguard\Vanguard-Windows-CPU
.\Start.cmd
```

请等待 `git clone` 完成 LFS 文件下载后再运行。EGG 设备未接入时，网页上的 EGG 页面不会显示推理结果。
