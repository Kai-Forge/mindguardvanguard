# 御郁先锋便携包：来源与许可证

本包为项目演示与本地使用分发。各第三方组件保留各自许可证；不统一改为项目许可证。

| 组件 | 作者 / 来源 | 随包许可证位置 |
|---|---|---|
| CPython 3.13.1 | Python Software Foundation, https://www.python.org/ | runtime/LICENSE.txt |
| PyTorch / torchaudio 2.6.0 CPU | PyTorch contributors, https://pytorch.org/ | runtime/Lib/site-packages/torch*、torchaudio* 的 LICENSE / dist-info |
| emotion2vec+ base | emotion2vec contributors / Ziyang Ma et al.; https://github.com/ddlBoJack/emotion2vec ; https://huggingface.co/emotion2vec/emotion2vec_plus_base | app/trimodal/assets/emotion2vec_plus_base/MODEL_LICENSE、README.md、sources.json |
| FunASR | Alibaba DAMO Academy / FunASR contributors, https://github.com/modelscope/FunASR | runtime/Lib/site-packages/funasr-*.dist-info |
| MobileFaceNet 表情识别 | OpenCV Zoo contributors, https://github.com/opencv/opencv_zoo | app/trimodal/assets/LICENSE-opencv-zoo.txt、README-expression.md、sources.json |
| YuNet | Shiqi Yu / OpenCV Zoo contributors | app/trimodal/assets/LICENSE-yunet.txt、sources.json |

其他 Python 依赖的版本见 runtime-packages.json；原始许可文件与发行包 metadata 一并保留在 runtime/Lib/site-packages。
Windows C/C++ 运行库保留 Microsoft 版权：vcruntime140 系列来自 CPython 的随包运行库；msvcp140.dll 是随 NumPy Windows wheel 分发的同一二进制的应用本地副本，原文件同时保留在 numpy.libs 中。
emotion2vec 权重使用模型卡所指向的模型协议，不能将代码仓库的 MIT 许可证替代权重许可。模型名称和出处予以保留。

运行环境采用随应用分发模式：
- Python 官方说明：https://docs.python.org/3.13/using/windows.html#the-embeddable-package
- PyTorch 官方 CPU 版本来源：https://pytorch.org/get-started/previous-versions/

随包仅含推理所需代码和权重；不含 S70 数据库、历史训练结果集合、采集录音或用户账户信息。
