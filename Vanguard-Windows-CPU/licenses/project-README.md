# 图像 + 语音 + EGG 独立分类及线性加权融合

新增：本地实时网页运行 `python -m trimodal.server`，打开 http://127.0.0.1:8765 。支持电脑摄像头、麦克风、分模态展示与本地窗口采集；EGG实时暂无接入。详见 [DASHBOARD.md](DASHBOARD.md)。

本模块只做推理，不训练、不调参、不修改历史实验。默认三路等权。

2026-10-30 更新：默认语音已替换为官方 emotion2vec+ base（16kHz），旧Speech-only仅保留为显式回退。新增运行环境与类别映射详见 [OPEN_SPEECH.md](OPEN_SPEECH.md)。下面的初始接入记录中，P1语音配置及8kHz语音说明仅适用于legacy后端。

## 模型

| 路线 | 实际使用 | 来源 |
|---|---|---|
| 图像 | YuNet 人脸检测及五点定位 + Progressive Teacher / MobileFaceNet 七类表情 ONNX | OpenCV Zoo |
| 语音 | 单通道 CNN-BiLSTM，P1 seed42 的已训练 checkpoint | experiments_p1/p1_speech_seed42/best.pt |
| EGG | 单通道 EGG Log-Mel CNN-BiLSTM，P1 seed42 的已训练 checkpoint | experiments_p1/p1_egg_seed42/best.pt |

选用现有单模态权重是为了直接跑通，不按本次融合效果选择 seed。EGG 不是残差模型，也不读取语音通道。

人脸候选核查：
- 采用 https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition ：有现成 ONNX、原始对齐代码；README 声明该目录 Apache-2.0，保存仓库许可证。README 的目录内 LICENSE 链接在检索时未列出文件，保留该证据边界。YuNet 为 MIT，独立保留许可证。
- 备选 https://github.com/serengil/deepface ：仓库代码 MIT，未安装；本轮选择已有 ONNX 的较轻依赖方案，不对其全部第三方模型权重许可证作推断。

下载器会验证 ONNX 与上游 Git LFS SHA-256 一致，来源、大小及所有文件哈希在 assets/sources.json。若上游文件改变，拒绝覆盖现有文件。上游 alignment 代码未修改。主分支 URL 不是版本号，以已保存文件哈希识别本次版本。

## 标签和公式

统一顺序 `[negative, neutral, positive]`。
图像原顺序 `[angry, disgust, fearful, happy, neutral, sad, surprised]`，先 softmax：
- negative = angry + disgust + fearful + sad
- neutral = neutral
- positive = happy
- surprise 不强制映射，保留其概率，将其余概率除以其总和；这是以非惊讶类别为条件的概率。

语音/EGG 权重元数据的类别与模态会严格校验；按原训练的8kHz、64Mel、功率20log10、top80、变长padding-safe流程处理。图像按上游112×112五点对齐、RGB、除255后减0.5再除0.5。已检查ONNX末端是Gemm+Identity，输出为logits，只softmax一次。

最终 p = (w_image*p_image + w_speech*p_speech + w_egg*p_egg) / sum(weights)。
权重必须有限、非负且不全零。概率归一化不等于模型置信度已校准。

## 运行

以下 PowerShell 命令在项目目录运行：

```powershell
Set-Location 'D:\claudeprojetcs\voiceintelligence\mindguardvanguard'
..\venv\Scripts\python.exe -m pip install -r trimodal/requirements.txt
..\venv\Scripts\python.exe -m trimodal.download_models
..\venv\Scripts\python.exe -m unittest trimodal.test_fusion -v

# 使用自己的同人、同时间窗口的脸部图片和双通道WAV。
# 修改下面两个输入路径；--paired是调用者声明，程序不做身份验证。
..\venv\Scripts\python.exe -m trimodal.infer --image 'D:\capture\face.jpg' --wav 'D:\capture\speech_egg.wav' --paired --output trimodal/outputs/paired_result.json

# 权重顺序：图像、语音、EGG；例如改成1:2:1。
..\venv\Scripts\python.exe -m trimodal.infer --image 'D:\capture\face.jpg' --wav 'D:\capture\speech_egg.wav' --paired --weights 1 2 1
```

要求 WAV 恰好两路，0为语音、1为EGG，不自动将单声道复制成两路。
无脸时报错，多脸必须用 `--face-index` 明确选择；编号按人脸左上角从左到右排序，不能视为身份跟踪。
图像作为该语音窗口的代表帧；目前不实现摄像头、视频采样、身份跟踪或同步采集。
输出包含每路概率/原始分数、最终概率/类别、惊讶排除量、权重、模型及输入哈希。
已有输出文件拒绝覆盖。

## 工程演示与边界

本地S70没有配套图像。运行检查采用上游已标注多人示例图和一条S70录音，它们不是同一个人/同一窗口，必须带 `--demo-unpaired`，不代表有意义的真实融合情绪，也不用于报告准确率。
示例图片来源：https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/example_outputs/selfie.jpg 。保存在本地被git忽略的 outputs/ 内，仅用于模型连通检查。

```powershell
..\venv\Scripts\python.exe -m trimodal.infer --image trimodal/outputs/upstream_selfie.jpg --face-index 0 --wav data/s70/bF0C11031.wav --demo-unpaired
```

正式同人三模态识别仍需用户采集的配套图片。没有启动任何新训练或科研对照实验。
