# 官方开源语音模型替换（2026-10-30）

网页语音默认后端已切换至 **emotion2vec+ base**。图像、EGG模型不变，没有重新训练。

## 选择与来源

- 官方代码：https://github.com/ddlBoJack/emotion2vec 。研究代码为MIT；不能据此把所有模型权重也称为MIT。
- 官方权重：https://huggingface.co/emotion2vec/emotion2vec_plus_base 。模型卡声明自定义model-license并指向FunASR；本地保留FunASR MODEL_LICENSE原文及哈希，必须保留来源、作者及模型名称。
- 运行实现：FunASR 1.4.17，使用官方 `AutoModel`，本地路径加载，无运行时音频上传。
- 权重固定revision：`b318240bfe67db81a8c572ecb37ce9c3759b81c9`。
- 权重SHA-256：`60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd`，与官方HuggingFace LFS元数据一致。
- 文件大小1,118,245,678字节（约1.12GB）；不能按推理参数量直接估算检查点文件大小。
- 实际实例参数93,797,909。官方描述base约90M、微调数据4788小时，其中含筛选的伪标签数据，不等于全部为人工标注。
- 同系列large约300M、官方记录42526小时；本轮未下载/测试，优先用base降低实时计算与存储成本。seed版训练数据更少，本轮也未测试。

## 实际数据链

浏览器麦克风原采样率音频 → 直接重采样16kHz → 官方emotion2vec waveform LayerNorm/编码器/分类头 → 官方9类概率 → 标签映射 → 网页三分类。

重要修正：不再先把所有信号统一降到8kHz。服务器按模态分别处理：语音16kHz，EGG仍为8kHz。原始采集音频保存逻辑不变。命令行融合入口也使用同一语音配置。

9类：angry、disgusted、fearful、happy、neutral、other、sad、surprised、unknown。
negative = angry + disgusted + fearful + sad；neutral = neutral；positive = happy。
other/surprised/unknown不强行映射；保留原始9类概率和排除概率，将剩余质量重新归一化。若剩余概率小于1e-6，报告不能映射而非生成伪造概率。未映射质量较高时，三分类是条件概率，不能理解为模型确信三类之一；页面显示原始最高类别及未映射概率。
FunASR输出已是softmax概率，没有二次softmax。

## 隔离运行

FunASR安装到 `trimodal/.venv_ser`，模型在持久子进程中加载一次，通过本机管道接收16kHz数组，结果来自真实模型。
此环境复用项目已验证的PyTorch 2.6.0+cu124及torchaudio，未升级或替换网页环境里的这些包。当前自动选择CUDA，实际设备为RTX 4050；没有CUDA则用CPU，但本轮没有测CPU速度。
首次服务启动包含导入与加载，可能需要约一分钟。每次录音推理不重复加载权重。
worker异常/超时会明确报错，不静默退回旧模型。停止服务器时关闭worker。

复建：在PowerShell运行 `trimodal/setup_speech.ps1`（约1.12GB权重加依赖下载）。
常规启动仍为：

```powershell
Set-Location 'D:\claudeprojetcs\voiceintelligence\mindguardvanguard'
..\venv\Scripts\python.exe -m trimodal.server
```

`trimodal/config.json` 的 `speech_backend` 为 `emotion2vec_plus_base`。
旧自训checkpoint保持原样；若主动选择回退，将该值设为 `legacy` 并重启，页面会如实显示旧模型名称。没有自动回退。

## 实际检查

- 官方权重加载日志：All keys matched successfully。
- 一条S70真实语音（bF0C11031.wav，约1.24秒）连续三次推理：1506ms、61ms、44ms；首轮含GPU首次调用开销。这是连通与耗时检查，不是准确率对照。
- 结果见 `outputs/open_speech_smoke.json`。该例原始模型最高类为neutral，而文件名标签为sad；这一单例不能说明整体优劣，不能宣称换模型必然更准。
- `python -m unittest trimodal.test_fusion trimodal.test_open_speech trimodal.test_server -v`：22项通过，46.076秒（含模型启动），覆盖真实模型、类别映射、缺EGG不输出融合、记录保存与导出。
- 重启后的 http://127.0.0.1:8765 已通过浏览器回归：页面显示emotion2vec+，真实文件三路推理成功，语音确认为16kHz并保留9类概率，浏览器虚拟麦克风传输调用真实模型，停止正常，无JavaScript错误。实际硬件效果仍待现场使用。检查记录为 `outputs/dashboard_browser_checks.json`。
- 无新增训练、调参或模型效果对照实验。现场中文短句、噪声、麦克风和2秒窗口上的准确率未验证。

使用同一浏览器页面，刷新后语音卡片应显示 `emotion2vec+ base · 官方预训练`。所有保存的窗口携带实际模型路径/哈希、9类概率与映射详情，人工标签仍单独记录。
