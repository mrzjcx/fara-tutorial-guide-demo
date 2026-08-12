# Fara Tutorial Guide RAG (demo1.0.2)

基于 **Fara1.5-4B** 视觉模型的网页（或某些终端）操作指南系统：
用户输入意图 + 实时截图 → 通过 RAG 检索说明书 → Fara 输出点击坐标 → Checker 验证闭环。
目前任务：模型仅输出点击位置提示作为辅助而不替用户做决定。

说明：目前问题在于：
1. 模型如果太弱/没有适配特定的程序，针对各种复杂网页UI可能会出现误判，如果checker检查器本身识图能力不够，也会导致误判； 
2. 模型参数量、推理准确性和推理速度之间存在矛盾，需要取舍； 


> 适用环境：NVIDIA GPU（单卡/多卡自动适配）、Python 3.12。

## 默认架构与端口
## 模型可以替换
| 模块 | 模型 | 承载 | 端口 |
|------|------|------|------|
| Fara（坐标推理） | [`Fara1.5-4B`](https://modelscope.cn/models/microsoft/Fara1.5-4B)（视觉） | vLLM（OpenAI 兼容） | **5002** |
| Checker（坐标验证） | [`Qwen3.5-0.8B`](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B)（视觉） | vLLM（OpenAI 兼容） | **5003** |
| RAG 检索 | [`BAAI/bge-small-zh-v1.5`](https://modelscope.cn/models/BAAI/bge-small-zh-v1.5) | 内嵌于 demo/web_demo | — |
| Web Demo | — | Flask | **8090** |
| 前端静态页（截图源） | — | http.server | **8080** |

数据流：用户意图 → RAG 检索说明书切片 → Fara 结合截图+切片出坐标与引导语 → Checker 参考验证（仅建议，不否决）→ 人在回路自行判断。

## 操作方式（辅助引导）

> 定位：**辅助用户适应新系统**。Fara 只给出建议坐标与中文引导语，**不替用户做决定 / 不自动点击**。

1. **用户操作页面**（左侧 iframe）→ 输入意图 → 点「执行」
2. 前端用 **html-to-image 截图** （或者替换为CDP截图，按需使用）发送给后端
3. **Fara** 结合截图 + RAG 说明书切片 → 输出**引导语**（面向用户的中文操作指引）+ **点击坐标**（1000×1000 归一化）
4. 网页端**红圈标注**坐标位置，用户对照引导语自行核对
5. **Checker** 检查坐标附近**放大图中心**的内容是否符合教程 → 给出参考意见
6. **人机确认点**：涉及提交/确认等不可逆操作时提示用户确认
7. 模型无法确定坐标时，**降级为文字指导**（只给引导语，不给坐标）
8. 涉及到用户信息的填写部分——可能无法起效

## Checker 角色变更

早期 Checker 判错即触发 Fara 重试，多次失败则降级——但小模型（如Qwen3.5-0.8B）视觉/指令能力有限，**幻觉误判会否决正确坐标**。现调整为**建议者**仅输出建议，同时也为后续更新提供一个debug的视角：

## 目录结构

```
├── start_all.py / stop_all.py   # 一键启动 / 停止入口
├── loop.py                      # 编排核心（RAG→Fara→Checker 闭环）
├── scripts/                     # 全部脚本
│   ├── config.py                # 全局集中配置
│   ├── launcher.py / stopper.py # 启动/停止逻辑
│   ├── model_server.py          # vLLM 模型服务管理
│   ├── demo.py / web_demo.py / static_server.py
├── fara/  checker/  rag/        # 三大功能模块
├── html/                        # 前端页面（挂号/缴费/取药）
├── docs/manual + docs/img       # 说明书 PDF + 测试截图
└── requirements.txt / environment.yml
```

## 配置

所有默认配置集中在 **`scripts/config.py`**（环境变量可覆盖）。
常用项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODELS_CACHE` | `~/workspace/models` | 模型缓存根目录 |
| `FARA_MODEL_PATH` / `CHECKER_MODEL_PATH` | 模型路径（含尾 `/`） | 也是 vLLM served-model-name |
| `FARA_PORT` / `CHECKER_PORT` | 5002 / 5003 | 模型服务端口 |
| `SERVE_PORT` / `WEB_DEMO_PORT` | 8080 / 8090 | Web 端口 |
| `FARA_GPU` / `CHECKER_GPU` | **自动分配** | ≥2 卡→0/1；1 卡→都放 0 |
| `DEFAULT_INTENT` | 默认意图 | Web 界面初始意图 |
| `PYTHON_BIN` | 自动查找（venv/conda） | Web 子进程解释器 |

环境变量示例：`FARA_MODEL_PATH=... CHECKER_MODEL_PATH=... python start_all.py`

## 环境准备（Miniforge 部署）

> 依赖已打包在 `environment.yml` + `requirements.txt` 中

```bash
# 0) 安装 Miniforge（已装 conda/miniforge 可跳过）
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b
~/miniforge3/bin/conda init bash && source ~/.bashrc

# 1) 克隆项目
cd ~ && git clone https://github.com/mrzjcx/fara-tutorial-guide-demo.git
cd fara-tutorial-guide-demo

# 2) 一键创建环境（Python 3.12 + 全部依赖，含 vllm 0.19.1 / torch 2.10.0 / transformers 5.14.1）
conda env create -f environment.yml
conda activate fara15

# 3) 下载模型（魔搭，或者huggingface；bge 嵌入模型代码会自动下载；按需下载合适的版本）
modelscope download --model microsoft/Fara1.5-4B --local_dir ~/workspace/models/microsoft--Fara1.5-4B
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ~/workspace/models/Qwen--Qwen3.5-0.8B

# 4) 构建 RAG 索引（离线一次性，说明书已随仓库提供）
python -m rag build --pdf docs/manual/manual.pdf --models-cache ~/workspace/models

# 5) 一键启动（模型 + Web，自动按 GPU 数量分配）
python start_all.py
```

> 备选：不装 Miniforge 也可 `pip install -r requirements.txt`（需自行准备 Python 3.12 与 CUDA 环境）。

## 启动方式

```bash
# 一键启动全部（模型 + Web），等待就绪
python start_all.py

# 只启动模型服务 / 查看状态 / 停止
python scripts/model_server.py --status
python stop_all.py                 # 停止全部（模型 + Web）
python stop_all.py --status        # 查看状态

# 命令行端到端 demo（单次）
python scripts/demo.py --screenshot docs/img/02_register_dept.png --intent "在线挂号选骨科"

# Web 界面
#   打开 http://<IP>:8090  （Web Demo 主界面）
#   打开 http://<IP>:8080  （前端页面）
```

## 版本要求

- Fara1.5-4B 官方要求：`torch>=2.11` / `transformers>=5.2` / `vllm>=0.19.1`
- vllm 0.20+/torch 2.11+ 为 CUDA 13 构建，需驱动 >=580；驱动仅支持 CUDA 12.x 时锁定 `vllm==0.19.1 torch==2.10.0 transformers==5.14.1`（见 `requirements.txt` 注释）

## 模型链接与量化版本

> 以下为相关模型的 ModelScope 链接，以及 Fara1.5-4B 各量化版本（替换模型时留意格式与兼容性）。

### 当前使用的模型

| 模型 | ModelScope 链接 |
|------|----------------|
| Fara1.5-4B（坐标推理） | https://modelscope.cn/models/microsoft/Fara1.5-4B |
| Qwen3.5-0.8B（Checker 验证） | https://modelscope.cn/models/Qwen/Qwen3.5-0.8B |
| BAAI/bge-small-zh-v1.5（RAG 嵌入） | https://modelscope.cn/models/BAAI/bge-small-zh-v1.5 |

### Fara1.5-4B 量化版本

| 版本 | 格式 | 说明 |
|------|------|------|
| [latentnode/Fara1.5-4B-OptiQ-4bit](https://modelscope.cn/models/latentnode/Fara1.5-4B-OptiQ-4bit) | MLX | Apple Silicon 专用（mlx-optiq），不适合本 vLLM 项目 |
| [mlx-community/Fara1.5-4B-OptiQ-4bit](https://modelscope.cn/models/mlx-community/Fara1.5-4B-OptiQ-4bit) | MLX | 同上 |
| [mlx-community/Fara1.5-4B-8bit](https://modelscope.cn/models/mlx-community/Fara1.5-4B-8bit) | MLX | Apple Silicon 专用 |
| [bartowski/Fara1.5-4B-GGUF](https://modelscope.cn/models/bartowski/Fara1.5-4B-GGUF) | GGUF | llama.cpp 路线；缺 mmproj，视觉不可用 |
| [prithivMLmods/Fara1.5-4B-GGUF](https://modelscope.cn/models/prithivMLmods/Fara1.5-4B-GGUF) | GGUF | llama.cpp 路线；同上 |
| [DevQuasar/microsoft.Fara1.5-4B-GGUF](https://modelscope.cn/models/DevQuasar/microsoft.Fara1.5-4B-GGUF) | GGUF | llama.cpp 路线；同上 |
| [prithivMLmods/Fara1.5-4B-FP8](https://modelscope.cn/models/prithivMLmods/Fara1.5-4B-FP8) | Safetensors FP8 | vLLM 可能兼容 |

## License

MIT
