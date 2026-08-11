# Fara Tutorial Guide RAG (demo1.0)

基于 **Fara1.5-4B** 视觉模型的网页操作指南系统：用户输入意图 + 截图 → RAG 检索说明书 → Fara 输出点击坐标 → Checker 验证闭环（最多 3 轮反馈迭代）。

> 适用环境：NVIDIA GPU（单卡/多卡自动适配）、Python 3.12、可访问魔搭 ModelScope 下载模型。

## 架构与端口

| 模块 | 模型 | 承载 | 端口 |
|------|------|------|------|
| Fara（坐标推理） | `Fara1.5-4B`（视觉） | vLLM（OpenAI 兼容） | **5002** |
| Checker（坐标验证） | `Qwen3.5-0.8B`（视觉） | vLLM（OpenAI 兼容） | **5003** |
| RAG 检索 | `BAAI/bge-small-zh-v1.5` | 内嵌于 demo/web_demo | — |
| Web Demo | — | Flask | **8090** |
| 前端静态页（截图源） | — | http.server | **8080** |

数据流：用户意图 → RAG 检索说明书切片 → Fara 结合截图+切片出坐标 → Checker 验证（错则反馈迭代，最多 3 轮）。

## 目录结构

```
├── start_all.py / stop_all.py   # 一键启动 / 停止入口
├── loop.py                      # 编排核心（RAG→Fara→Checker 闭环）
├── scripts/                     # 全部脚本
│   ├── config.py                # ★ 全局集中配置（唯一配置源）
│   ├── launcher.py / stopper.py # 启动/停止逻辑
│   ├── model_server.py          # vLLM 模型服务管理
│   ├── demo.py / web_demo.py / static_server.py
├── fara/  checker/  rag/        # 三大功能模块
├── html/                        # 前端页面（挂号/缴费/取药）
├── docs/manual + docs/img       # 说明书 PDF + 测试截图
└── requirements.txt / environment.yml
```

## 配置

所有配置集中在 **`scripts/config.py`**（环境变量可覆盖）。常用项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODELS_CACHE` | `~/workspace/models` | 模型缓存根目录 |
| `FARA_MODEL_PATH` / `CHECKER_MODEL_PATH` | 模型路径（含尾 `/`） | 也是 vLLM served-model-name |
| `FARA_PORT` / `CHECKER_PORT` | 5002 / 5003 | 模型服务端口 |
| `SERVE_PORT` / `WEB_DEMO_PORT` | 8080 / 8090 | Web 端口 |
| `FARA_GPU` / `CHECKER_GPU` | **自动分配** | ≥2 卡→0/1；1 卡→都放 0 |
| `DEFAULT_INTENT` | 默认意图 | Web 界面初始意图 |
| `PYTHON_BIN` | 自动查找 venv | Web 子进程解释器 |

环境变量示例：`FARA_MODEL_PATH=... CHECKER_MODEL_PATH=... python start_all.py`

## 环境准备

```bash
# 方式一：conda（推荐）
conda env create -f environment.yml
conda activate fara15

# 方式二：pip
pip install -r requirements.txt

# 下载模型（魔搭，路径需与 config 一致）
modelscope download --model microsoft/Fara1.5-4B --local_dir ~/workspace/models/microsoft--Fara1.5-4B
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ~/workspace/models/Qwen--Qwen3.5-0.8B

# 构建 RAG 索引（离线一次性）
python -m rag build --pdf docs/manual/manual.pdf --models-cache ~/workspace/models
```

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

## 版本要求（重要）

- Fara1.5-4B 官方要求：`torch>=2.11` / `transformers>=5.2` / `vllm>=0.19.1`
- vllm 0.20+/torch 2.11+ 为 CUDA 13 构建，需驱动 >=580；驱动仅支持 CUDA 12.x 时锁定 `vllm==0.19.1 torch==2.10.0 transformers==5.14.1`（见 `requirements.txt` 注释）

## License

MIT
