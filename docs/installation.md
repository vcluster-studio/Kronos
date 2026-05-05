# 安装指南

## 系统要求

- Python 3.10+
- CUDA 支持的 GPU（推荐）

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 核心依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| numpy | - | 数值计算 |
| pandas | 2.2.2 | 数据处理 |
| torch | >=2.0.0 | 深度学习框架 |
| einops | 0.8.1 | 张量操作 |
| huggingface_hub | 0.33.1 | 模型下载 |
| matplotlib | 3.9.3 | 可视化 |
| tqdm | 4.67.1 | 进度条 |
| safetensors | 0.6.2 | 模型序列化 |

## 可选依赖

### Qlib 微调支持

如需使用 Qlib 进行 A 股数据微调：

```bash
pip install pyqlib
```

### Web UI 支持

```bash
cd webui
pip install -r requirements.txt
```

Web UI 依赖：
- flask 2.3.3
- flask-cors 4.0.0
- plotly 5.17.0

## 验证安装

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 测试模型加载
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
print("安装成功!")
```

## 常见问题

### CUDA 内存不足

使用较小的模型（如 Kronos-mini）或减少 `max_context` 参数。

### Hugging Face 下载慢

配置镜像源或使用代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```