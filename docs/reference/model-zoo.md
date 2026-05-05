# 模型仓库

## 可用模型

所有模型均可在 Hugging Face Hub 上获取。

### Kronos-mini

| 属性 | 值 |
|------|-----|
| 参数量 | 4.1M |
| 上下文长度 | 2048 |
| 分词器 | Kronos-Tokenizer-2k |
| 状态 | ✅ 开源 |

**适用场景**: 资源受限环境、快速原型验证

```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
```

---

### Kronos-small

| 属性 | 值 |
|------|-----|
| 参数量 | 24.7M |
| 上下文长度 | 512 |
| 分词器 | Kronos-Tokenizer-base |
| 状态 | ✅ 开源 |

**适用场景**: 平衡性能与资源消耗

```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

---

### Kronos-base

| 属性 | 值 |
|------|-----|
| 参数量 | 102.3M |
| 上下文长度 | 512 |
| 分词器 | Kronos-Tokenizer-base |
| 状态 | ✅ 开源 |

**适用场景**: 高精度预测任务

```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
```

---

### Kronos-large

| 属性 | 值 |
|------|-----|
| 参数量 | 499.2M |
| 上下文长度 | 512 |
| 分词器 | Kronos-Tokenizer-base |
| 状态 | ❌ 待发布 |

---

## 模型选择指南

| 场景 | 推荐模型 |
|------|----------|
| 快速测试/原型 | Kronos-mini |
| 生产环境（有限资源） | Kronos-small |
| 生产环境（追求精度） | Kronos-base |
| 研究用途 | Kronos-base |

## 上下文长度说明

- `max_context` 决定了模型能处理的最大序列长度
- 建议输入数据长度不超过 `max_context`
- 更长的上下文会消耗更多显存

## 分词器匹配

| 模型 | 所需分词器 |
|------|-----------|
| Kronos-mini | Kronos-Tokenizer-2k |
| Kronos-small | Kronos-Tokenizer-base |
| Kronos-base | Kronos-Tokenizer-base |
| Kronos-large | Kronos-Tokenizer-base |

**注意**: 使用不匹配的分词器会导致错误。

---

## 本地模型加载

模型下载后会缓存在本地。也可以手动指定本地路径：

```python
# 从本地路径加载
tokenizer = KronosTokenizer.from_pretrained("/path/to/local/tokenizer")
model = Kronos.from_pretrained("/path/to/local/model")
```

## Hugging Face 镜像

如果下载缓慢，可配置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或使用代理下载后本地加载。