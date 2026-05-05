# API 参考

## KronosTokenizer

K线数据分词器，将连续OHLCV数据量化为离散token。

### 初始化参数

```python
KronosTokenizer(
    d_in,           # 输入维度
    d_model,        # 模型维度
    n_heads,        # 注意力头数
    ff_dim,         # 前馈网络维度
    n_enc_layers,   # 编码器层数
    n_dec_layers,   # 解码器层数
    ffn_dropout_p,  # FFN dropout概率
    attn_dropout_p, # 注意力dropout概率
    resid_dropout_p,# 残差dropout概率
    s1_bits,        # pre token位数
    s2_bits,        # post token位数
    beta,           # BSQuantizer参数
    gamma0,         # BSQuantizer参数
    gamma,          # BSQuantizer参数
    zeta,           # BSQuantizer参数
    group_size      # 分组大小
)
```

### 主要方法

#### from_pretrained()

从 Hugging Face Hub 加载预训练分词器。

```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
```

#### encode()

将输入数据编码为量化索引。

```python
z_indices = tokenizer.encode(x, half=False)
# x: (batch_size, seq_len, d_in)
# 返回: 量化索引
```

#### decode()

将量化索引解码为原始数据空间。

```python
output = tokenizer.decode(x, half=False)
# x: 量化索引
# 返回: (batch_size, seq_len, d_in)
```

---

## Kronos

主模型类，decoder-only Transformer架构。

### 初始化参数

```python
Kronos(
    s1_bits,          # pre token位数
    s2_bits,          # post token位数
    n_layers,         # Transformer层数
    d_model,          # 模型维度
    n_heads,          # 注意力头数
    ff_dim,           # 前馈网络维度
    ffn_dropout_p,    # FFN dropout
    attn_dropout_p,   # 注意力dropout
    resid_dropout_p,  # 残差dropout
    token_dropout_p,   # token dropout
    learn_te          # 是否使用可学习时间嵌入
)
```

### 主要方法

#### from_pretrained()

从 Hugging Face Hub 加载预训练模型。

```python
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

#### forward()

前向传播。

```python
s1_logits, s2_logits = model(
    s1_ids,           # s1 token IDs: [batch_size, seq_len]
    s2_ids,           # s2 token IDs: [batch_size, seq_len]
    stamp=None,       # 时间戳
    padding_mask=None, # 填充掩码
    use_teacher_forcing=False,
    s1_targets=None
)
```

#### decode_s1() / decode_s2()

分步解码方法。

---

## KronosPredictor

高级预测接口，封装完整的预测流程。

### 初始化

```python
predictor = KronosPredictor(
    model,           # Kronos模型实例
    tokenizer,       # KronosTokenizer实例
    device=None,     # 设备（自动检测）
    max_context=512, # 最大上下文长度
    clip=5           # 数据裁剪阈值
)
```

### predict()

单序列预测。

```python
pred_df = predictor.predict(
    df,              # DataFrame: 历史数据
    x_timestamp,     # 历史时间戳
    y_timestamp,     # 预测时间戳
    pred_len,        # 预测长度
    T=1.0,           # 温度
    top_k=0,         # Top-k
    top_p=0.9,       # Top-p
    sample_count=1,  # 采样次数
    verbose=True     # 显示进度
)
```

**返回值**: DataFrame，包含预测的 `open, high, low, close, volume, amount` 列。

### predict_batch()

批量预测。

```python
pred_df_list = predictor.predict_batch(
    df_list,           # DataFrame列表
    x_timestamp_list,  # 历史时间戳列表
    y_timestamp_list,  # 预测时间戳列表
    pred_len,
    T=1.0,
    top_k=0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

**返回值**: DataFrame列表，每个包含预测结果。

---

## 工具函数

### calc_time_stamps()

计算时间特征。

```python
from model.kronos import calc_time_stamps

time_df = calc_time_stamps(timestamps)
# 返回: DataFrame包含 ['minute', 'hour', 'weekday', 'day', 'month']
```

### sample_from_logits()

从logits采样。

```python
from model.kronos import sample_from_logits

sample = sample_from_logits(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None,
    sample_logits=True
)
```