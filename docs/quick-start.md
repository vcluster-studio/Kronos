# 快速开始

## 基本预测流程

### 1. 加载模型和分词器

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 从 Hugging Face Hub 加载
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

### 2. 创建预测器

```python
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

### 3. 准备数据

数据要求：
- DataFrame 必须包含 `['open', 'high', 'low', 'close']` 列
- `volume` 和 `amount` 可选
- 需要提供时间戳序列

```python
import pandas as pd

# 加载数据
df = pd.read_csv("./data/your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 定义回看窗口和预测长度
lookback = 400
pred_len = 120

# 准备输入
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

### 4. 生成预测

```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,           # 温度参数
    top_p=0.9,       # 核采样概率
    sample_count=1   # 预测路径数量
)

print(pred_df.head())
```

## 批量预测

处理多个时间序列：

```python
# 准备多个数据集
df_list = [df1, df2, df3]
x_timestamp_list = [x_ts1, x_ts2, x_ts3]
y_timestamp_list = [y_ts1, y_ts2, y_ts3]

# 批量预测
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

**批量预测要求**：
- 所有序列的历史长度必须相同
- 所有序列的预测长度（pred_len）必须相同

## 参数说明

### predict() 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| df | DataFrame | 必需 | 历史K线数据 |
| x_timestamp | Series | 必需 | 历史时间戳 |
| y_timestamp | Series | 必需 | 预测时间戳 |
| pred_len | int | 必需 | 预测长度 |
| T | float | 1.0 | 采样温度 |
| top_k | int | 0 | Top-k 过滤 |
| top_p | float | 0.9 | 核采样阈值 |
| sample_count | int | 1 | 采样次数 |
| verbose | bool | True | 显示进度 |

## 示例脚本

运行完整示例：

```bash
python examples/prediction_example.py
```

无成交量预测示例：

```bash
python examples/prediction_wo_vol_example.py
```

## Web UI

启动 Web 界面：

```bash
cd webui
python run.py
# 访问 http://localhost:5000
```