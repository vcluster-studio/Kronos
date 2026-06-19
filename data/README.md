# 原始数据总集

项目原始数据源，从 `kline_daily.sql` 解析生成。

## 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `kline_daily.sql` | 3.5 GB | PostgreSQL 导出的原始 SQL |
| `kline_daily_raw.pkl` | 253 MB | 解析后的 pickle 格式 |

## 生成

```bash
cd data
python generate_raw.py
```

## 数据格式

`kline_daily_raw.pkl` 包含字典：`{symbol: {'values': np.ndarray, 'index': DatetimeIndex}}`

- `values`: (T, 6) 原始 OHLCV 数据 `[open, high, low, close, vol, amt]`
- `index`: DatetimeIndex 时间索引

## 统计

- 有效股票：4861 只（过滤掉少于 250 天的股票）
- 时间范围：2018-01-02 ~ 2026-05-18