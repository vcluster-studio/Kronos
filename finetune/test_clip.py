"""测试剪裁逻辑"""

import pandas as pd
import numpy as np
from batch_inference_sql import clip_prediction, get_price_limit

# 测试数据
last_close = 10.0
price_limit = 0.10  # 主板10%

# 模拟预测结果（超出涨跌幅限制）
pred_df = pd.DataFrame({
    'open': [11.5, 9.5, 10.2],   # 超涨/正常/正常
    'high': [12.0, 10.5, 10.8],  # 超涨
    'low': [8.5, 8.8, 9.5],      # 超跌
    'close': [11.0, 9.0, 10.5],
    'vol': [100, 100, 100],
    'amt': [1000, 1000, 1000]
}, index=pd.date_range('2026-05-19', periods=3))

print("原始预测:")
print(pred_df[['open', 'high', 'low', 'close']])

print(f"\n基准收盘价: {last_close}")
print(f"涨跌幅限制: {price_limit*100}%")
print(f"上限: {last_close * (1 + price_limit)}")
print(f"下限: {last_close * (1 - price_limit)}")

# 剪裁
clipped = clip_prediction(pred_df, last_close, price_limit)
print("\n剪裁后:")
print(clipped[['open', 'high', 'low', 'close']])

# 测试涨跌幅限制判断
print("\n涨跌幅限制:")
test_codes = ['600000.SH', '000001.SZ', '300750.SZ', '688001.SH']
for code in test_codes:
    limit = get_price_limit(code)
    print(f"  {code}: ±{limit*100}%")