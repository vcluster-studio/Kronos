"""
重新整理：保留季度文件和合并年度文件
"""

import re
import os
from collections import defaultdict

output_dir = 'C:/workbench/Kronos/outputs/prediction_results'

# 读取2025年度完整数据
all_records = {}

# 从predictions_2025_full.sql读取
fpath = os.path.join(output_dir, 'predictions_2025_full.sql')
print(f"读取: predictions_2025_full.sql")
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith('insert'):
            continue

        m = re.search(r"values \('([^']+)', '([^']+)', '(\d{4}-\d{2}-\d{2})'", line)
        if m:
            stock_code = m.group(1)
            pred_date = m.group(3)
            key = (stock_code, pred_date)
            all_records[key] = line

print(f"2025年记录数: {len(all_records)}")

# 按季度分组
quarters = defaultdict(list)

for (stock_code, pred_date), sql in all_records.items():
    year = int(pred_date[:4])
    month = int(pred_date[5:7])
    quarter = (month - 1) // 3 + 1
    quarter_key = f"{year}_Q{quarter}"
    quarters[quarter_key].append(sql)

# 写入季度文件
for quarter_key in ['2025_Q1', '2025_Q2', '2025_Q3', '2025_Q4']:
    if quarter_key in quarters:
        sql_lines = quarters[quarter_key]
        output_file = os.path.join(output_dir, f'predictions_{quarter_key}.sql')
        with open(output_file, 'w', encoding='utf-8') as f:
            for sql in sql_lines:
                f.write(sql + '\n')
        print(f"写入: {quarter_key} -> {len(sql_lines)} 条记录")

# 写入Q1Q2合并文件
q1q2_records = quarters['2025_Q1'] + quarters['2025_Q2']
output_file = os.path.join(output_dir, 'predictions_2025_Q1Q2.sql')
with open(output_file, 'w', encoding='utf-8') as f:
    for sql in q1q2_records:
        f.write(sql + '\n')
print(f"写入: 2025_Q1Q2 -> {len(q1q2_records)} 条记录")

print("\n整理完成!")