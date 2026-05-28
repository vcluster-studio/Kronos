"""
合并所有预测结果，按季度分割
包含：第一批、第二批、第三批
"""

import re
import os
from collections import defaultdict

output_dir = 'C:/workbench/Kronos/outputs/prediction_results'

files = [
    'predictions_merged_2025_Q3.sql',
    'predictions_merged_2025_Q4.sql',
    'predictions_merged_2026_Q1.sql',
    'predictions_merged_2026_Q2.sql',
    'predictions_mini_part3.sql'
]

# 读取所有数据，按 (stock_code, pred_date) 去重
all_records = {}

for fname in files:
    fpath = os.path.join(output_dir, fname)
    print(f"读取: {fname}")
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

print(f"\n去重后总记录数: {len(all_records)}")

# 按季度分组
quarters = defaultdict(list)

for (stock_code, pred_date), sql in all_records.items():
    year = int(pred_date[:4])
    month = int(pred_date[5:7])
    quarter = (month - 1) // 3 + 1
    quarter_key = f"{year}_Q{quarter}"
    quarters[quarter_key].append(sql)

# 写入季度文件
for quarter_key, sql_lines in sorted(quarters.items()):
    output_file = os.path.join(output_dir, f'predictions_{quarter_key}.sql')
    with open(output_file, 'w', encoding='utf-8') as f:
        for sql in sql_lines:
            f.write(sql + '\n')
    print(f"写入: {quarter_key} -> {len(sql_lines)} 条记录")

print("\n合并完成!")

# 统计各季度
print("\n季度统计:")
total = 0
for q in sorted(quarters.keys()):
    count = len(quarters[q])
    total += count
    print(f"  {q}: {count} 条")
print(f"  总计: {total} 条")