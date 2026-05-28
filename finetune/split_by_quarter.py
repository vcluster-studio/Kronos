"""
按季度拆分预测结果SQL文件
"""

import os
import re
from collections import defaultdict

input_file = 'outputs/prediction_results/predictions_mini.sql'
output_dir = 'outputs/prediction_results'

# 读取文件
print(f"读取文件: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"总行数: {len(lines)}")

# 按季度分组
quarter_files = defaultdict(list)

for line in lines:
    # 提取 pred_date，格式: '2026-05-18'
    match = re.search(r", '(\d{4}-\d{2}-\d{2})',", line)
    if match:
        pred_date = match.group(1)
        year = int(pred_date[:4])
        month = int(pred_date[5:7])

        # 计算季度
        quarter = (month - 1) // 3 + 1
        quarter_key = f"{year}_Q{quarter}"
        quarter_files[quarter_key].append(line)

# 统计
print("\n各季度统计:")
for q in sorted(quarter_files.keys()):
    print(f"  {q}: {len(quarter_files[q])} 条")

# 保存各季度文件
for quarter, lines in quarter_files.items():
    output_file = os.path.join(output_dir, f'predictions_{quarter}.sql')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"保存: {output_file} ({size_mb:.1f} MB)")

print("\n完成!")