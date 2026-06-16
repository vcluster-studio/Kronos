# 数据目录

按归一化方式组织数据。

## 结构

```
data/
├── source/              # 原始数据源
├── global_norm/         # 全窗口归一化
├── ma60_norm/           # MA60滑动归一化
└── README.md
```

## 归一化方式

### global_norm — 全窗口归一化

对每个样本的整个窗口计算 mean/std，然后归一化。适用于 mode1_original。

### ma60_norm — MA60滑动归一化

对每个时间点使用前60步的局部 mean/std 进行归一化。适用于 mode2/3/4。

## 生成数据

每个数据子目录包含 `generate.py` 脚本，用于生成对应格式的数据：

```bash
cd finetune/data/{norm_type}/{layout}
python generate.py
```

## 数据格式

特征顺序：`[open, high, low, close, vol, amt]`