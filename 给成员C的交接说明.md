# 给成员 C 的交接说明

模块 B 已完成并通过 10/10 项自动化测试。正式结果如下：

- 清洗后训练集/测试集缺失值均为 0。
- 训练集和测试集字段、顺序与行数均已校验。
- 编码后共有 67 个候选特征，折内选择得到 20 个。
- 无数据泄漏嵌套 CV Accuracy 为 `0.8350 ± 0.0226`。
- 独立从零重复运行后，17/17 个确定性产物哈希完全一致。

你建模时有两种用法：

1. **正式模型比较和报告分数（推荐）**

   使用 `source/train.csv`，并从 `titanic_b_pipeline.py` 导入
   `make_model_pipeline`。填补、编码、标准化和特征选择都会在每一折训练数据
   上单独拟合，这个分数可以写进最终报告。

2. **快速试跑或最终训练**

   可以直接使用 `data/Titanic_train_model_ready.csv` 和
   `data/Titanic_test_model_ready.csv`。这两个文件已全部数值化并对齐 20 个
   特征，但不要再用它们报告正式交叉验证分数。

可读、方便人工核对的清洗数据是：

- `data/Titanic_train_clean.csv`
- `data/Titanic_test_clean.csv`

完整方法、字段字典、缺失值审计、CV 结果和测试结论请看：

- `Titanic_B_完整交付报告.xlsx`
- `Titanic_B_数据清洗与特征工程.html`
- `reports/quality_report.json`
- `reports/test_results.json`

最简正式建模代码：

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from titanic_b_pipeline import RAW_FEATURE_COLUMNS, make_model_pipeline

train = pd.read_csv("source/train.csv")
X = train[RAW_FEATURE_COLUMNS]
y = train["Survived"]

pipeline = make_model_pipeline(k=20)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(scores.mean(), scores.std())
```
