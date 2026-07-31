# Titanic 竞赛模块 B：最终完整交付

本目录完成了模块 B 要求的缺失值处理、特征工程、分类变量编码、连续变量
标准化、特征选择、无数据泄漏交叉验证和质量测试。

## 已完成内容

- `Age`：按 `Title + Pclass + Sex` 的训练折中位数填补，并逐级回退。
- `Fare`：按训练折 `Pclass` 中位数填补。
- `Embarked`：按训练折众数填补。
- `Cabin`：提取 `HasCabin`、`CabinMissing` 和 `CabinDeck`。
- 新特征：`Title`、`FamilySize`、`FamilyGroup`、`IsAlone`、
  `TicketPrefix`、`FarePerPerson`、`SexPclass` 等。
- 编码：分类字段使用 `OneHotEncoder(handle_unknown="ignore")`。
- 标准化：`AgeFilled`、`FareFilled` 使用 `StandardScaler`。
- 特征选择：用互信息 `SelectKBest`，`k` 在交叉验证训练折内选择。
- 正式评估：外层 5 折、内层 4 折的嵌套分层交叉验证。

## 验收结果

- 自动化测试：10/10 通过。
- 独立重复运行：17/17 个确定性产物的 SHA-256 完全一致。
- 清洗后缺失值：训练集 0，测试集 0。
- 原始数据：训练集 891×12，测试集 418×11。
- 清洗数据：训练集 891×28，测试集 418×27。
- 编码后候选特征：67。
- 最终选择特征：20。
- 无泄漏嵌套 CV Accuracy：`0.8350 ± 0.0226`。
- OOF Accuracy：`0.8350`。

## 目录说明

```text
source/
  train.csv                         原始 Kaggle 训练数据
  test.csv                          原始 Kaggle 测试数据
data/
  Titanic_train_clean.csv           可读清洗版，含 Survived
  Titanic_test_clean.csv            可读清洗版
  Titanic_train_model_ready.csv     全训练集拟合后的数值便利版
  Titanic_test_model_ready.csv      全训练集拟合后的数值便利版
reports/
  quality_report.json               总体验收结论
  test_results.json                 自动化测试结果
  reproducibility_check.json        独立重复运行哈希比对
  nested_cv_results.csv             外层 5 折结果
  feature_selection_results.csv     k 的选择结果
  feature_scores.csv                全部候选特征互信息分数
  selected_features.txt             最终入选特征
  missing_values_before_after.csv   清洗前后缺失值审计
  field_dictionary.csv              字段字典
  cleaning_rules.csv                清洗规则
  preprocessing_parameters.json     填补参数、文件哈希和随机种子
models/
  titanic_b_transform_selector.joblib
  titanic_b_logistic_baseline.joblib
titanic_b_pipeline.py               正式无泄漏流水线
test_titanic_b_pipeline.py          10 项单元/集成测试
run_tests.py                        测试入口
verify_reproducibility.py           重复运行产物比对工具
Titanic_B_数据清洗与特征工程.ipynb   已执行 Notebook
Titanic_B_数据清洗与特征工程.html    可直接浏览的 Notebook 报告
Titanic_B_完整交付报告.xlsx          汇总工作簿
给成员C的交接说明.md                 直接转发给成员 C 的说明
```

## 正式使用方法

创建环境并复现全部输出：

```bash
python -m pip install -r requirements.txt
python titanic_b_pipeline.py \
  --train source/train.csv \
  --test source/test.csv \
  --output .
python run_tests.py
```

成员 C 做正式模型比较时，应把**原始数据**直接传给完整 Pipeline：

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from titanic_b_pipeline import RAW_FEATURE_COLUMNS, make_model_pipeline

train = pd.read_csv("source/train.csv")
X = train[RAW_FEATURE_COLUMNS]
y = train["Survived"]

pipeline = make_model_pipeline(
    RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    ),
    k=20,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(scores.mean(), scores.std())
```

## 重要方法说明

`Titanic_*_model_ready.csv` 是用**全部训练集**学习填补、编码、缩放和特征
选择参数后导出的便利版，只适合快速基线、最终模型拟合或提交预测。正式交叉
验证必须使用 `source/train.csv` 和 `make_model_pipeline(...)`，否则验证折会
间接参与预处理，造成数据泄漏。

`PassengerId` 只用于追踪和提交，不进入模型特征。测试集没有 `Survived`，
这是 Kaggle 标准格式，不是缺失。
