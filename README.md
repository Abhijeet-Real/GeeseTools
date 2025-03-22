# 🪿🪿 GeeseTools 🛠️🛠️

**Modular and Extensible Data Preprocessing Library for Machine Learning**

`GeeseTools` is a plug-and-play, mixin-based Python library that streamlines the preprocessing of tabular datasets for machine learning tasks. Whether you’re cleaning messy data, encoding categories, transforming skewed distributions, or scaling features — this package has you covered.

---

## 🚀 Features

- 🧼 Handle missing data
- 🔢 Convert object columns to numeric
- 🔍 Identify feature types (categorical, ordinal, nominal, etc.)
- ⚙️ Encode nominal and ordinal features
- 🔄 Transform skewed and heavy-tailed features
- 📏 Scale features with standard or power transformations
- 🧪 Train-test split with optional oversampling
- 📊 Transformation logs for transparency and reproducibility
- 🔌 Built using Mixins for modular extension

---

## ⚙️ Installation

You can install the package directly from **PyPI**:

```bash
pip install GeeseTools
```

Or, after building your wheel file (`.whl`) from the source:

```bash
pip install dist/GeeseTools-0.1.8-py3-none-any.whl
```

Or install directly in **editable mode** (for development):

```bash
pip install -e .
```


---

## 🧪 Usage

```python
import GeeseTools as gt

# Instantiate with a dataset
obj = gt(
    dataframe=df,
    target_variable='target',
    ordinal_features=['education_level'],
    ordinal_categories=[['Low', 'Medium', 'High']],
    use_one_hot_encoding=True
)

# Apply full preprocessing pipeline
X_train, X_test, y_train, y_test = obj.pre_process()

# Access logs
print(obj.transformation_log_df)
```

---

## 🗂 Default Sample Dataset

If no DataFrame is provided, the processor loads a built-in `heart.csv` dataset:

```python
obj = GeeseTools()  # Uses sample heart dataset

# Apply full preprocessing pipeline
X_train, X_test, y_train, y_test = obj.pre_process()
```

---

## 📁 Project Structure

```
📦 GeeseTools/
├── 📂 data/                            # 📁 Contains bundled datasets
│   ├── 📄 heart.csv                    # 📊 Sample dataset (CSV format)
│   └── 📜 __init__.py                  # 📦 Makes 'data' a subpackage
│
├── 📜 GeeseTools.py                    # 🧠 Core toolkit initializer or controller
├── 📜 datasets.py                      # 📂 Dataset loading utilities
├── 🧩 display_mixin.py                 # 🖥️ Display-related mixin
├── 🧩 drop_features_mixin.py           # ✂️ Drop unwanted features
├── 🧩 drop_records_mixin.py            # 🗑️ Drop records based on rules
├── 🧩 encode_mixin.py                  # 🔤 Encoding (label, one-hot)
├── 🧩 feature_target_split_mixin.py    # 🔀 Split into features & target
├── 🧩 feature_type_mixin.py            # 🧬 Feature type detection
├── 🧩 impute_features_mixin.py         # 🩹 Fill missing values
├── 🧩 missing_data_summary_mixin.py    # 📉 Summary of missing data
├── 🧩 oversample_mixin.py              # 🧪 Oversampling (e.g., SMOTE)
├── 🧩 pre_process_mixin.py             # ⚙️ Complete preprocessing pipeline
├── 🧩 sample_data_mixin.py             # 🎲 Random sampling utilities
├── 🧩 scale_mixin.py                   # 📏 Scaling methods
├── 🧩 split_dataframe_mixin.py         # 🧯 Split dataframe columns
├── 🧩 to_numeric_mixin.py              # 🔢 Convert to numeric
├── 🧩 transform_mixin.py               # 🔧 Feature transformations
├── 🧩 unique_value_summary_mixin.py    # 🧾 Unique value summary
└── 📜 __init__.py                      # 📦 Initializes GeeseTools package
```

---

## ⚙️ Requirements

- Python 3.9–3.11
- pandas
- scikit-learn
- imbalanced-learn
- scipy
- ipython
- openpyxl

---

## 📜 License

MIT © Abhijeet  
_You're free to use, modify, and distribute this project with proper attribution._

---

## ✨ Contributions Welcome

Want to add new mixins or support more file types? Fork it, branch it, push it, and let’s build together!
