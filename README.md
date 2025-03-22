# 🪿 Goose

**Modular and Extensible Data Preprocessing Library for Machine Learning**

`Goose` is a plug-and-play, mixin-based Python library that streamlines the preprocessing of tabular datasets for machine learning tasks. Whether you’re cleaning messy data, encoding categories, transforming skewed distributions, or scaling features — this package has you covered.

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

## 📂 Installation

You can install the package directly from **Test PyPI**:

```bash
pip install --index-url https://test.pypi.org/simple/ Goose
```

Or, after building your wheel file (`.whl`) from the source:

```bash
pip install dist/Goose-0.1.3-py3-none-any.whl
```

Or install directly in **editable mode** (for development):

```bash
pip install -e .
```


---

## 🧪 Usage

```python
import Goose as Goose

# Instantiate with a dataset
obj = Goose(
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
obj = Goose()  # Uses sample heart dataset
```

---

## 📁 Project Structure

```
src/
│
├── Goose/
│   ├── Goose.py                  # Main class
│   ├── mixins/                 # Modular preprocessing logic
│   ├── data/heart.csv          # Default dataset
│   ├── datasets.py             # Heart dataset loader
│   └── __init__.py
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
