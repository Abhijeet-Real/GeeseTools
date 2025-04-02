# 🪿🪿GeeseTools
- Fast & Flexible Data Analysis Toolkit

Welcome to **GeeseTools** – a lightweight and modular toolkit designed for quick data preprocessing, model building, evaluation, and visualizations. Perfect for quick experiments and rapid prototyping in machine learning workflows!

---

## Features

- Clean and preprocess your datasets effortlessly with `datapreprocessor`
- Quickly train and evaluate models with `utils`
- Auto-generate plots for insights and performance metrics
- Minimal setup, beginner-friendly, and fully extensible

---

## Module Structure

```
📦 GeeseTools/
├── 📂 data/                            
│   └──📄 heart.csv             # Default Dataset            
│    
├── 📁 DataPreProcessor       
│   └──📜 DataPreProcessor.py   # Main Script
│
└── 📁 utils                    
    ├──📜 train_models.py       # Model training,
    ├──📜 evaluation.py         # Model evaluation
    └──📜 plot.py               # Evaluation visualization


```

##  Installation


```bash
pip install GeeseTools
```

---

## 📚 How to Use

### 1. Import the modules

```python
from datapreprocessor import preprocess_data
from utils import train_model as tm
from utils import evaluate_model as eval
from utils import plot
```

---

### 2. Preprocess your data

```python
# Creating object for DataPreProcessor Class
obj = dpp(pd.read_csv("heart.csv"), target="diagnosis")
```

---

### 3. Train a model

```python
model, task_type, history = tm.train_model(X_train, y_train)
```

---

### 4. Evaluate the model

```python
metric, y_pred = eval.evaluate_model(model, X_test, y_test, task_type)
```

---

### 5. Plot results

```python
plot.plot_model_outputs(y_test, y_pred)
```

---

##  Example Notebook

Check out [DataAnalysis.ipynb](https://github.com/Abhijeet-Real/DataPreProcessor/blob/main/DataAnalysis.ipynb)  for a full example pipeline from preprocessing to visualization.

---

##  Dependencies

- `scipy` `pandas` `ipython`
- `seaborn` `openpyxl`
- `matplotlib` `scikit-learn`
- `imbalanced-learn`

---

## Contributing

Feel free to fork and improve! PRs are welcome for new features, improvements, or bug fixes.

---

## Contact

Made with ❤️ by Abhijeet  
[LinkedIn](https://www.linkedin.com/in/abhijeet-099670300/) | [GitHub](https://github.com/Abhijeet-Real/)

---

## License
MIT © Abhijeet
You're free to use, modify, and distribute this project with proper attribution.

---