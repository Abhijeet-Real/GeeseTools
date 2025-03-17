# DataPreprocessor: A Comprehensive Data Preprocessing Utility

## Overview
The `DataPreprocessor` class is designed to automate and streamline the data preprocessing pipeline for machine learning projects. It handles essential preprocessing steps such as missing value imputation, categorical encoding, feature transformation, scaling, and train-test splitting. This ensures a structured and efficient workflow for preparing datasets for machine learning models.

## Features
- **Handling Missing Values**:
  - Drops features and records based on a missing value threshold.
  - Imputes missing values using statistical methods (mean, median, or mode).

- **Feature Encoding**:
  - Supports both **Ordinal Encoding** (for ordinal categorical features with meaningful order) and **One-Hot Encoding** (for nominal categorical features).

- **Feature Transformation**:
  - Log, Box-Cox, and Yeo-Johnson transformations to normalize skewed data and stabilize variance.

- **Feature Scaling**:
  - Supports **Min-Max Scaling** (scales values between 0 and 1) and **Standard Scaling** (removes mean and scales to unit variance).

- **Train-Test Splitting**:
  - Splits the dataset into training and testing sets based on a configurable percentage.

- **Imbalanced Data Handling**:
  - Supports **Random Oversampling** to balance imbalanced datasets.

- **Logging & Summarization**:
  - Provides summary tables for missing values, unique values, encoding transformations, and other preprocessing steps.

## Installation
This module relies on standard Python libraries and `scikit-learn`. Install the required dependencies using:
```bash
pip install pandas numpy scikit-learn imbalanced-learn scipy
```

## Usage
### Importing the module
```python
import pandas as pd
from data_preprocessor import DataPreprocessor
```

### Initializing the Preprocessor
```python
# Load dataset
df = pd.read_csv("data.csv")

# Initialize DataPreprocessor
preprocessor = DataPreprocessor(
    dataframe=df,
    target_variable="target_column",
    sample_size=100,
    missing_threshold=25,
    ordinal_features=["education_level"],
    ordinal_categories=[["High School", "Bachelor's", "Master's", "PhD"]],
    use_one_hot_encoding=True,
    cv_split_percentage=20,
    oversample=True
)
```

### Generating Data Summary
```python
# Summary of unique values per feature
unique_summary = preprocessor.unique_value_summary()

# Summary of missing values
missing_summary = preprocessor.missing_data_summary()
```

### Executing Data Preprocessing
```python
X_train, X_test, y_train, y_test = preprocessor.pre_process()
```

### Displaying Preprocessed Data
```python
preprocessor.display_all_features()
```

## Methods & Functions

### `unique_value_summary()`
Generates a summary of unique values for each column in the dataset.

### `missing_data_summary()`
Computes the count and percentage of missing values for each column.

### `pre_process()`
Performs the complete preprocessing pipeline:
1. Drops features with excessive missing values.
2. Drops records with excessive missing values.
3. Imputes missing values.
4. Encodes categorical variables.
5. Transforms features.
6. Scales numeric features.
7. Splits the dataset into training and testing sets.
8. Applies oversampling (if specified).

### `display_all_features()`
Displays all columns of the dataset without truncation.

## Internal Methods
These are used within the class for data transformation:
- `__feature_type()`: Identifies categorical and numerical features.
- `__sample_data()`: Dynamically samples data.
- `__to_numeric()`: Converts possible numeric-like strings.
- `__drop_features()`: Drops features based on missing value percentage.
- `__drop_records()`: Drops records exceeding missing value threshold.
- `__impute_features()`: Handles missing value imputation.
- `__feature_target_split()`: Separates features and target variable.
- `__encode()`: Applies categorical encoding.
- `__transform()`: Applies log, Box-Cox, or Yeo-Johnson transformations.
- `__scale()`: Scales numeric columns.
- `__split_dataframe()`: Splits the data into training and testing sets.
- `__oversample_data()`: Performs random oversampling for imbalanced datasets.

## Customization
Users can adjust the preprocessing configurations based on:
- **Sampling**: Specify the sample size as an integer (number of records) or a float (percentage).
- **Missing Value Handling**: Adjust the missing value threshold.
- **Encoding Methods**: Choose between one-hot encoding or ordinal encoding.
- **Scaling Methods**: Select between standard scaling and min-max scaling.
- **Train-Test Split Ratio**: Define the split percentage.
- **Oversampling**: Enable or disable oversampling for handling imbalanced datasets.

## Example Workflow
```python
# Load data
import pandas as pd

# Example dataset
data = pd.DataFrame({
    "age": [25, 30, 35, None, 40],
    "income": [50000, 60000, None, 80000, 90000],
    "education": ["Bachelor's", "Master's", "PhD", "High School", None],
    "purchase": [1, 0, 1, 0, 1]
})

# Initialize the preprocessor
preprocessor = DataPreprocessor(
    dataframe=data,
    target_variable="purchase",
    missing_threshold=20,
    use_one_hot_encoding=True
)

# Run preprocessing
X_train, X_test, y_train, y_test = preprocessor.pre_process()

# Display processed data
preprocessor.display_all_features()
```

## Conclusion
The `DataPreprocessor` class simplifies the data preprocessing pipeline, making it efficient and structured. It ensures that data is cleaned, transformed, and ready for machine learning models with minimal manual intervention.

---
**Author:** Abhijeet  
**License:** MIT