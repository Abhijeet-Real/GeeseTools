# Data manipulation and numerical operations
import pandas as pd  
import numpy as np  

# Handling missing values
from sklearn.impute import SimpleImputer  # For imputing missing values with statistical methods

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

class ImputeFeaturesMixin:
    # Impute missing values
    def _impute_features(self) -> "DataPreProcessor":
        """
        Main method to impute missing values and store imputation log.
        """
        self.imputation_logs = []

        for col in self.non_categorical_features:
            self.__impute_numerical_column(col)

        self.__impute_categorical_columns()

        self.imputation_log_df = pd.DataFrame(self.imputation_logs)
        

    def __impute_numerical_column(self, col):
        self.working_df[col] = self.working_df[col].replace([np.inf, -np.inf], np.nan)

        if self.working_df[col].isnull().sum() == 0:
            self._log_imputation_details(col, "None (No Missing Values)", 0, 0.00)
            return

        unique_values = self.working_df[col].dropna().unique()

        if len(unique_values) == 2:
            imputer = SimpleImputer(strategy="median")
            method = "Median"
            sig_diff = 0
            pct_diff = 0.00
        else:
            mean = self.working_df[col].mean()
            median = self.working_df[col].median()
            pct_diff = abs(mean - median) / max(abs(mean), abs(median)) * 100
            sig_diff = int(pct_diff > 10)

            if sig_diff:
                imputer = SimpleImputer(strategy="median")
                method = "Median"
            else:
                imputer = SimpleImputer(strategy="mean")
                method = "Mean"

        self.working_df[[col]] = imputer.fit_transform(self.working_df[[col]])
        self._log_imputation_details(col, method, sig_diff, round(pct_diff, 2))

    def __impute_categorical_columns(self):
        imputer = SimpleImputer(strategy="most_frequent")

        for col in self.categorical_features:
            if self.working_df[col].isnull().sum() > 0:
                self.working_df[[col]] = imputer.fit_transform(self.working_df[[col]])
                self._log_imputation_details(col, "Most Frequent", "N/A", "N/A")

    def _log_imputation_details(self, col, method, sig_diff, pct_diff):
        self.imputation_logs.append({
            "Variable": col,
            "Imputation Method": method,
            "Significant Difference": sig_diff,
            "Percentage Difference": pct_diff
        })

    