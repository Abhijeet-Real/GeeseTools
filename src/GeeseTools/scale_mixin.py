# Data manipulation and numerical operations
import pandas as pd  
import numpy as np  


# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
    MinMaxScaler,  
    StandardScaler  
)

class ScaleMixin:
    # Scales numeric columns of the input DataFrame
    def _scale(self, method: str = "standard") -> "DataPreProcessor":
        """
        Main method to scale numeric columns (excluding binary) and log transformations.
        """
        self.scaling_logs = []
        non_binary_cols = self.__get_non_binary_numeric_columns()

        scaler = self.__get_scaler(method)

        for col in non_binary_cols:
            self.__scale_column(col, scaler, method)

        self.scale_log_df = pd.DataFrame(self.scaling_logs)
        return self

    def __get_non_binary_numeric_columns(self):
        numeric_cols = self.features_df.select_dtypes(include=["number"]).columns
        return [col for col in numeric_cols if self.features_df[col].nunique() > 2]

    def __get_scaler(self, method):
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError("Invalid method. Use 'standard' or 'minmax'.")

    def __scale_column(self, col, scaler, method):
        self.features_df[col] = self.features_df[col].replace([np.inf, -np.inf], np.nan)

        if self.features_df[col].isna().sum() > 0:
            self.features_df[col] = self.features_df[col].fillna(self.features_df[col].median())

        original_min = self.features_df[col].min()
        original_max = self.features_df[col].max()

        self.features_df[col] = scaler.fit_transform(self.features_df[[col]])

        scaled_min = self.features_df[col].min()
        scaled_max = self.features_df[col].max()

        self.__log_scaling_details(col, method.capitalize(), original_min, original_max, scaled_min, scaled_max)

    def __log_scaling_details(self, col, method, original_min, original_max, scaled_min, scaled_max):
        self.scaling_logs.append({
            "Column Name": col,
            "Scaling Method": method,
            "Original Min": original_min,
            "Original Max": original_max,
            "Scaled Min": scaled_min,
            "Scaled Max": scaled_max
        })