# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)import pandas as pd
import numpy as np  # For numerical operations and handling arrays


# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeeseTools import DataPreprocessor

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
    MinMaxScaler,  # Scales features to a given range (default: 0 to 1)
    StandardScaler  # Standardizes features by removing the mean and scaling to unit variance
)

class ScaleMixin:
    # Scales numeric columns of the input DataFrame
    def _scale(self, method: str = "standard") -> "DataPreprocessor":
            """
            Scales numeric columns of the input DataFrame, excluding binary columns, and handles NaN/inf values.

            Args:
                method (str, optional): Scaling method, either 'standard' (default) for StandardScaler 
                                        or 'minmax' for MinMaxScaler.

            Returns:
                DataPreprocessor: The instance of the class with scaled features_df.
            """
            # Select numeric columns only
            numeric_cols = self.features_df.select_dtypes(include=["number"]).columns
            
            # Exclude binary columns (those with only two unique values)
            non_binary_cols = [col for col in numeric_cols if self.features_df[col].nunique() > 2]
            
            # Choose scaler based on the method
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid method. Use 'standard' or 'minmax'.")
            
            # Store scaling logs
            scaling_logs = []
            
            for col in non_binary_cols:
                # Handle infinite values by replacing them with NaN
                self.features_df[col] = self.features_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Handle NaN values by imputing with median
                if self.features_df[col].isna().sum() > 0:
                    self.features_df[col] = self.features_df[col].fillna(self.features_df[col].median())
                
                # Log original min and max values
                original_min = self.features_df[col].min()
                original_max = self.features_df[col].max()
                
                # Apply scaling
                self.features_df[col] = scaler.fit_transform(self.features_df[[col]])
                
                # Log scaled min and max values
                scaled_min = self.features_df[col].min()
                scaled_max = self.features_df[col].max()
                
                # Store log
                scaling_logs.append({
                    "Column Name": col,
                    "Scaling Method": method.capitalize(),
                    "Original Min": original_min,
                    "Original Max": original_max,
                    "Scaled Min": scaled_min,
                    "Scaled Max": scaled_max
                })
            
            # Store scaling logs in self.scale_log_df
            self.scale_log_df = pd.DataFrame(scaling_logs)