# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)
import numpy as np  # For numerical operations and handling arrays

# Handling missing values
from sklearn.impute import SimpleImputer  # For imputing missing values with statistical methods

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.dpp.DPP import DataPreprocessor

class ImputeFeaturesMixin:
    # Impute missing values
    def _impute_features(self) -> "DataPreprocessor":
        """
        Impute missing values in numeric columns of the DataFrame.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - The updated DataFrame after imputation.
                - A DataFrame with imputation details for each column.
        """
        imputation_details = []

        for col in self.non_categorical_features:
            # Replace invalid values with NaN explicitly
            self.working_df[col] = self.working_df[col].replace([np.inf, -np.inf], np.nan)

            # Skip if column has no missing values
            if self.working_df[col].isnull().sum() == 0:
                imputation_details.append({
                    "Variable": col,
                    "Imputation Method": "None (No Missing Values)",
                    "Significant Difference": 0,
                    "Percentage Difference": 0.00
                })
                continue

            # Check if the column is binary (only 2 unique values)
            unique_values = self.working_df[col].dropna().unique()
            if len(unique_values) == 2:
                imputation_method = "Median"
                imputer = SimpleImputer(strategy="median")
            else:
                # Calculate mean and median
                col_mean = self.working_df[col].mean()
                col_median = self.working_df[col].median()

                # Calculate percentage difference
                percentage_diff = abs(col_mean - col_median) / max(abs(col_mean), abs(col_median)) * 100
                significant_diff = int(percentage_diff > 10)  # Binary: 1 if significant, 0 otherwise

                # Choose strategy based on significant difference
                if significant_diff:
                    imputation_method = "Median"
                    imputer = SimpleImputer(strategy="median")
                else:
                    imputation_method = "Mean"
                    imputer = SimpleImputer(strategy="mean")

            # Apply the imputer
            self.working_df[[col]] = imputer.fit_transform(self.working_df[[col]])

            # Append details to the list
            imputation_details.append({
                "Variable": col,
                "Imputation Method": imputation_method,
                "Significant Difference": significant_diff if "significant_diff" in locals() else 0,
                "Percentage Difference": round(percentage_diff, 2) if "percentage_diff" in locals() else 0.00
            })

        # Handle missing values for categorical columns using the most frequent value
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        for col in self.categorical_features:
            # Apply imputation only if the column has missing values
            if self.working_df[col].isnull().sum() > 0:
                self.working_df[[col]] = categorical_imputer.fit_transform(self.working_df[[col]])

                # Log imputation details
                imputation_details.append({
                    "Variable": col,
                    "Imputation Method": "Most Frequent",
                    "Significant Difference": "N/A",
                    "Percentage Difference": "N/A"
                })

        # Update the imputation log DataFrame with categorical imputation details
        self.imputation_log_df = pd.DataFrame(imputation_details)