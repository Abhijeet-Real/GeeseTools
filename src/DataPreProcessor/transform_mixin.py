# Data manipulation and numerical operations
import pandas as pd  
import numpy as np  

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

# Statistical transformations
from scipy.stats import boxcox  # Applies Box-Cox transformation to normalize skewed data

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
        PowerTransformer,  # Applies power transformations to stabilize variance and reduce skewness
)

class TransformMixin:
    def _transform(self) -> "DataPreProcessor":
        """
        Apply transformations to numeric columns based on skewness and kurtosis.
        """
        self.__clean_dataframe()
        self.transformation_logs = []

        columns = [col for col in self.non_categorical_features if col not in self.target_variable]

        for column in columns:
            self.__transform_column(column)

        self.transformation_log_df = pd.DataFrame(self.transformation_logs)
        return self

    def __transform_column(self, column: str):
        skewness_before = self.features_df[column].skew()
        kurtosis_before = self.features_df[column].kurt()
        action = "None"

        series = self.features_df[column].astype(float)

        if skewness_before > 1:
            action = "Log Transformation"
            self.features_df[column] = np.log1p(series.clip(lower=1e-9))

        elif skewness_before < -1:
            action = "Reflect and Log Transformation"
            max_val = series.max()
            reflected = (max_val - series).clip(lower=1e-9)
            self.features_df[column] = np.log1p(reflected)

        # Apply power transformation based on kurtosis
        transformed = False
        if kurtosis_before > 3:
            try:
                action = "Box-Cox Transformation"
                self.features_df[column], _ = boxcox(self.features_df[column].clip(lower=1))
                transformed = True
            except ValueError:
                action = "Box-Cox Failed, Applied Yeo-Johnson"
        if not transformed and (kurtosis_before > 3 or kurtosis_before < 3 and action == "None"):
            action = "Yeo-Johnson Transformation"
            transformer = PowerTransformer(method="yeo-johnson")
            self.features_df[column] = transformer.fit_transform(self.features_df[[column]]).flatten()

        self._log_transformation(
            column,
            skewness_before,
            kurtosis_before,
            action,
            self.features_df[column].skew(),
            self.features_df[column].kurt()
        )

    def _log_transformation(self, column, skew_before, kurt_before, action, skew_after, kurt_after):
        self.transformation_logs.append({
            "Column Name": column,
            "Skewness Before": skew_before,
            "Kurtosis Before": kurt_before,
            "Action Taken": action,
            "Skewness After": skew_after,
            "Kurtosis After": kurt_after
        })

    def __clean_dataframe(self) -> None:
        """
        Replaces infinite values with NaN and fills entire-NaN columns with 0.
        """
        self.features_df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

        for col in self.features_df.columns:
            if self.features_df[col].isna().all():
                self.features_df[col].fillna(0, inplace=True)