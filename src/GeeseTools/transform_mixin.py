# Data manipulation and numerical operations
import pandas as pd  
import numpy as np  

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeeseTools import GeeseTools

# Statistical transformations
from scipy.stats import boxcox  # Applies Box-Cox transformation to normalize skewed data

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
        PowerTransformer,  # Applies power transformations to stabilize variance and reduce skewness
)

class TransformMixin:
    def _transform(self) -> "GeeseTools":
        """
        Apply transformations to numeric columns based on skewness and kurtosis.
        """
        self.__clean_dataframe()

        # Exclude target variable from transformation
        columns = [col for col in self.non_categorical_features if col not in self.target_variable]

        transformation_logs = []

        for column in columns:
            skewness = self.features_df[column].skew()
            kurtosis = self.features_df[column].kurt()
            action = "None"

            # FIX: Ensure log1p() only gets strictly positive values
            if skewness > 1:
                action = "Log Transformation"
                series = self.features_df[column].astype(float)  # Explicit cast BEFORE clip
                clipped = series.clip(lower=1e-9)
                self.features_df[column] = np.log1p(clipped)


            elif skewness < -1:
                action = "Reflect and Log Transformation"
                max_val = self.features_df[column].max()
                series = (max_val - self.features_df[column]).astype(float)
                clipped = series.clip(lower=1e-9)
                self.features_df[column] = np.log1p(clipped)


            if kurtosis > 3:
                try:
                    action = "Box-Cox Transformation"
                    self.features_df[column], _ = boxcox(self.features_df[column].clip(lower=1))
                    self.features_df[column] = self.features_df[column].astype(float)  
                except ValueError:
                    action = "Box-Cox Failed, Applied Yeo-Johnson"
                    transformer = PowerTransformer(method="yeo-johnson")
                    self.features_df[column] = pd.Series(
                        transformer.fit_transform(self.features_df[[column]]).flatten(),
                        index=self.features_df.index
                    )

            elif kurtosis < 3 and action == "None":
                action = "Yeo-Johnson Transformation"
                transformer = PowerTransformer(method="yeo-johnson")
                self.features_df[column] = pd.Series(
                    transformer.fit_transform(self.features_df[[column]]).flatten(),
                    index=self.features_df.index
                )

            transformation_logs.append({
                "Column Name": column,
                "Skewness Before": skewness,
                "Kurtosis Before": kurtosis,
                "Action Taken": action,
                "Skewness After": self.features_df[column].skew(),
                "Kurtosis After": self.features_df[column].kurt()
            })

        self.transformation_log_df = pd.DataFrame(transformation_logs)

    def __clean_dataframe(self) -> None:
        """
        Cleans the given DataFrame by:
        - Replacing infinite values (inf, -inf) with NaN
        - Optionally filling NaN values with 0 (if required)
        """
        self.features_df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

        # Fill NaNs only if needed
        for col in self.features_df.columns:
            if self.features_df[col].isna().all():
                self.features_df[col].fillna(0, inplace=True)  # Replace all NaNs with 0 ONLY if column is empty
