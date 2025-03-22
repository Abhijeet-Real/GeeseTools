# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)import pandas as pd

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeeseTools import DataPreprocessor

class DropFeaturesMixin:
    # Remove features from the DataFrame
    def _drop_features(self) -> "DataPreprocessor":
        """
        Remove features from the DataFrame with a missing percentage higher than the given threshold.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - The cleaned DataFrame with selected features.
                - A DataFrame listing omitted features and their missing percentages.
        """

        # Calculate missing percentage for each column
        missing_percentage = (self.working_df.isnull().sum() / len(self.working_df)) * 100

        # Identify variables to omit (missing percentage > threshold)
        variables_to_omit = missing_percentage[missing_percentage > self.missing_threshold]

        # Create a DataFrame for omitted variables
        self.dropped_features_log_df = pd.DataFrame({
            "Variable": variables_to_omit.index,
            "Missing Percentage": variables_to_omit.values.round(2)
        })

        # Identify variables to keep
        variables_to_keep = missing_percentage[missing_percentage <= self.missing_threshold].index.tolist()

        # Filter the dataset without modifying self.working_df
        self.working_df = self.working_df[variables_to_keep]