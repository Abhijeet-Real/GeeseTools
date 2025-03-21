# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)import pandas as pd

# Display utilities for Jupyter Notebooks
from IPython.display import display  # Displays dataframes in Jupyter Notebook in a readable format

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dpp.DPP import DataPreprocessor

class DisplayMixin:
    # Display all features without of DataFrame
    def display_all_features(self) -> "DataPreprocessor":
        """
        Display all features without of DataFrame without truncation.

        Args:
            dataframe (pd.DataFrame): The DataFrame to display.

        Returns:
            None
        """
        # Set display option to show all columns
        pd.set_option("display.max_columns", None)

        # Display the DataFrame
        display(self.working_df)

        # Reset display option to default
        pd.reset_option("display.max_columns")