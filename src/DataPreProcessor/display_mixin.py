# Data manipulation and numerical operations
import pandas as pd 

# Display utilities for Jupyter Notebooks
from IPython.display import display 

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

class DisplayMixin:
    # Display all features without of DataFrame
    def display_all_features(self) -> "DataPreProcessor":
        """
        Display all features without of DataFrame without truncation.
        """
        # Set display option to show all columns
        pd.set_option("display.max_columns", None)

        # Display the DataFrame
        display(self.working_df.head())

        # Reset display option to default
        pd.reset_option("display.max_columns")