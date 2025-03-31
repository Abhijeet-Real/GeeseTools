from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .GeeseTools import GeeseTools

class CreateBinsMixin:
    def _create_y_bins(self) -> "GeeseTools":
        """
        Creates bins for the target variable in the working DataFrame.

        This method checks if `nr_y_bins` is greater than zero and if `target_variable` 
        is a numeric column in `self.working_df`. If these conditions are met, 
        it determines the minimum and maximum values of `target_variable` 
        and creates `nr_y_bins` equal-width bins. The `target_variable` 
        values are then categorized into these bins.

        Returns:
            GeeseTools: The modified instance with updated `working_df`.

        Note:
            - Modifies `self.working_df` in place.
            - The function assumes `target_variable` exists in `working_df` and is numeric.
        """

        if self.target_df.ndim == 1:  # If target_df is a Series
            is_numeric = self.target_df.dtype.kind in "ifc"
        else:  # If target_df is a DataFrame
            is_numeric = all(dtype.kind in "ifc" for dtype in self.target_df.dtypes)

        if self.target_df.ndim == 1:  # If self.target_df is a Series
            if is_numeric:
                min_val = self.target_df.min()
                max_val = self.target_df.max()

                # Creating bins
                bin_edges = np.linspace(min_val, max_val, self.nr_y_bins + 1)
                bin_labels = [f"{bin_edges[i]} - {bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
                self.target_df = pd.cut(self.target_df, bins=bin_edges, labels=bin_labels, include_lowest=True).astype(str)

        else:  # If self.target_df is a DataFrame
            if is_numeric:
                for col in self.target_variable:
                    min_val = self.target_df[col].min()
                    max_val = self.target_df[col].max()

                    # Creating bins
                    bin_edges = np.linspace(min_val, max_val, self.nr_y_bins + 1)
                    bin_labels = [f"{bin_edges[i]} - {bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
                    self.target_df[col] = pd.cut(self.target_df[col], bins=bin_edges, labels=bin_labels, include_lowest=True).astype(str)

                print(self.target_df)
