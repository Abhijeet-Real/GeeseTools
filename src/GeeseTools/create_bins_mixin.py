from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

class CreateBinsMixin:
    def _create_y_bins(self) -> "DataPreProcessor":
        """
        Creates bins for the target variable in the working DataFrame.
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
                bin_labels = self._format_bin_edges(bin_edges)
                self.target_df = pd.cut(self.target_df, bins=bin_edges, labels=bin_labels, include_lowest=True).astype(str)

        else:  # If self.target_df is a DataFrame
            if is_numeric:
                for col in self.target_variable:
                    min_val = self.target_df[col].min()
                    max_val = self.target_df[col].max()

                    # Creating bins
                    bin_edges = np.linspace(min_val, max_val, self.nr_y_bins + 1)
                    bin_labels = self._format_bin_edges(bin_edges)
                    self.target_df[col] = pd.cut(self.target_df[col], bins=bin_edges, labels=bin_labels, include_lowest=True).astype(str)

                print(self.target_df)


    def _format_bin_edges(self, bin_edges, threshold=1e-20):
        def format_val(val):
            val = 0.0 if abs(val) < threshold else round(val, 2)
            return f"{val:.0f}"

        return [f"{format_val(bin_edges[i])} - {format_val(bin_edges[i + 1])}"
                for i in range(len(bin_edges) - 1)]
