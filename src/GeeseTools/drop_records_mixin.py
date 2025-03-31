# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeeseTools import GeeseTools

class DropRecordsMixin:
    # Removes records from the DataFrame
    def _drop_records(self) -> "GeeseTools":
        """
        Remove records from the DataFrame where the percentage of missing values 
        exceeds the specified threshold.

        Args:
            None

        """

        # Calculate the threshold for missing values based on the given percentage
        threshold = (self.missing_threshold / 100) * self.working_df.shape[1]

        # Identify records with missing values exceeding the threshold
        self.dropped_records_log_df = self.working_df[self.working_df.isnull().sum(axis=1) > threshold]

        # Create a cleaned DataFrame without modifying self.working_df
        self.working_df = self.working_df.drop(index=self.dropped_records_log_df.index)