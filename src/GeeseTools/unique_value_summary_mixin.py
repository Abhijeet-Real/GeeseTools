# Data manipulation and numerical operations
import pandas as pd

class UniqueValueSummaryMixin:
    # Generates a summary of unique values for each column in the dataset.
    def unique_value_summary(self) -> pd.DataFrame:
        """
        Generates a summary of unique values for each column in the dataset.

        This method calculates the number of unique values, total non-null values, 
        and their percentage representation for every column. This is useful 
        for detecting categorical variables, identifying high-cardinality columns, 
        and assessing data distribution.

        Returns:
            pd.DataFrame: 
        """

        unique_counts = self.working_df.nunique()
        total_counts = self.working_df.count()
        percentages = (unique_counts / total_counts) * 100

        self.unique_value_summary_df = pd.DataFrame({
            "Unique Values": unique_counts,
            "Total Values": total_counts,
            "Percentage (%)": percentages
        })

        return self.unique_value_summary_df