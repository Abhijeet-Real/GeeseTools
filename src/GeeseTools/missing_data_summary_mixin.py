# Data manipulation and numerical operations
import pandas as pd  

class MissingDataSummaryMixin:
    # Computes a summary of missing values for each column in the dataset.
    def missing_data_summary(self) -> pd.DataFrame:
        """Computes a summary of missing values for each column in the dataset.

        This method calculates the total number of missing values per column, 
        the percentage of missing values relative to the total dataset, and 
        presents a structured summary. This analysis helps in identifying 
        features that may require imputation or removal based on the 
        missing data threshold.

        Args:
            None


        Returns:
            pd.DataFrame: 
        """

        missing_count = self.working_df.isnull().sum()
        missing_percentage = (missing_count / len(self.working_df)) * 100

        self.missing_data_summary_df = pd.DataFrame({
            "Variable": self.working_df.columns,
            "Missing Count": missing_count.values,
            "Missing Percentage": missing_percentage.round(2).astype(str) + "%"
        }).reset_index(drop=True)

        return self.missing_data_summary_df