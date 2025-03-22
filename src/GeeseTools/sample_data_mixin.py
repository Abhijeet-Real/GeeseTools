import pandas as pd

class SampleDataMixin:
        # Dynamically samples a Pandas DataFrame based on the given input value.
    def _sample_data(self) -> pd.DataFrame:
        """
        Dynamically samples a Pandas DataFrame based on the given input value.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to sample from.
        input_value : Union[int, float]
            Determines how sampling is performed:
            - If between 0 and 1 (exclusive), it is treated as `frac`.
            - If greater than 1, a whole number, and less than total rows, it is treated as `n`.
            - Otherwise, the function returns the original DataFrame.

        Returns:
        --------
        pd.DataFrame
            A new DataFrame containing the sampled rows, or the original DataFrame if input_value is invalid.
        """

        self.num_rows = len(self.working_df)  # Get the number of rows in the DataFrame


          
        if 0 < self.sample_size < 100:  # Use as fraction
            self.sample_size /= 100 # Convert percentage to fraction
            self.working_df.sample(frac=self.sample_size)

        
        elif self.sample_size > 100 and isinstance(self.sample_size, int) and self.sample_size < self.num_rows:  # Use as n
            self.working_df.sample(n=self.sample_size)

        # Else, do nothing — keep original DataFrame
