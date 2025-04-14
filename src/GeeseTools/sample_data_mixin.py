import pandas as pd

class SampleDataMixin:
        # Dynamically samples a Pandas DataFrame based on the given input value.
    def _sample_data(self) -> pd.DataFrame:
        """
        Dynamically samples a Pandas DataFrame based on the given input value.
        """

        self.num_rows = len(self.working_df)  # Get the number of rows in the DataFrame


          
        if 0 < self.sample_size < 100:  # Use as fraction
            self.sample_size /= 100 # Convert percentage to fraction
            self.working_df.sample(frac=self.sample_size)

        
        elif self.sample_size > 100 and isinstance(self.sample_size, int) and self.sample_size < self.num_rows:  # Use as n
            self.working_df.sample(n=self.sample_size)

        # Else, do nothing — keep original DataFrame
