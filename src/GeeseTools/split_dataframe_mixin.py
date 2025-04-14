# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split 

class SplitDataFrameMixin:
        # Performs Train Test Splits
    def _split_dataframe(self) -> "DataPreProcessor":
        """
        Train Test Splits.
        
        Returns:
            pd.DataFrame: Training Dataframe of features
            pd.DataFrame: Testing Dataframe of features
            pd.DataFrame: Training Dataframe of target
            pd.DataFrame: Testing Dataframe of target
        """

        try:
        # Perform train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features_df, self.target_df, 
                                                            test_size=self.train_test_split/100, stratify=self.target_df, 
                                                            random_state=1112)
        except ValueError  as e:
            if self.target_df.ndim == 1:  # If target_df is a Series
                is_numeric = self.target_df.dtype.kind in "ifc"
            else:  # If target_df is a DataFrame
                is_numeric = all(dtype.kind in "ifc" for dtype in self.target_df.dtypes)
        
            if is_numeric and self.nr_y_bins == 0:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features_df, self.target_df, 
                                                        test_size=self.train_test_split/100, random_state=1112)