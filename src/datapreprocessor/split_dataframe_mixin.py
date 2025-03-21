# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.datapreprocessor.DPP import DataPreprocessor

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split  # Splits dataset into training and testing sets

class SplitDataFrameMixin:
        # Performs Train Test Splits
    def _split_dataframe(self) -> "DataPreprocessor":
        """
        Train Test Splits.

        Args:
            None
        Returns:
            pd.DataFrame: Training Dataframe of features
            pd.DataFrame: Testing Dataframe of features
            pd.DataFrame: Training Dataframe of target
            pd.DataFrame: Testing Dataframe of target
        """

        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features_df, self.target_df, 
                                                            test_size=self.train_test_split/100, stratify=self.target_df, 
                                                            random_state=1112)