# Data manipulation and numerical operations
import pandas as pd

# Type hinting for better code readability and function definitions
from typing import Union, Tuple

class PreProcessMixin():
    # Pre-process the input DataFrame by performing the specific steps
    def pre_process(self) -> Tuple[ pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
        """
        Pre-process the input DataFrame by performing the following steps:
        1. Drop variables with more than `missing_threshold%` missing values.
        2. Drop records with more than `missing_threshold%` missing values.
        3. Impute missing values.
        4. Encode categorical variables.
        5. Transform features.
        6. Scale numeric features.

        Args:
            None
        Returns:
            Tuple containing:
            - Processed features DataFrame.
            - Target variable (Series or DataFrame).
        """
        
        # Sample Size
        self._sample_data()

        # Convert possible numeric-like strings
        self._to_numeric()

        # Drop variables with more than `missing_threshold%` missing values
        self._drop_features()

        # Identify feature types
        self.categorical_features, self.non_categorical_features = self._feature_type()

        # Identify nominal features by excluding ordinal features
        self.nominal_features = [col for col in self.categorical_features if col not in self.ordinal_features + self.target_variable]
        
        # Drop records with more than `missing_threshold%` missing values
        self._drop_records()

        # Impute missing values
        self._impute_features()

        # Split features and target
        self._feature_target_split()

        # Encode categorical variables
        self._encode()

        # Transform features
        self._transform()

        # Scale numeric features
        self._scale()

        self._split_dataframe()

        # Free up space
        self.features_df = None
        self.target_df = None

        # Oversample data if required
        if self.oversample:
            self._oversample_data()

        return self.X_train, self.X_test, self.y_train, self.y_test