# Importing essential libraries

# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)

# Type hinting for better code readability and function definitions
from typing import Union, Optional 

# Import mixins class
from .unique_value_summary_mixin import UniqueValueSummaryMixin
from .missing_data_summary_mixin import MissingDataSummaryMixin
from .display_mixin import DisplayMixin

from .sample_data_mixin import SampleDataMixin
from .feature_type_mixin import FeatureTypeMixin
from .to_numeric_mixin import ToNumericMixin
from .drop_features_mixin import DropFeaturesMixin
from .drop_records_mixin import DropRecordsMixin
from .impute_features_mixin import ImputeFeaturesMixin
from .feature_target_split_mixin import FeatureTargetSplitMixin
from .encode_mixin import EncodeMixin
from .transform_mixin import TransformMixin
from .scale_mixin import ScaleMixin
from .split_dataframe_mixin import SplitDataFrameMixin
from .oversample_mixin import OverSampleMixin
from .pre_process_mixin import PreProcessMixin

# Import dataset
from .datasets import load_heart_dataset


# A comprehensive data preprocessing class
class GeeseTools(UniqueValueSummaryMixin, 
                       MissingDataSummaryMixin, 
                       DisplayMixin,
                       SampleDataMixin, 
                       FeatureTypeMixin, 
                       ToNumericMixin, 
                       DropFeaturesMixin, 
                       DropRecordsMixin, 
                       ImputeFeaturesMixin,
                       FeatureTargetSplitMixin, 
                       EncodeMixin, 
                       TransformMixin,
                       ScaleMixin, 
                       SplitDataFrameMixin, 
                       OverSampleMixin, 
                       PreProcessMixin):
    """A comprehensive data preprocessing class that handles missing values, categorical encoding, 
    feature transformation, and scaling. This class automates data cleaning steps, ensuring 
    structured and efficient preprocessing for machine learning models."""

    # Initializes the DataPreprocessor class with the input dataset and preprocessing configurations.
    def __init__(self, dataframe: Optional[pd.DataFrame] = None, 
                 target_variable: Union[str, list] = 'target', 
                 sample_size: Union[int, float] = 100,              # Related to _sample_data
                 missing_threshold: float = 25,                     # Related to _impute_features
                 ordinal_features: list = [],                       # Related to _encode
                 ordinal_categories: Optional[list] = None,         # Related to _encode
                 use_one_hot_encoding: bool = False,                # Related to _encode
                 cv_split_percentage: int = 20,                     # Related to __split_dataframe
                 oversample: bool = False                           # Related to __oversample_data
                 ) -> None:
        """
        Initializes the DataPreprocessor class with the input dataset and preprocessing configurations.

        This method sets up essential attributes, identifies categorical and numerical features, 
        and applies preprocessing configurations, including handling missing values, encoding 
        categorical variables, and logging transformations.

    
        Args:
            dataframe (pd.DataFrame): The input dataset to be preprocessed.
            target_variable (Union[str, list]): The target column(s) for prediction or classification.
            sample_size (Optional[Union[int, float]], optional): The number of records to sample from the dataset.
                If an integer, it specifies the exact number of records. If a float (0 < x ≤ 100), it represents the percentage 
                of data to sample. Defaults to None.
            missing_threshold (float, optional): The percentage threshold for dropping features with missing values.
                Features exceeding this threshold will be removed. Defaults to 25.
            ordinal_features (list, optional): List of categorical features that should be ordinal encoded. Defaults to an empty list.
            ordinal_categories (Optional[list], optional): List of lists specifying the category order for ordinal features.
                Each list should contain category values in increasing order of rank. Defaults to None.
            use_one_hot_encoding (bool, optional): Whether to apply one-hot encoding to categorical features. Defaults to False.
            cv_split_percentage (int, optional): The percentage of data to allocate for testing during train-test split.
                Defaults to 20 (i.e., 20% test and 80% train).
            oversample (bool, optional): Whether to apply oversampling techniques to balance imbalanced datasets. Defaults to False.


        Returns:
            None: This method initializes the preprocessing class and prepares attributes for subsequent transformations.
        """
        
        # Summary of orignal Dataframe
        self.missing_data_summary_df: pd.DataFrame = None
        self.unique_value_summary_df: pd.DataFrame = None
        
        # Logs of various preprocessing steps
        self.to_numeric_log_df: pd.DataFrame = None
        self.dropped_features_log_df: pd.DataFrame = None
        self.dropped_records_log_df: pd.DataFrame = None
        self.imputation_log_df: pd.DataFrame = None
        self.encode_log_df: pd.DataFrame = None
        self.transformation_log_df: pd.DataFrame = None
        self.scale_log_df: pd.DataFrame = None

        if dataframe is None:
            self.working_df = load_heart_dataset()
        else:
            # Setting up working dataframe
            self.working_df = dataframe.copy()

        # Sample Size
        self.sample_size = sample_size
        self.oversample = oversample

        
        # Defining Variables
        self.target_variable = target_variable
        # Ensure target_variable is a list
        if isinstance(target_variable, str):
            self.target_variable = [target_variable]

        # Identify feature types
        self.categorical_features, self.non_categorical_features = self._feature_type()
        
        self.ordinal_features = ordinal_features
        self.ordinal_categories = ordinal_categories
        # Identify nominal features by excluding ordinal features
        self.nominal_features = [col for col in self.categorical_features if col not in self.ordinal_features + self.target_variable]

        # One Hot Encoding
        self.use_one_hot_encoding = use_one_hot_encoding
        # Missing thresholde limit
        self.missing_threshold = missing_threshold
        # Train test split
        self.train_test_split = cv_split_percentage

        # New DataFrames
        self.features_df: pd.DataFrame = None
        self.target_df: pd.DataFrame = None

        # Final DataFrames
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

# Expose mixin methods to IDEs
def _expose_mixin_methods(self):
    """
    Dummy method to expose mixin methods to IDEs like VSCode.
    """
    self.display_df
    self.pre_process
    self.show_missing_data_summary
    self.get_unique_value_summary
