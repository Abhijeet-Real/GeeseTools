"""
Initializes the DataPreprocessor class with the input dataset and preprocessing configurations.

This method sets up key attributes, identifies categorical and numerical features, and prepares
the dataset for downstream preprocessing steps such as sampling, handling missing values, encoding,
scaling, and transformation — while maintaining logs for traceability.

Args:
    dataframe (Optional[pd.DataFrame]): The input dataset to preprocess. If None, a default dataset 
        (e.g., heart dataset) will be loaded automatically.

    target (Union[str, list]): The name of the target column(s) for prediction or classification.
        If a single column, pass as a string; for multi-output tasks, use a list of column names.

    sample_size (Union[int, float], optional): Specifies how much data to sample.  
        - If an integer: samples that many records (e.g., 500).  
        - If a float: treated as percentage (0 < x ≤ 100) of the data (e.g., 20.0 means 20%).  
        Defaults to 100 (i.e., full dataset).

    test_size_percent (int, optional): The percentage of data to allocate for the test set during 
        train-test split. Must be between 0 and 100. For example, 20 means 80% train and 20% test. 
        Defaults to 20.

    oversample (bool, optional): Whether to apply oversampling (e.g., SMOTE) to balance imbalanced
        classes in the training data. Defaults to False.

    nr_y_bins (int, optional): Number of bins to discretize the target variable. Useful for 
        stratified binning or classification thresholds. Set to 0 to skip binning. Defaults to 0.

    missing_threshold_percent (float, optional): Threshold (%) for dropping features with missing 
        values. Features with a missing percentage above this value are dropped. Range: 0–100. 
        Defaults to 25.

    ordinal_features (list, optional): List of categorical feature names that should be encoded 
        using ordinal encoding. Defaults to an empty list.

    ordinal_categories (Optional[list], optional): List of lists defining category order for each 
        ordinal feature. Each list should contain categories in increasing rank order. Must align 
        with `ordinal_features`. Defaults to None.

    use_one_hot_encoding (bool, optional): If True, applies one-hot encoding to nominal features. 
        If False, uses ordinal encoding for all categorical variables. Defaults to False.

"""
# Importing essential libraries

# Data manipulation and numerical operations
import pandas as pd  

# Type hinting for better code readability and function definitions
from typing                                 import Union, Optional 

# Import mixins class
# Import helper classes
from .unique_value_summary_mixin            import UniqueValueSummaryMixin
from .missing_data_summary_mixin            import MissingDataSummaryMixin
from .display_mixin                         import DisplayMixin
# Import Protected classes
from .sample_data_mixin                     import SampleDataMixin
from .feature_type_mixin                    import FeatureTypeMixin
from .to_numeric_mixin                      import ToNumericMixin
from .drop_features_mixin                   import DropFeaturesMixin
from .drop_records_mixin                    import DropRecordsMixin
from .impute_features_mixin                 import ImputeFeaturesMixin
from .create_bins_mixin                     import CreateBinsMixin
from .feature_target_split_mixin            import FeatureTargetSplitMixin
from .encode_mixin                          import EncodeMixin
from .transform_mixin                       import TransformMixin
from .scale_mixin                           import ScaleMixin
from .split_dataframe_mixin                 import SplitDataFrameMixin
from .oversample_mixin                      import OverSampleMixin
# Import MAIN class
from .pre_process_mixin                     import PreProcessMixin

# Import dataset
from .datasets                              import load_heart_dataset

# A comprehensive data preprocessing class
class DataPreProcessor(UniqueValueSummaryMixin, 
                MissingDataSummaryMixin, 
                DisplayMixin,
                SampleDataMixin, 
                FeatureTypeMixin, 
                ToNumericMixin, 
                DropFeaturesMixin, 
                DropRecordsMixin, 
                ImputeFeaturesMixin,
                CreateBinsMixin,
                FeatureTargetSplitMixin, 
                EncodeMixin, 
                TransformMixin,
                ScaleMixin, 
                SplitDataFrameMixin, 
                OverSampleMixin, 
                PreProcessMixin
                ):
    """
    A comprehensive data preprocessing class that handles missing values, categorical encoding, 
    feature transformation, and scaling. This class automates data cleaning steps, ensuring 
    structured and efficient preprocessing for machine learning models.
    """

    # Initializes the DataPreprocessor class with the input dataset and preprocessing configurations.
    def __init__(self,
             dataframe: Optional[pd.DataFrame] = None,
             target: Union[str, list] = 'diagnosis',

             # Sampling and Splitting
             sample_size: Union[int, float] = 100,
             test_size_percent: int = 20,
             oversample: bool = False,
             num_y_bins: int = 0,

             # Missing Data
             missing_threshold_percent: float = 25,

             # Encoding Settings
             ordinal_features: list = [],
             ordinal_categories: Optional[list] = None,
             use_one_hot_encoding: bool = False
             ) -> None:
        """
        Initialize DataPreProcessor with preprocessing configurations.
        """

        # Load and prepare working DataFrame
        self.working_df = load_heart_dataset() if dataframe is None else dataframe.copy()

        # ======================== Target & Sampling Configuration ========================
        self.target_variable = [target] if isinstance(target, str) else target
        self.sample_size = sample_size
        self.oversample = oversample
        self.nr_y_bins = num_y_bins
        self.train_test_split = test_size_percent

        # ======================== Missing Data Configuration ========================
        self.missing_threshold = missing_threshold_percent

        # ======================== Encoding Configuration ========================
        self.ordinal_features = ordinal_features
        self.ordinal_categories = ordinal_categories
        self.one_hot_encoding = use_one_hot_encoding

        # Identify feature types
        self.categorical_features, self.non_categorical_features = self._feature_type()
        self.nominal_features = [
            col for col in self.categorical_features
            if col not in self.ordinal_features + self.target_variable
        ]

        # ======================== Working Copies of Data ========================
        self.features_df:               pd.DataFrame = None
        self.target_df:                 pd.DataFrame = None

        # ======================== Final Train-Test Splits ========================
        self.X_train:                   pd.DataFrame = None
        self.X_test:                    pd.DataFrame = None
        self.y_train:                   pd.DataFrame = None
        self.y_test:                    pd.DataFrame = None

        # ======================== Summary DataFrames ========================
        self.missing_data_summary_df:   pd.DataFrame = None
        self.unique_value_summary_df:   pd.DataFrame = None

        # ======================== Transformation Logs ========================
        self.to_numeric_log_df:         pd.DataFrame = None
        self.dropped_features_log_df:   pd.DataFrame = None
        self.dropped_records_log_df:    pd.DataFrame = None
        self.imputation_log_df:         pd.DataFrame = None
        self.encode_log_df:             pd.DataFrame = None
        self.transformation_log_df:     pd.DataFrame = None
        self.scale_log_df:              pd.DataFrame = None


    # Expose mixin methods to IDEs
    def _expose_mixin_methods(self):
        """
        Dummy method to expose mixin methods to IDEs like VSCode.
        """

        self.display_all_features
        self.unique_value_summary
        self.missing_data_summary
        self.pre_process