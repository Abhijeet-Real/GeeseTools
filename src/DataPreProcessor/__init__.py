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

from .DataPreProcessor import DataPreProcessor
__all__ = ["DataPreProcessor"]
