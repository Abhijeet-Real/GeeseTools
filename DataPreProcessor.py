# Importing essential libraries

# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)
import numpy as np  # For numerical operations and handling arrays

# Type hinting for better code readability and function definitions
from typing import Union, Tuple ,Optional 

# Handling missing values
from sklearn.impute import SimpleImputer  # For imputing missing values with statistical methods

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
    MinMaxScaler,  # Scales features to a given range (default: 0 to 1)
    OneHotEncoder,  # Converts categorical variables into a binary matrix
    OrdinalEncoder,  # Encodes ordinal categorical features with meaningful order
    PowerTransformer,  # Applies power transformations to stabilize variance and reduce skewness
    StandardScaler  # Standardizes features by removing the mean and scaling to unit variance
)

# Statistical transformations
from scipy.stats import boxcox  # Applies Box-Cox transformation to normalize skewed data

# Display utilities for Jupyter Notebooks
from IPython.display import display  # Displays dataframes in Jupyter Notebook in a readable format

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split  # Splits dataset into training and testing sets

# Sampling techniques for imbalanced datasets
from imblearn.over_sampling import RandomOverSampler  # Randomly oversamples the minority class


# A comprehensive data preprocessing class
class DataPreprocessor:
    """A comprehensive data preprocessing class that handles missing values, categorical encoding, 
    feature transformation, and scaling. This class automates data cleaning steps, ensuring 
    structured and efficient preprocessing for machine learning models."""

    # Initializes the DataPreprocessor class with the input dataset and preprocessing configurations.
    def __init__(self, dataframe: pd.DataFrame, target_variable: Union[str, list], 
                 sample_size: Union[int, float] = 100,  # Related to __sample_data
                 missing_threshold: float = 25,                    # Related to __impute_features
                 ordinal_features: list = [],                      # Related to __encode
                 ordinal_categories: Optional[list] = None,        # Related to __encode
                 use_one_hot_encoding: bool = False,               # Related to __encode
                 cv_split_percentage: int = 20,            # Related to __split_dataframe
                 oversample: bool = False                          # Related to __oversample_data
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
        self.categorical_features, self.non_categorical_features = self.__feature_type()
        
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

    # Generates a summary of unique values for each column in the dataset.
    def unique_value_summary(self) -> pd.DataFrame:
        """
        Generates a summary of unique values for each column in the dataset.

        This method calculates the number of unique values, total non-null values, 
        and their percentage representation for every column. This is useful 
        for detecting categorical variables, identifying high-cardinality columns, 
        and assessing data distribution.

        Args:
            None

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
        self.__sample_data()

        # Convert possible numeric-like strings
        self.__to_numeric()

        # Drop variables with more than `missing_threshold%` missing values
        self.__drop_features()

        # Identify feature types
        self.categorical_features, self.non_categorical_features = self.__feature_type()

        # Identify nominal features by excluding ordinal features
        self.nominal_features = [col for col in self.categorical_features if col not in self.ordinal_features + self.target_variable]
        
        # Drop records with more than `missing_threshold%` missing values
        self.__drop_records()

        # Impute missing values
        self.__impute_features()

        # Split features and target
        self.__feature_target_split()

        # Encode categorical variables
        self.__encode()

        # Transform features
        self.__transform()

        # Scale numeric features
        self.__scale()

        self.__split_dataframe()

        # Free up space
        self.features_df = None
        self.target_df = None

        # Oversample data if required
        if self.oversample:
            self.__oversample_data()

        return self.X_train, self.X_test, self.y_train, self.y_test

    # Display all features without of DataFrame
    def display_all_features(self) -> "DataPreprocessor":
        """
        Display all features without of DataFrame without truncation.

        Args:
            dataframe (pd.DataFrame): The DataFrame to display.

        Returns:
            None
        """
        # Set display option to show all columns
        pd.set_option("display.max_columns", None)

        # Display the DataFrame
        display(self.working_df)

        # Reset display option to default
        pd.reset_option("display.max_columns")

        return self

    # Identifies categorical and non-categorical columns in the DataFrame.
    def __feature_type(self) -> Tuple[list, list]:
        """
        Identifies categorical and non-categorical columns in the DataFrame.

        Args:
            None

        Returns:
            tuple[list, list]: A tuple containing:
                - List of categorical column names.
                - List of non-categorical column names.
        """

        self.categorical_features = self.working_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.non_categorical_features = self.working_df.select_dtypes(exclude=["object", "category"]).columns.tolist()

        return self.categorical_features, self.non_categorical_features

    # Dynamically samples a Pandas DataFrame based on the given input value.
    def __sample_data(self) -> pd.DataFrame:
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

        
        return self  # Return original DataFrame if input_value is invalid

    # Converts all columns in the DataFrame into numeric types where possible.
    def __to_numeric(self) -> "DataPreprocessor":
        """
        Converts all columns in the DataFrame into numeric types where possible.
        
        - Strings will be converted to numeric if feasible.
        - "True"/"False" (case insensitive) will be converted to 1 and 0.
        - If a value cannot be converted to numeric, it will remain as is.
        - A summary DataFrame is created to log all transformations.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame with numeric conversions applied where possible.
        """

        def safe_convert(value):
            """Attempts to convert values to numeric, handling boolean strings, and logs changes."""
            original_value = value  # Store the original value

            if isinstance(value, str):
                value = value.strip().lower()
                if value == "true":
                    new_value = 1
                elif value == "false":
                    new_value = 0
                else:
                    try:
                        new_value = pd.to_numeric(value, errors="raise")
                    except (ValueError, TypeError):
                        new_value = value  # Keep as original if conversion fails
            else:
                try:
                    new_value = pd.to_numeric(value, errors="raise")
                except (ValueError, TypeError):
                    new_value = value  # Keep as original if conversion fails

            # Log the conversion if the value changed
            if original_value != new_value:
                conversion_log.append({
                    "Column Name": current_column,
                    "Original Value": original_value,
                    "Converted Value": new_value,
                    "Conversion Type": f"{type(original_value).__name__} to {type(new_value).__name__}"
                })

            return new_value

        # Store conversion logs
        conversion_log = []

        # Apply `safe_convert` column-wise
        for current_column in self.working_df.select_dtypes(include=["object"]).columns:
            self.working_df[current_column] = self.working_df[current_column].map(safe_convert)

        # Create a summary DataFrame for logging changes
        self.to_numeric_log_df = pd.DataFrame(conversion_log)

        return self

    # Remove features from the DataFrame
    def __drop_features(self) -> "DataPreprocessor":
        """
        Remove features from the DataFrame with a missing percentage higher than the given threshold.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - The cleaned DataFrame with selected features.
                - A DataFrame listing omitted features and their missing percentages.
        """

        # Calculate missing percentage for each column
        missing_percentage = (self.working_df.isnull().sum() / len(self.working_df)) * 100

        # Identify variables to omit (missing percentage > threshold)
        variables_to_omit = missing_percentage[missing_percentage > self.missing_threshold]

        # Create a DataFrame for omitted variables
        self.dropped_features_log_df = pd.DataFrame({
            "Variable": variables_to_omit.index,
            "Missing Percentage": variables_to_omit.values.round(2)
        })

        # Identify variables to keep
        variables_to_keep = missing_percentage[missing_percentage <= self.missing_threshold].index.tolist()

        # Filter the dataset without modifying self.working_df
        self.working_df = self.working_df[variables_to_keep]

        return self

    # Removes records from the DataFrame
    def __drop_records(self) -> "DataPreprocessor":
        """
        Remove records from the DataFrame where the percentage of missing values 
        exceeds the specified threshold.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - The cleaned DataFrame with records removed.
                - A DataFrame listing removed records with excessive missing values.
        """

        # Calculate the threshold for missing values based on the given percentage
        threshold = (self.missing_threshold / 100) * self.working_df.shape[1]

        # Identify records with missing values exceeding the threshold
        self.dropped_records_log_df = self.working_df[self.working_df.isnull().sum(axis=1) > threshold]

        # Create a cleaned DataFrame without modifying self.working_df
        self.working_df = self.working_df.drop(index=self.dropped_records_log_df.index)

        return self

    # Impute missing values
    def __impute_features(self) -> "DataPreprocessor":
        """
        Impute missing values in numeric columns of the DataFrame.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - The updated DataFrame after imputation.
                - A DataFrame with imputation details for each column.
        """
        imputation_details = []

        for col in self.non_categorical_features:
            # Replace invalid values with NaN explicitly
            self.working_df[col] = self.working_df[col].replace([np.inf, -np.inf], np.nan)

            # Skip if column has no missing values
            if self.working_df[col].isnull().sum() == 0:
                imputation_details.append({
                    "Variable": col,
                    "Imputation Method": "None (No Missing Values)",
                    "Significant Difference": 0,
                    "Percentage Difference": 0.00
                })
                continue

            # Check if the column is binary (only 2 unique values)
            unique_values = self.working_df[col].dropna().unique()
            if len(unique_values) == 2:
                imputation_method = "Median"
                imputer = SimpleImputer(strategy="median")
            else:
                # Calculate mean and median
                col_mean = self.working_df[col].mean()
                col_median = self.working_df[col].median()

                # Calculate percentage difference
                percentage_diff = abs(col_mean - col_median) / max(abs(col_mean), abs(col_median)) * 100
                significant_diff = int(percentage_diff > 10)  # Binary: 1 if significant, 0 otherwise

                # Choose strategy based on significant difference
                if significant_diff:
                    imputation_method = "Median"
                    imputer = SimpleImputer(strategy="median")
                else:
                    imputation_method = "Mean"
                    imputer = SimpleImputer(strategy="mean")

            # Apply the imputer
            self.working_df[[col]] = imputer.fit_transform(self.working_df[[col]])

            # Append details to the list
            imputation_details.append({
                "Variable": col,
                "Imputation Method": imputation_method,
                "Significant Difference": significant_diff if "significant_diff" in locals() else 0,
                "Percentage Difference": round(percentage_diff, 2) if "percentage_diff" in locals() else 0.00
            })

        # Handle missing values for categorical columns using the most frequent value
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        for col in self.categorical_features:
            # Apply imputation only if the column has missing values
            if self.working_df[col].isnull().sum() > 0:
                self.working_df[[col]] = categorical_imputer.fit_transform(self.working_df[[col]])

                # Log imputation details
                imputation_details.append({
                    "Variable": col,
                    "Imputation Method": "Most Frequent",
                    "Significant Difference": "N/A",
                    "Percentage Difference": "N/A"
                })

        # Update the imputation log DataFrame with categorical imputation details
        self.imputation_log_df = pd.DataFrame(imputation_details)

        return self.working_df
  
    # Splits the DataFrame into features and target variable
    def __feature_target_split(self) -> "DataPreprocessor":
        """
        Splits the DataFrame into features and target variable.

        Args:
            None

        Returns:
            Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]: 
                - A DataFrame containing features.
                - A Series if the target is a single column, otherwise a DataFrame.
        """

        # Extract features and target without modifying self.working_df
        self.features_df = self.working_df.drop(columns=self.target_variable)
        self.target_df = self.working_df[self.target_variable]

        # Convert to Series if only one target column
        if len(self.target_variable) == 1:
            self.target_df = self.target_df.iloc[:, 0]  # Convert DataFrame to Series

        return self
    
    # Encodes categorical columns in the DataFrame
    def __encode(self) -> "DataPreprocessor":
        """
        Encodes categorical columns in the DataFrame using either OrdinalEncoder for ordinal columns 
        or OneHotEncoder for nominal columns.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame with encoded categorical columns.
        """

        # Store encoding logs
        encoding_logs = []

        # Initialize OrdinalEncoder for ordinal columns with specified order
        if self.ordinal_features:
            ordinal_encoder = OrdinalEncoder(categories=self.ordinal_categories) if self.ordinal_categories else OrdinalEncoder()

            for col in self.ordinal_features:
                original_values = self.features_df[col].unique()  # Store original unique values
                self.features_df[col] = ordinal_encoder.fit_transform(self.features_df[[col]].astype(str))  # FIX: Use 2D input
                encoded_values = self.features_df[col].unique()  # Store new unique values

                # Log the transformation
                encoding_logs.append({
                    "Column Name": col,
                    "Original Unique Values": list(original_values),
                    "Encoding Method": "Ordinal",
                    "Encoded Unique Values": list(encoded_values)
                })

        # Encode nominal columns
        if self.use_one_hot_encoding:
            # Exclude already numeric columns from one-hot encoding
            nominal_features_to_encode = [col for col in self.nominal_features if not pd.api.types.is_numeric_dtype(self.features_df[col])]

            if nominal_features_to_encode:
                one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")
                encoded_nominal_features = one_hot_encoder.fit_transform(self.features_df[nominal_features_to_encode].astype(str))

                encoded_nominal_dataframe = pd.DataFrame(
                    encoded_nominal_features,
                    columns=one_hot_encoder.get_feature_names_out(nominal_features_to_encode),
                    index=self.features_df.index
                )

                # Log each one-hot encoded feature
                for col in nominal_features_to_encode:
                    encoding_logs.append({
                        "Column Name": col,
                        "Original Unique Values": list(self.features_df[col].unique()),
                        "Encoding Method": "One-Hot",
                        "Encoded Unique Values": list(encoded_nominal_dataframe.columns)
                    })

                self.features_df.drop(columns=nominal_features_to_encode, inplace=True)
                self.features_df = pd.concat([self.features_df, encoded_nominal_dataframe], axis=1)
        else:
            ordinal_encoder_nominal = OrdinalEncoder()
            for col in self.nominal_features:
                original_values = self.features_df[col].unique()
                self.features_df[col] = ordinal_encoder_nominal.fit_transform(self.features_df[[col]].astype(str))  # FIX: Use 2D input
                encoded_values = self.features_df[col].unique()

                # Log the transformation
                encoding_logs.append({
                    "Column Name": col,
                    "Original Unique Values": list(original_values),
                    "Encoding Method": "Ordinal (Nominal)",
                    "Encoded Unique Values": list(encoded_values)
                })

        # Store encoding logs in self.encode_log_df
        self.encode_log_df = pd.DataFrame(encoding_logs)

        return self

    # Apply transformations to numeric columns
    def __transform(self) -> "DataPreprocessor":
        """
        Apply transformations to numeric columns in the DataFrame based on skewness and kurtosis.

        - Log transformation for right-skewed data.
        - Reflect & log transformation for left-skewed data.
        - Box-Cox transformation for heavy-tailed distributions.
        - Yeo-Johnson transformation for light-tailed distributions.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - The transformed DataFrame.
                - A DataFrame with transformation logs.
        """

        # Exclude target variable from transformation
        columns = [col for col in self.non_categorical_features if col not in self.target_variable]

        # List to store transformation logs
        transformation_logs = []

        for column in columns:
            # Compute skewness and kurtosis
            skewness = self.features_df[column].skew()
            kurtosis = self.features_df[column].kurt()
            action = "None"  # Default action

            # Handle Right Skew (Positive Skew)
            if skewness > 1:
                action = "Log Transformation"
                self.features_df[column] = np.log1p(self.features_df[column])

            # Handle Left Skew (Negative Skew)
            elif skewness < -1:
                action = "Reflect and Log Transformation"
                self.features_df[column] = np.log1p(self.features_df[column].max() - self.features_df[column])

            # Handle High Kurtosis (Heavy Tails)
            if kurtosis > 3:
                try:
                    action = "Box-Cox Transformation"
                    self.features_df[column], _ = boxcox(self.features_df[column].clip(lower=1))
                except ValueError:
                    action = "Box-Cox Failed, Applied Yeo-Johnson"
                    transformer = PowerTransformer(method="yeo-johnson")
                    self.features_df[column] = transformer.fit_transform(self.features_df[[column]])

            # Handle Low Kurtosis (Light Tails) if no other transformation was applied
            elif kurtosis < 3 and action == "None":
                action = "Yeo-Johnson Transformation"
                transformer = PowerTransformer(method="yeo-johnson")
                self.features_df[column] = transformer.fit_transform(self.features_df[[column]])

            # Compute skewness and kurtosis after transformation
            skewness_after = self.features_df[column].skew()
            kurtosis_after = self.features_df[column].kurt()

            # Append the log entry
            transformation_logs.append({
                "Column Name": column,
                "Skewness Before Transformation": skewness,
                "Kurtosis Before Transformation": kurtosis,
                "Action Taken": action,
                "Skewness After Transformation": skewness_after,
                "Kurtosis After Transformation": kurtosis_after
            })

        # Create a DataFrame for transformation logs
        self.transformation_log_df = pd.DataFrame(transformation_logs)

        return self

    # Scales numeric columns of the input DataFrame
    def __scale(self, method: str = "standard") -> "DataPreprocessor":
            """
            Scales numeric columns of the input DataFrame, excluding binary columns, and handles NaN/inf values.

            Args:
                method (str, optional): Scaling method, either 'standard' (default) for StandardScaler 
                                        or 'minmax' for MinMaxScaler.

            Returns:
                DataPreprocessor: The instance of the class with scaled features_df.
            """
            # Select numeric columns only
            numeric_cols = self.features_df.select_dtypes(include=["number"]).columns
            
            # Exclude binary columns (those with only two unique values)
            non_binary_cols = [col for col in numeric_cols if self.features_df[col].nunique() > 2]
            
            # Choose scaler based on the method
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid method. Use 'standard' or 'minmax'.")
            
            # Store scaling logs
            scaling_logs = []
            
            for col in non_binary_cols:
                # Handle infinite values by replacing them with NaN
                self.features_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Handle NaN values by imputing with median
                if self.features_df[col].isna().sum() > 0:
                    self.features_df[col].fillna(self.features_df[col].median(), inplace=True)
                
                # Log original min and max values
                original_min = self.features_df[col].min()
                original_max = self.features_df[col].max()
                
                # Apply scaling
                self.features_df[col] = scaler.fit_transform(self.features_df[[col]])
                
                # Log scaled min and max values
                scaled_min = self.features_df[col].min()
                scaled_max = self.features_df[col].max()
                
                # Store log
                scaling_logs.append({
                    "Column Name": col,
                    "Scaling Method": method.capitalize(),
                    "Original Min": original_min,
                    "Original Max": original_max,
                    "Scaled Min": scaled_min,
                    "Scaled Max": scaled_max
                })
            
            # Store scaling logs in self.scale_log_df
            self.scale_log_df = pd.DataFrame(scaling_logs)
            
            return self

    # Performs Train Test Splits
    def __split_dataframe(self) -> "DataPreprocessor":
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
        return self

    # Function for oversampling
    def __oversample_data(self) -> "DataPreprocessor":
        """
        Performs random oversampling to balance the dataset by increasing the number of instances in the minority class.

        Args:
            None (operates on instance attributes `self.X_train` and `self.y_train`).

        Returns:
            DataPreprocessor: The updated instance with an oversampled training dataset.
        """

        # Perform random oversampling
        oversampler = RandomOverSampler(random_state=55002)
        self.X_train, self.y_train = oversampler.fit_resample(self.X_train, self.y_train)
        return self