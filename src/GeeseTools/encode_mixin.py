# Data manipulation and numerical operations
import pandas as pd  # For handling structured data (DataFrames)import pandas as pd

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeeseTools import GeeseTools

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
    OneHotEncoder,  # Converts categorical variables into a binary matrix
    OrdinalEncoder,  # Encodes ordinal categorical features with meaningful order
)

class EncodeMixin:
    # Encodes categorical columns in the DataFrame
    def _encode(self) -> "GeeseTools":
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