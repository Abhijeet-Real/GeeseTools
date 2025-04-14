# Data manipulation and numerical operations
import pandas as pd  

# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

# Feature scaling and encoding techniques
from sklearn.preprocessing import (
    OneHotEncoder,  # Converts categorical variables into a binary matrix
    OrdinalEncoder,  # Encodes ordinal categorical features with meaningful order
)

class EncodeMixin:
    # Encodes categorical columns in the DataFrame
    def _encode(self)-> "DataPreProcessor":
        self.encoding_logs = []

        if self.ordinal_features:
            self.__encode_ordinal_features()

        if self.one_hot_encoding:
            self.__encode_nominal_features_one_hot()
        else:
            self.__encode_nominal_features()

        self.encode_log_df = pd.DataFrame(self.encoding_logs)

    def __encode_ordinal_features(self)-> "DataPreProcessor":
        ordinal_encoder = OrdinalEncoder(categories=self.ordinal_categories) if self.ordinal_categories else OrdinalEncoder()

        for col in self.ordinal_features:
            original_values = self.features_df[col].unique()
            self.features_df[col] = ordinal_encoder.fit_transform(self.features_df[[col]].astype(str))
            encoded_values = self.features_df[col].unique()
            self.log_encoding(col, original_values, "Ordinal", encoded_values)

    def __encode_nominal_features_one_hot(self)-> "DataPreProcessor":
        nominal_features_to_encode = [col for col in self.nominal_features if not pd.api.types.is_numeric_dtype(self.features_df[col])]

        if not nominal_features_to_encode:
            return

        one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_nominal_features = one_hot_encoder.fit_transform(self.features_df[nominal_features_to_encode].astype(str))

        encoded_df = pd.DataFrame(
            encoded_nominal_features,
            columns=one_hot_encoder.get_feature_names_out(nominal_features_to_encode),
            index=self.features_df.index
        )

        for col in nominal_features_to_encode:
            self.log_encoding(col, self.features_df[col].unique(), "One-Hot", encoded_df.columns.tolist())

        self.features_df.drop(columns=nominal_features_to_encode, inplace=True)
        self.features_df = pd.concat([self.features_df, encoded_df], axis=1)

    def __encode_nominal_features(self)-> "DataPreProcessor":
        ordinal_encoder = OrdinalEncoder()

        for col in self.nominal_features:
            original_values = self.features_df[col].unique()
            self.features_df[col] = ordinal_encoder.fit_transform(self.features_df[[col]].astype(str))
            encoded_values = self.features_df[col].unique()
            self.log_encoding(col, original_values, "Nominal", encoded_values)

    def log_encoding(self, column_name, original_values, method, encoded_values)-> "DataPreProcessor":
        self.encoding_logs.append({
            "Column Name": column_name,
            "Original Unique Values": list(original_values),
            "Encoding Method": method,
            "Encoded Unique Values": list(encoded_values)
        })
