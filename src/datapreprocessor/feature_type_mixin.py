from typing import Tuple

class FeatureTypeMixin:
     # Identifies categorical and non-categorical columns in the DataFrame.
    def _feature_type(self) -> Tuple[list, list]:
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
