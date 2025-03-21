import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dpp.DPP import DataPreprocessor

class ToNumericMixin:
    # Converts all columns in the DataFrame into numeric types where possible.
    def _to_numeric(self) -> "DataPreprocessor":
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