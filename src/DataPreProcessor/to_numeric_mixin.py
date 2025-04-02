import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

class ToNumericMixin:
    def _to_numeric(self) -> "DataPreProcessor":
        self.conversion_log = []

        for self.current_column in self.working_df.select_dtypes(include=["object"]).columns:
            self.working_df[self.current_column] = self.working_df[self.current_column].map(self.__safe_convert)

        self.to_numeric_log_df = pd.DataFrame(self.conversion_log)

    def __safe_convert(self, value):
        original_value = value

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
                    new_value = value
        else:
            try:
                new_value = pd.to_numeric(value, errors="raise")
            except (ValueError, TypeError):
                new_value = value

        if original_value != new_value:
            self.conversion_log.append({
                "Column Name": self.current_column,
                "Original Value": original_value,
                "Converted Value": new_value,
                "Conversion Type": f"{type(original_value).__name__} to {type(new_value).__name__}"
            })

        return new_value
