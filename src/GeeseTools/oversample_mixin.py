# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataPreProcessor import DataPreProcessor

# Sampling techniques for imbalanced datasets
from imblearn.over_sampling import RandomOverSampler  # Randomly oversamples the minority class

class OverSampleMixin:
    # Function for oversampling
    def _oversample_data(self) -> "DataPreProcessor":
        """
        Performs random oversampling to balance the dataset by increasing the number of instances in the minority class.
        """

        # Perform random oversampling
        oversampler = RandomOverSampler(random_state=55002)
        self.X_train, self.y_train = oversampler.fit_resample(self.X_train, self.y_train)