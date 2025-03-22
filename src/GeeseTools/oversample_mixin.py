# Type hinting for better code readability and function definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeeseTools import GeeseTools

# Sampling techniques for imbalanced datasets
from imblearn.over_sampling import RandomOverSampler  # Randomly oversamples the minority class

class OverSampleMixin:
    # Function for oversampling
    def _oversample_data(self) -> "GeeseTools":
        """
        Performs random oversampling to balance the dataset by increasing the number of instances in the minority class.

        Args:
            None (operates on instance attributes `self.X_train` and `self.y_train`).

        Returns:
            GeeseTools: The updated instance with an oversampled training dataset.
        """

        # Perform random oversampling
        oversampler = RandomOverSampler(random_state=55002)
        self.X_train, self.y_train = oversampler.fit_resample(self.X_train, self.y_train)