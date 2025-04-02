import numpy as np

from . import train_classification_model as tcm
from . import train_regression_model as trm

def train_model(X_train, y_train, epochs=50, batch_size=16, class_threshold=10):
    # Convert y_train to array for consistency
    y_train = np.array(y_train)

    # Rule: if number of unique classes is small and dtype is object/string OR integer → classification
    unique_classes = np.unique(y_train)
    n_unique = len(unique_classes)
    dtype_kind = y_train.dtype.kind  # 'i' = int, 'f' = float, 'O' = object, 'U' = unicode (string)

    is_classification = (n_unique <= class_threshold) and (dtype_kind in ['i', 'O', 'U'])

    if is_classification:
        print("🔍 Classification problem detected.")
        return tcm.train_classification_model(X_train, y_train, epochs, batch_size)
    else:
        print("🔍 Regression problem detected.")
        return trm.train_regression_model(X_train, y_train, epochs, batch_size)
