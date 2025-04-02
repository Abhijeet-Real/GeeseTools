from . import plot_regression_outputs as pro
from . import plot_classification_outputs as pco

def plot_model_outputs(history):
        pro.plot_regression_outputs(history)

def plot_model_outputs(y_test, y_pred):
        pco.plot_classification_outputs(y_test, y_pred)