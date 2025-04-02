from . import evaluate_classification as ec
from . import evaluate_regression as er

def evaluate_model(model, X_test, y_test, task_type):

    if task_type == 'classification':
        return ec.evaluate_classification(model, X_test, y_test)
    elif task_type == 'regression':
        return er.evaluate_regression(model, X_test, y_test)
