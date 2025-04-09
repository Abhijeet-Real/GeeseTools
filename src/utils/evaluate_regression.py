from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_regression(model, X_test, y_test):  
    y_pred = model.predict(X_test)  # No .flatten() here!
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)

    for i, rmse_val in enumerate(rmse):
        print(f'RMSE for output {i}: {rmse_val:.2f}')

    return rmse, y_pred
