from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_regression(model, X_test, y_test):  
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'Root Mean Squared Error: {rmse:.2f}')
    return rmse, y_pred
