import matplotlib.pyplot as plt
import numpy as np

def plot_regression_outputs(history=None):
    # RMSE Plot
    if history and 'loss' in history.history and 'val_loss' in history.history:
        rmse = np.sqrt(history.history['loss'])
        val_rmse = np.sqrt(history.history['val_loss'])

        plt.figure(figsize=(8, 5))
        plt.plot(rmse, label='Training RMSE')
        plt.plot(val_rmse, label='Validation RMSE', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Root Mean Squared Error')
        plt.title('Training vs Validation RMSE')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()