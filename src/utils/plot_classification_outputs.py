
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_classification_outputs(y_test, y_pred):
    print("Plotting Confusion Matrix (Classification)")

    y_pred_classes = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Red Hot)')
    plt.tight_layout()
    plt.show()
