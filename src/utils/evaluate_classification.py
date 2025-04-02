from sklearn.metrics import accuracy_score

def evaluate_classification(model, X_test, y_test):    
    y_pred = model.predict(X_test).flatten()
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2%}')
    return accuracy, y_pred
