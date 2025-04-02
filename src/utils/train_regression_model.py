from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore

def train_regression_model(X_train, y_train, epochs=50, batch_size=16):
    # Define deeper ANN model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Linear output for regression
    ])

    # Compile the model
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mse']
    )

    # Train the model with validation split
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model, "regression", history  # Returning None instead of scaler
