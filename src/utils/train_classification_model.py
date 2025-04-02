from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore

def train_classification_model(X_train, y_train, epochs=20, batch_size=16):
    # Define the ANN model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,  # Already encoded
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model, "classification", history  # Return None for compatibility
