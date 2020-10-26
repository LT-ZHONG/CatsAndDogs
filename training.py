from tensorflow import keras


def start_training(model, train_data, validate_data):
    epochs = 50

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_data, epochs=epochs, callbacks=callbacks, validation_data=validate_data,
    )
