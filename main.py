import tensorflow as tf

from dataPreprocessing import make_train_gen
from dataPreprocessing import make_validate_gen
from model import make_model
from inference import inference


epochs = 1
batch_size = 32
image_size = (180, 180)

train_gen = make_train_gen()
validate_gen = make_validate_gen()

model = make_model(input_shape=image_size + (3,), num_classes=2)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath="model_weights.h5", save_best_only=True, save_weights_only=True)
]

model.load_weights('first_try.h5')

model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=2000 // batch_size,
    validation_data=validate_gen,
    validation_steps=800 // batch_size
)

model.save_weights('first_try.h5')

inference(image_size=image_size, model=model)
