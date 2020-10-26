from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (180, 180)
batch_size = 32
directory = '/home/hellolt/miniconda3/Datasets/CatsAndDogs/PetImages'


def make_train_gen():
    train_data_gen = ImageDataGenerator(rotation_range=5, horizontal_flip=True, rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_data_gen.flow_from_directory(directory=directory,
                                                   target_size=image_size,
                                                   class_mode='binary',
                                                   batch_size=batch_size,
                                                   seed=1024,
                                                   subset='training')
    return train_gen


def make_validate_gen():
    validate_data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    validate_gen = validate_data_gen.flow_from_directory(directory=directory,
                                                         target_size=image_size,
                                                         class_mode='binary',
                                                         batch_size=batch_size,
                                                         seed=1024,
                                                         subset='validation')
    return validate_gen
