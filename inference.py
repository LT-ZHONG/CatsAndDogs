import tensorflow as tf


def inference(image_size, model):
    file_path = '/home/hellolt/miniconda3/Datasets/CatsAndDogs/PetImages/Cat/6779.jpg'

    img = tf.keras.preprocessing.image.load_img(path=file_path, target_size=image_size)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    score = predictions[0]
    print(score)
