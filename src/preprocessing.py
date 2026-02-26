import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def preprocess(image_path, label, num_classes=3):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)

    label = tf.one_hot(label, depth=num_classes)

    return img, label
