from tensorflow import keras
from tensorflow.keras.applications import ResNet50


def build_resnet_model(num_classes=3, trainable=False):
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = trainable

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=base_model.input, outputs=output)

    return model, base_model
