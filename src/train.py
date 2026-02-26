import tensorflow as tf
from tensorflow import keras
from src.data_loader import load_image_paths, split_data, create_dataset
from src.preprocessing import preprocess
from src.models import build_resnet_model
from src.losses import categorical_focal_loss, AdaptiveCategoricalFocalLoss
from src.callbacks import AdaptiveAlphaGammaCallback


BASE_PATH = "ECG_DATA/train"

CLASS_FOLDERS = [
    "Normal Person ECG Images (284x12=3408)",
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)",
    "ECG Images of Myocardial Infarction Patients (240x12=2880)"
]

NUM_CLASSES = 3
BATCH_SIZE = 32


# Load Data
images, labels = load_image_paths(BASE_PATH, CLASS_FOLDERS)
train_imgs, val_imgs, train_lbls, val_lbls = split_data(images, labels)

train_ds = create_dataset(train_imgs, train_lbls,
                          lambda x, y: preprocess(x, y, NUM_CLASSES),
                          BATCH_SIZE)

val_ds = create_dataset(val_imgs, val_lbls,
                        lambda x, y: preprocess(x, y, NUM_CLASSES),
                        BATCH_SIZE)


# Phase 1 Training
model, base_model = build_resnet_model(NUM_CLASSES, trainable=False)

alpha = [0.98, 1.19, 0.87]
gamma = [2, 2, 2]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=categorical_focal_loss(alpha, gamma),
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("saved_models/best_model.keras")


# Phase 2 Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

loss_fn = AdaptiveCategoricalFocalLoss(alpha, gamma)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss_fn,
    metrics=["accuracy"]
)

adaptive_callback = AdaptiveAlphaGammaCallback(
    val_ds,
    loss_fn,
    NUM_CLASSES
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[adaptive_callback]
)

model.save("saved_models/fine_tune_best_model.keras")
