import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import load_image_paths, create_dataset
from src.preprocessing import preprocess


BASE_PATH = "ECG_DATA/test"

CLASS_FOLDERS = [
    "Normal Person ECG Images (284x12=3408)",
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)",
    "ECG Images of Myocardial Infarction Patients (240x12=2880)"
]

NUM_CLASSES = 3
BATCH_SIZE = 32


images, labels = load_image_paths(BASE_PATH, CLASS_FOLDERS)

test_ds = create_dataset(
    images,
    labels,
    lambda x, y: preprocess(x, y, NUM_CLASSES),
    BATCH_SIZE
)

model = tf.keras.models.load_model(
    "saved_models/fine_tune_best_model.keras",
    compile=False
)

y_pred = []
y_true = []

for imgs, lbls in test_ds:
    pred = model.predict(imgs)
    y_pred.extend(np.argmax(pred, axis=1))
    y_true.extend(np.argmax(lbls.numpy(), axis=1))

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
