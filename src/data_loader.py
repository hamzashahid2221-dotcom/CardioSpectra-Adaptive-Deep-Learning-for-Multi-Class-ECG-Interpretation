import os
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_image_paths(base_path, class_folders):
    all_images = []
    all_labels = []

    for idx, folder in enumerate(class_folders):
        path = os.path.join(base_path, folder)
        data_dir = pathlib.Path(path)
        images = list(data_dir.glob("*"))

        all_images.extend([str(img) for img in images])
        all_labels.extend([idx] * len(images))

    return np.array(all_images), np.array(all_labels)


def split_data(images, labels, test_size=0.1):
    return train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )


def create_dataset(image_paths, labels, preprocess_fn, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
