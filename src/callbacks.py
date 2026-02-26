import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score


class AdaptiveAlphaGammaCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_data, loss_fn, num_classes):
        super().__init__()
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for x_batch, y_batch in self.val_data:
            pred = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))

        recalls = recall_score(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average=None
        )

        new_gamma = 2 + (1 - recalls) * 2
        new_alpha = 1 + (1 - recalls) * 2

        self.loss_fn.gamma.assign(new_gamma)
        self.loss_fn.alpha.assign(new_alpha)

        print(f"Updated gamma: {new_gamma}")
        print(f"Updated alpha: {new_alpha}")
