import tensorflow as tf


def categorical_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def loss(y_true, y_pred):
        cross_entropy = -y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1))
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = cross_entropy * weight
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return loss


class AdaptiveCategoricalFocalLoss(tf.keras.losses.Loss):

    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = tf.Variable(tf.convert_to_tensor(alpha, dtype=tf.float32))
        self.gamma = tf.Variable(tf.convert_to_tensor(gamma, dtype=tf.float32))

    def call(self, y_true, y_pred):
        cross_entropy = -y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1))
        weight = self.alpha * tf.math.pow(1 - y_pred, self.gamma)
        loss = cross_entropy * weight
        return tf.reduce_sum(loss)
