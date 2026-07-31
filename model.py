import tensorflow as tf
import keras
from keras import layers, regularizers, Model, Input
from config import IMG_SIZE, N_CHANNELS, NUM_CLASSES, LEARNING_RATE, FOCAL_GAMMA

L2 = 1e-4


@keras.saving.register_keras_serializable(package="EmotionCNN")
def focal_loss(y_true, y_pred):
    n_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
    p_t       = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
    p_t       = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(-tf.pow(1.0 - p_t, FOCAL_GAMMA) * tf.math.log(p_t))


class CNNBlock:

    def __init__(self, filters: int, dropout_rate: float = 0.0):
        self.filters      = filters
        self.dropout_rate = dropout_rate

    def build(self, inputs):
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(L2),
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        if self.dropout_rate > 0.0:
            x = layers.Dropout(self.dropout_rate)(x)
        return x


class EmotionCNN:

    def build_model(self) -> Model:
        inputs = Input(shape=(*IMG_SIZE, N_CHANNELS), name="mel_spectrogram")

        x = CNNBlock(32).build(inputs)
        x = CNNBlock(64).build(x)
        x = CNNBlock(128, dropout_rate=0.3).build(x)
        x = CNNBlock(256, dropout_rate=0.3).build(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2),
            name="dense_256",
        )(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

        model = Model(inputs=inputs, outputs=outputs, name="EmotionCNN")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,
                                               clipnorm=1.0),
            loss=focal_loss,
            metrics=["accuracy"],
        )
        return model

    def summary(self):
        model = self.build_model()
        model.summary()
        return model


if __name__ == "__main__":
    EmotionCNN().summary()
