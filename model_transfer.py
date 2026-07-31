import keras
import tensorflow as tf
from keras import layers, regularizers, Model, Input
from config import IMG_SIZE, N_CHANNELS, NUM_CLASSES, TRANSFER_LR, FINETUNE_LR, FOCAL_GAMMA

L2 = 1e-4
FINETUNE_LAYERS = 20
BASE_INPUT = 224


@keras.saving.register_keras_serializable(package="TransferEmotionCNN")
class ZScoreToPixels(layers.Layer):

    def call(self, x):
        x = tf.clip_by_value(x, -3.0, 3.0)
        return (x + 3.0) / 6.0 * 255.0

    def get_config(self):
        return super().get_config()


class TransferEmotionModel:

    def build_model(self) -> Model:
        inputs = Input(shape=(*IMG_SIZE, N_CHANNELS), name="mel_spectrogram")

        x = ZScoreToPixels(name="z_to_pixels")(inputs)
        x = layers.Resizing(BASE_INPUT, BASE_INPUT, name="upsample")(x)

        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(BASE_INPUT, BASE_INPUT, 3),
        )
        base.trainable = False

        x = base(x, training=False)

        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="head_bn")(x)
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2),
            name="dense_256",
        )(x)
        x = layers.Dropout(0.5, name="dropout")(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

        model = Model(inputs=inputs, outputs=outputs, name="TransferEmotionCNN")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TRANSFER_LR,
                                               clipnorm=1.0),
            loss=_focal_loss,
            metrics=["accuracy"],
        )
        return model

    def prepare_finetuning(self, model: Model) -> Model:
        base = model.get_layer("efficientnetb0")
        base.trainable = True

        for layer in base.layers[:-FINETUNE_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR,
                                               clipnorm=1.0),
            loss=_focal_loss,
            metrics=["accuracy"],
        )
        return model


@keras.saving.register_keras_serializable(package="TransferEmotionCNN")
def _focal_loss(y_true, y_pred):
    n_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
    p_t       = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
    p_t       = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(-tf.pow(1.0 - p_t, FOCAL_GAMMA) * tf.math.log(p_t))


if __name__ == "__main__":
    m = TransferEmotionModel().build_model()
    m.summary()
    print("model_transfer.py loaded OK")
