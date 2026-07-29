"""
model_transfer.py — Transfer learning model for RAVDESS emotion classification.

Architecture
------------
Input (128, 128, 3)    — mel + delta + delta-delta as the three "RGB" channels
  → ZScoreToPixels     — map Z-scores to [0, 255] pixel range
  → EfficientNetB0     — ImageNet pretrained feature extractor (include_top=False)
  → GlobalAveragePooling2D
  → BatchNormalization
  → Dense(256, ReLU, L2)
  → Dropout(0.5)
  → Dense(8, Softmax)

Two-phase training
------------------
Phase 1 (frozen base):
    EfficientNetB0 base is frozen. Only the new head is trained.
    Faster convergence, avoids destroying pretrained features.
    LR = TRANSFER_LR (1e-3), PHASE1_EPOCHS epochs.

Phase 2 (fine-tuning):
    Unfreeze the top FINETUNE_LAYERS layers of EfficientNetB0.
    Lower LR = FINETUNE_LR (1e-4) to gently adjust pretrained weights.
    PHASE2_EPOCHS additional epochs.

Why EfficientNetB0?
-------------------
ImageNet features (edges, textures, shapes) transfer well to spectrograms —
both are 2D images with local structure. Fine-tuning on top of strong pretrained
features bypasses the data-scarcity problem that limits training from scratch.
"""

import keras
import tensorflow as tf
from keras import layers, regularizers, Model, Input
from config import IMG_SIZE, N_CHANNELS, NUM_CLASSES, TRANSFER_LR, FINETUNE_LR, FOCAL_GAMMA

L2 = 1e-4
FINETUNE_LAYERS = 60   # number of EfficientNetB0 layers to unfreeze in Phase 2


@keras.saving.register_keras_serializable(package="TransferEmotionCNN")
class ZScoreToPixels(layers.Layer):
    """
    Rescale Z-scored spectrogram values to [0, 255] pixel range.

    Why this is necessary
    ---------------------
    Our preprocessing pipeline produces Z-scored spectrograms with mean≈0
    and values roughly in [-3, 3]. EfficientNetB0 was pretrained on ImageNet
    where pixel values are in [0, 255]. Without rescaling, the pretrained
    conv filters see completely different input magnitudes, making the
    learned ImageNet features useless in Phase 1.

    Mapping: clip to [-3, 3], then linearly scale to [0, 255].
      pixel = (z + 3) / 6 * 255
    After this layer, EfficientNet's internal preprocessing (÷ 255 + BN)
    brings values back to the [0, 1] range it was trained on.
    """

    def call(self, x):
        x = tf.clip_by_value(x, -3.0, 3.0)
        return (x + 3.0) / 6.0 * 255.0

    def get_config(self):
        return super().get_config()


class TransferEmotionModel:
    """
    Builds the two-phase transfer learning model.

    Methods
    -------
    build_model()
        Returns compiled Phase 1 model (frozen base).
    prepare_finetuning(model)
        Unfreezes top FINETUNE_LAYERS, recompiles with lower LR.
        Returns the same model object (modified in-place).
    """

    def build_model(self) -> Model:
        """Build and compile Phase 1 model (EfficientNetB0 base frozen)."""
        inputs = Input(shape=(*IMG_SIZE, N_CHANNELS), name="mel_spectrogram")

        # Z-score [-3, 3] → pixel [0, 255] so EfficientNet sees expected input range
        x = ZScoreToPixels(name="z_to_pixels")(inputs)

        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMG_SIZE, 3),
        )
        base.trainable = False   # Phase 1: freeze all base layers

        x = base(x, training=False)   # training=False keeps BN in inference mode

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
        """
        Switch from Phase 1 to Phase 2.

        Unfreezes the last FINETUNE_LAYERS layers of the EfficientNetB0 base,
        then recompiles with FINETUNE_LR (10× smaller than Phase 1 LR).
        All earlier base layers stay frozen to preserve low-level ImageNet features.
        """
        base = model.get_layer("efficientnetb0")
        base.trainable = True

        # Freeze all layers except the last FINETUNE_LAYERS
        for layer in base.layers[:-FINETUNE_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR,
                                               clipnorm=1.0),
            loss=_focal_loss,
            metrics=["accuracy"],
        )
        return model


# ── Focal loss (same formula as model.py, separate registration namespace) ──
@keras.saving.register_keras_serializable(package="TransferEmotionCNN")
def _focal_loss(y_true, y_pred):
    """
    Focal loss (Lin et al., 2017): FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Registered under TransferEmotionCNN namespace to avoid collision with
    the EmotionCNN namespace used in model.py.
    """
    n_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
    p_t       = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
    p_t       = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(-tf.pow(1.0 - p_t, FOCAL_GAMMA) * tf.math.log(p_t))


if __name__ == "__main__":
    m = TransferEmotionModel().build_model()
    m.summary()
    print("model_transfer.py loaded OK")
