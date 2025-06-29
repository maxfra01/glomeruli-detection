import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Alternative implementation of focal loss as a direct function.
    
    Args:
        y_true: Ground truth binary masks
        y_pred: Predicted probabilities
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    ce_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    # Calculate focal weight
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = tf.pow(1 - p_t, gamma)
    
    # Calculate alpha weight
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    # Combine all components
    focal_loss_val = alpha_t * focal_weight * ce_loss
    
    return tf.reduce_mean(focal_loss_val)

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def build_segnet_vgg19_binary(input_shape=(384, 384, 3), learning_rate=1e-3):
    """
    Builds a binary segmentation model based on the SegNet architecture using VGG19 as the encoder.

    Args:
        input_shape (tuple): Shape of the input images, e.g., (384, 384, 3)
        learning_rate (float): Learning rate for the Adam optimizer

    Returns:
        model (tf.keras.Model): Compiled Keras model for binary segmentation (glomeruli vs background)
    """

    # Encoder: VGG19 pretrained on ImageNet, without fully connected layers
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze encoder weights (you can unfreeze later for fine-tuning)
    for layer in base_model.layers:
        layer.trainable = False

    # Collect outputs from pooling layers to define encoder structure
    encoder_outputs = [
        base_model.get_layer("block1_pool").output,  # 192x192
        base_model.get_layer("block2_pool").output,  # 96x96
        base_model.get_layer("block3_pool").output,  # 48x48
        base_model.get_layer("block4_pool").output,  # 24x24
        base_model.get_layer("block5_pool").output   # 12x12
    ]

    # Start decoder from the last encoder output
    x = encoder_outputs[-1]

    # Decoder filters (reverse order of encoder depth)
    decoder_filters = [512, 512, 256, 128, 64]

    # Decoder: upsampling + convolution blocks (SegNet style, no skip connections)
    for filters in decoder_filters:
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    # Final 1x1 convolution for binary segmentation
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    # Define the full model
    model = models.Model(inputs=base_model.input, outputs=output)

    # Compile the model using Adam and binary crossentropy for 2-class segmentation
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryIoU(name='mean_io_u', threshold=0.5)
        ]
    )

    return model
