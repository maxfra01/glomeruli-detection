import tensorflow as tf
from keras.applications import VGG19
from keras import layers, models
from keras.optimizers import Adam


def build_vgg19_segmentation(input_shape=(384, 384, 3), learning_rate=0.001, num_classes=2):
    """
    Builds a binary segmentation model using the VGG19 architecture as a backbone.
    It classifies each pixel in the input image into either glomerular or non-glomerular tissue (background).
    
    Args:
        input_shape (tuple): The input shape of the image (height, width, channels).
        num_classes (int): The number of output classes for segmentation.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum for the SGD optimizer.

    Returns:
        model: A Keras Model instance.
    """
    vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in vgg19_base.layers:
        layer.trainable = False

    # Encoder: Extract features from the VGG19 base
    x = vgg19_base.output
    
    # Decoder: Upsample the features to create segmentation output
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    # Final layer for segmentation 
    x = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    
    # Create the model
    model = models.Model(inputs=vgg19_base.input, outputs=x)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.MeanIoU(num_classes=num_classes),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ])
    
    return model

if __name__ == "__main__":
    model = build_vgg19_segmentation()
    model.summary()