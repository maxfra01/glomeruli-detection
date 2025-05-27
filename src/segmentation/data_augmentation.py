import tensorflow as tf

def augment(image, mask, crop_size=(384, 384)):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image)
        mask = tf.image.rot90(mask)
        
    # Color jittering
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # if tf.random.uniform(()) > 0.5:
    #     zoom_factor = tf.random.uniform((), minval=0.8, maxval=1.2)  # Zoom factor
    #     height = tf.cast(tf.shape(image)[0], tf.float32)
    #     width = tf.cast(tf.shape(image)[1], tf.float32)
    #     new_height = tf.cast(height * zoom_factor, tf.int32)
    #     new_width = tf.cast(width * zoom_factor, tf.int32)
        
    #     # Reshape image and mask
    #     image = tf.image.resize(image, [new_height, new_width])
    #     mask = tf.image.resize(mask, [new_height, new_width])

    #     # Random crop
    #     image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    #     mask = tf.image.random_crop(mask, size=[crop_size[0], crop_size[1], 1])    
    
    return image, mask