import tensorflow as tf
import os

def _list_image_mask_pairs(directory, mask_suffix="_mask"):
    files = sorted([
        os.path.relpath(os.path.join(root, f), start=directory)
        for root, _, filenames in os.walk(directory)
        for f in filenames if f.endswith(".png")
    ])
    image_mask_pairs = []

    for f in files:
        if mask_suffix in f:
            continue  # skip masks
        base_name = f.replace(".png", "")
        mask_name = base_name + mask_suffix + ".png"
        image_path = os.path.join(directory, f)
        mask_path = os.path.join(directory, mask_name)
        if os.path.exists(mask_path):
            image_mask_pairs.append((image_path, mask_path))

    return image_mask_pairs

def _decode_and_preprocess(image_path, mask_path, crop_size=(384, 384)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0

    combined = tf.concat([image, mask], axis=-1) 
    combined = tf.image.resize_with_pad(combined, crop_size[0], crop_size[1])

    image = combined[:, :, :3]
    mask = combined[:, :, 3:]
    return image, mask

def get_dataset(directory, crop_size=(384, 384)):
    pairs = _list_image_mask_pairs(directory)
    image_paths, mask_paths = zip(*pairs)

    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))
    dataset = dataset.map(lambda img, msk: _decode_and_preprocess(img, msk, crop_size),
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Example usage
    print(os.getcwd())
    dataset = get_dataset("./data/")
    for images, masks in dataset.take(1):
        print("Images shape:", images.shape)
        print("Masks shape:", masks.shape)
    