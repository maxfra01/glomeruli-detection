import tensorflow as tf

# List physical devices
gpus = tf.config.list_physical_devices('GPU')

# Print result
if gpus:
    print(f"TensorFlow detected {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("No GPU detected by TensorFlow.")
