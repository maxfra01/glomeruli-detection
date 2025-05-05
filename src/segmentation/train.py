from model import build_vgg19_segmentation
from dataset import get_dataset
import tensorflow as tf
import os
from utils import plot_random_samples, plot_training_history

tf.random.set_seed(42)

DATASET_DIR = "./data/"

# Hyperparameters
IMG_HEIGHT = 384
IMG_WIDTH = 384
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9
PATIENCE = 5 # Early stopping patience

if __name__ == "__main__":
    
    dataset = get_dataset(DATASET_DIR, crop_size=(IMG_HEIGHT, IMG_WIDTH))
    
    plot_random_samples(dataset, num_samples=5) # Visualize random samples

    #TODO: add data augmentation

    total_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.64 * total_size)
    val_size = int(0.16 * total_size)

    train_ds = dataset.take(train_size).shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset.skip(train_size + val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print(f"Total samples: {total_size}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {total_size - train_size - val_size}")

    # Build model
    model = build_vgg19_segmentation(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM
    )

    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    model_specs = f"model_lr{LEARNING_RATE}_mm{MOMENTUM}_bs{BATCH_SIZE}" # Model specifications

    # Train model
    history = model.fit( 
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(f"snapshots/{model_specs}.keras", save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
        ]       
    )
    
    plot_training_history(history, 'plots/' + model_specs + '.png') # Save training history

    # Evaluate on test set
    test_metrics = model.evaluate(test_ds)
    print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))