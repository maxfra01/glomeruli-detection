from model import build_segnet_vgg19_binary
from dataset import get_dataset
import tensorflow as tf
import os
from utils import plot_random_samples, plot_training_history
from data_augmentation import augment
from manifold_analysis import analyze_layer_manifolds

tf.random.set_seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DATASET_DIR = "./data/"

# Hyperparameters
IMG_HEIGHT = 384
IMG_WIDTH = 384
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10 # Early stopping patience

if __name__ == "__main__":
    
    dataset = get_dataset(DATASET_DIR, crop_size=(IMG_HEIGHT, IMG_WIDTH))
    

    total_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.33 * total_size)
    val_size = int(0.33 * total_size)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset.skip(train_size + val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
    for _ in range(3):
        train_ds_aug = train_ds.map(
            lambda x, y: (augment(x, y)), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.concatenate(train_ds_aug)
    
    train_ds = train_ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print(f"Total samples: {total_size}")
    print(f"Batch count")
    print(f"Train: {tf.data.experimental.cardinality(train_ds).numpy()}, Val: {tf.data.experimental.cardinality(val_ds).numpy()}, Test: {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Build model
    model = build_segnet_vgg19_binary(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        learning_rate=LEARNING_RATE,
    )

    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    model_specs = f"model_lr{LEARNING_RATE}_bs{BATCH_SIZE}" # Model specifications

    # Train model
    history = model.fit( 
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"snapshots/{model_specs}.keras",
                save_best_only=True,
                monitor="val_mean_io_u",  
                mode="max"   
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=PATIENCE,
                restore_best_weights=True,
                monitor="val_mean_io_u",
                mode="max"
            )
        ]       
    )

    
    plot_training_history(history, 'plots/' + model_specs + '.png') # Save training history

    # Evaluate on test set
    test_metrics = model.evaluate(test_ds)
    print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))