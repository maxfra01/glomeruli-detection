import tensorflow as tf
import matplotlib.pyplot as plt
import os

from dataset import get_dataset


DATA_DIR = "./data/"
MODEL_PATH = "./snapshots/model_lr0.0001_bs64_wd0.0001_ar5.keras" 
INPUT_SHAPE = (384, 384, 3)
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
BATCH_SIZE = 16
SEED = 42

dataset = get_dataset(DATA_DIR, crop_size=(384, 384))

# --- Split test ---
tf.random.set_seed(SEED)
total_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int((1 - TEST_SPLIT - VAL_SPLIT) * total_size)
val_size = int(VAL_SPLIT * total_size)
test_size = total_size - train_size - val_size

test_ds = dataset.skip(train_size + val_size).take(test_size)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Test set size (batch): {test_size}")

# --- Load model ---
if os.path.exists(MODEL_PATH):
    print(f"CLoading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryIoU(name='mean_iou', threshold=0.5)
        ]
    )
else:
    print(f"Model non found in {MODEL_PATH}. Exit.")
    exit(1)
    
    
# --- Evaluate model on test set ---
print("\n--- Model evaluation on test set ---")
results = model.evaluate(test_ds, return_dict=True)

print("\nResults on test set:")
for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")


# --- Visualize predictions ---
output_dir = "test_preview"
os.makedirs(output_dir, exist_ok=True)

for images, true_masks in test_ds.take(1):
    preds = model.predict(images)
    preds = (preds > 0.5).astype(float)

    n = images.shape[0]
    for i in range(n):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_masks[i, :, :, 0], cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(preds[i, :, :, 0], cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
        plt.close()

