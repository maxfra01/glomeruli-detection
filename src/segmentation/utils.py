import matplotlib.pyplot as plt
import tensorflow as tf

def plot_random_samples(dataset: tf.data.Dataset, num_samples: int = 5):
    """
    Plot random samples from the dataset.
    
    Args:
        dataset: tf.data.Dataset object.
        num_samples: Number of random samples to plot.
    """
    shuffled_dataset = dataset.shuffle(buffer_size=100) 
    
    fig, ax = plt.subplots(2, num_samples, figsize=(15, 5))
    
    for i, (image, mask) in enumerate(shuffled_dataset.take(num_samples)):
        ax[0, i].imshow(image)
        ax[0, i].axis("off")
        ax[1, i].imshow(mask, cmap="gray")
        ax[1, i].axis("off")
        
    plt.suptitle("Random Samples from Dataset")
    plt.tight_layout()
    plt.show()
    plt.close()
    
def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: History object returned by model.fit().
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax[0].plot(history.history["loss"], label="Train Loss")
    ax[0].plot(history.history["val_loss"], label="Validation Loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(history.history["accuracy"], label="Train Accuracy")
    ax[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    
    plt.suptitle("Training History")
    #plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
        