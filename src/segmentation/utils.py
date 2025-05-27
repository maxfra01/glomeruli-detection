import matplotlib.pyplot as plt
import tensorflow as tf

def plot_random_samples(dataset: tf.data.Dataset, num_samples: int = 5):
    """
    Plot random samples from the dataset.
    
    Args:
        dataset: tf.data.Dataset object.
        num_samples: Number of random samples to plot.
    """
    shuffled_dataset = dataset # If needed, shuffle the dataset 
    
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
    Plot training and validation metrics: loss, precision, recall, and IoU.

    Args:
        history: History object returned by model.fit().
        save_path (str): Optional path to save the plot image.
    """
    metrics = ['loss', 'precision', 'recall', 'mean_io_u']
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.plot(history.history.get(metric, []), label=f"Train {metric.capitalize()}")
        ax.plot(history.history.get(f"val_{metric}", []), label=f"Val {metric.capitalize()}")
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.suptitle("Training History", fontsize=16)

    if save_path:
        plt.savefig(save_path)
    plt.close()

        