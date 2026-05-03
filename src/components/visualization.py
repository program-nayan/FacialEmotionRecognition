import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from sklearn.metrics import confusion_matrix
from src.logger import logging

class Visualizer:
    @staticmethod
    def plot_metrics(results, output_path="artifacts/training_metrics.png"):
        """
        Plots training and test loss and accuracy from the results dictionary.
        """
        try:
            train_loss = results["train_loss"]
            train_acc = results["train_acc"]
            test_loss = results["test_loss"]
            test_acc = results["test_acc"]
            
            epochs = range(1, len(train_loss) + 1)

            plt.figure(figsize=(15, 6))

            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
            plt.plot(epochs, test_loss, 'ro-', label='Test Loss')
            plt.title('Training and Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Plot Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_acc, 'bo-', label='Training Acc')
            plt.plot(epochs, test_acc, 'ro-', label='Test Acc')
            plt.title('Training and Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Metrics plot saved at {output_path}")
        except Exception as e:
            logging.error(f"Error plotting metrics: {e}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, output_path="artifacts/confusion_matrix.png"):
        """
        Plots the confusion matrix.
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Confusion matrix plot saved at {output_path}")
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
