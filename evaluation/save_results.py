import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from utils.tensorboard_logger import get_tensorboard_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
from pathlib import Path

logger = logging.getLogger(__name__)

class ResultsSaver:
    LABELS = ["W", "N1", "N2", "N3", "REM"]
    OVERALL_METRICS_FILENAME = 'overall_{}.csv'
    PER_CLASS_METRICS_FILENAME = 'perclass_{}.csv'

    def __init__(self, results_folder, experiment_num):
        """
        Initialize ResultsSaver with experiment-specific subfolder.
        
        Parameters:
        - results_folder (str): Base results directory
        - experiment_num (int): Experiment number for subfolder naming
        """
        self.experiment_num = experiment_num
        
        # Crea sottocartella per l'esperimento
        self.results_folder = os.path.join(results_folder, f"experiment_{experiment_num}")
        
        # Crea la directory se non esiste
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_folder}")

    def save_classification_results(self, predictions, true_labels, num_classes):
        tensorboard_logger = get_tensorboard_logger()
        os.makedirs(self.results_folder, exist_ok=True)
        logger.info(f"Results folder '{self.results_folder}' created or already exists.")

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1_score, support = precision_recall_fscore_support(
            true_labels, predictions, labels=range(num_classes)
        )
        macro_f1 = np.mean(f1_score)

        overall_metrics = {
            "Metric": ["Overall Accuracy", "Macro F1"],
            "Value": [accuracy, macro_f1]
        }
        overall_metrics_df = pd.DataFrame(overall_metrics)
        overall_metrics_df.to_csv(os.path.join(self.results_folder, self.OVERALL_METRICS_FILENAME.format(self.experiment_num)), index=False)
        logger.info(f"Overall metrics saved to {os.path.join(self.results_folder, self.OVERALL_METRICS_FILENAME.format(self.experiment_num))}")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1 Score: {macro_f1:.4f}")

        # Log overall metrics to TensorBoard
        tensorboard_logger.add_scalar('Overall Accuracy', accuracy, self.experiment_num)
        tensorboard_logger.add_scalar('Macro F1 Score', macro_f1, self.experiment_num)
        
        # Log per-class metrics to TensorBoard
        for i in range(num_classes):
            tensorboard_logger.add_scalar(f'Precision/Class {i}', precision[i], self.experiment_num)
            tensorboard_logger.add_scalar(f'Recall/Class {i}', recall[i], self.experiment_num)
            tensorboard_logger.add_scalar(f'F1 Score/Class {i}', f1_score[i], self.experiment_num)

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=range(num_classes))
        cm_percentage = confusion_matrix(true_labels, predictions, labels=range(num_classes), normalize='true') * 100
        
        # Save confusion matrix numbers
        cm_df = pd.DataFrame(cm, index=self.LABELS, columns=self.LABELS)
        cm_df.to_csv(os.path.join(self.results_folder, f'confusion_matrix_{self.experiment_num}.csv'), index=True)
        logger.info(f"Confusion matrix saved to {os.path.join(self.results_folder, f'confusion_matrix_{self.experiment_num}.csv')}")
        
        # Save confusion matrix percentages
        cm_pct_df = pd.DataFrame(cm_percentage, index=self.LABELS, columns=self.LABELS)
        cm_pct_df.to_csv(os.path.join(self.results_folder, f'confusion_matrix_percentage_{self.experiment_num}.csv'), index=True)
        logger.info(f"Confusion matrix percentages saved to {os.path.join(self.results_folder, f'confusion_matrix_percentage_{self.experiment_num}.csv')}")

        # Plot and log confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_df = pd.DataFrame(cm_percentage, index=self.LABELS, columns=self.LABELS)
        cax = ax.matshow(cm_df, cmap='Blues')
        plt.colorbar(cax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(cm_df.columns)
        ax.set_yticklabels(cm_df.index)
        
        for (i, j), val in np.ndenumerate(cm_percentage):
            ax.text(j, i, f"{val:.2f}%\n({cm[i, j]})", ha='center', va='center', color='black', fontsize=11)
        
        ax.set_title('Confusion Matrix')
        tensorboard_logger.add_figure('Confusion Matrix', fig, self.experiment_num)
        
        # Salva l'immagine nella sottocartella
        fig.savefig(os.path.join(self.results_folder, f'confusion_matrix_{self.experiment_num}.png'))
        plt.close(fig)
        logger.info(f"Confusion matrix plot saved to {os.path.join(self.results_folder, f'confusion_matrix_{self.experiment_num}.png')}")

        class_report = classification_report(
            true_labels,
            predictions,
            labels=range(num_classes),
            target_names=[f"Class {i}" for i in range(num_classes)],
            output_dict=True
        )
        class_metrics_df = pd.DataFrame(class_report).transpose()
        class_metrics_df.to_csv(os.path.join(self.results_folder, self.PER_CLASS_METRICS_FILENAME.format(self.experiment_num)), index=True)
        logger.info(f"Per-class metrics saved to {os.path.join(self.results_folder, self.PER_CLASS_METRICS_FILENAME.format(self.experiment_num))}")