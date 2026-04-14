import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import logging

from utils.tensorboard_logger import get_tensorboard_logger

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import umap

logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

    def extract_embeddings(self):
        """
        Extracts embeddings from the model using the provided dataloader.
        """
        try:
            self.model.eval()
            embeddings_list = []
            labels_list = []
            with torch.no_grad():
                for batch in self.dataloader:
                    x_i, _, y = batch
                    x_i = x_i.to(self.device)
                    z_i = self.model(x_i)
                    embeddings_list.append(z_i.cpu().numpy())
                    labels_list.append(y.numpy())
            embeddings = np.concatenate(embeddings_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            unique_labels = np.unique(labels)
            return embeddings, labels, unique_labels
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            raise

class DimensionalityReducer:
    def __init__(self, embeddings, use_umap=True, use_pca=True, use_tsne=True):
        self.embeddings = embeddings
        self.use_umap = use_umap
        self.use_pca = use_pca
        self.use_tsne = use_tsne
        self.results = {}

    def apply_dimensionality_reduction(self):
        """
        Applies the selected dimensionality reduction techniques to the embeddings.
        """
        reduction_methods = {
            'tSNE': self.apply_tsne,
            'UMAP': self.apply_umap,
            'PCA': self.apply_pca
        }
        for method, func in reduction_methods.items():
            if getattr(self, f'use_{method.lower().replace("-", "_")}'):
                try:
                    logger.info(f"Applying {method}...")
                    embeddings_2d = func()
                    self.results[method] = {'embeddings': embeddings_2d}
                    logger.info(f"{method} completed.")
                except Exception as e:
                    logger.error(f"Error applying {method}: {e}")
                    self.results[method] = {'embeddings': None, 'error': str(e)}
        return self.results

    def apply_tsne(self):
        """
        Applies t-SNE to the scaled embeddings.

        Returns:
            array-like: 2D embeddings after t-SNE transformation.
        """
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            n_iter=1000,
            learning_rate='auto',
            init='random',
            verbose=0
        )
        return tsne.fit_transform(self.embeddings)

    def apply_umap(self):
        """
        Applies UMAP to the scaled embeddings.

        Returns:
            array-like: 2D embeddings after UMAP transformation.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="umap")
            umap_reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                n_jobs=-1
            )
            return umap_reducer.fit_transform(self.embeddings)

    def apply_pca(self):
        """
        Applies PCA to the scaled embeddings.

        Returns:
            array-like: 2D embeddings after PCA transformation.
        """
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(self.embeddings)

class MetricEvaluator:
    def __init__(self, embeddings, labels, n_clusters, output_metrics_dir, experiment_num):
        """
        Initializes the MetricEvaluator.

        Parameters:
            embeddings: Embeddings array.
            labels: True labels.
            n_clusters (int): Number of clusters.
            output_metrics_dir (Path): Directory to save metrics CSV files.
            experiment_num (str): Identifier for the experiment run.
        """
        self.embeddings = embeddings
        self.labels = labels
        self.n_clusters = n_clusters
        self.output_metrics_dir = Path(output_metrics_dir)
        self.experiment_num = experiment_num

    def evaluate_metrics(self, results):
        """
        Evaluates clustering metrics for each dimensionality reduction result.
        """
        tensorboard_logger = get_tensorboard_logger()
        logger.info("Evaluating metrics...")
        for method, result in results.items():
            if result['embeddings'] is None:
                logger.warning(f"Skipping metrics evaluation for {method} due to previous errors.")
                continue
            try:
                embeddings_2d = result['embeddings']
                metrics = self.compute_clustering_metrics(embeddings_2d)
                results[method]['metrics'] = metrics

                # Save metrics to CSV
                metrics_df = pd.DataFrame([metrics])
                metrics_filename = f"{method.lower()}_metrics_{self.experiment_num}.csv"
                metrics_path = self.output_metrics_dir / metrics_filename
                metrics_df.to_csv(metrics_path, index=False)
                logger.info(f"Metrics for {method} saved to {metrics_path}.")
                
                # Log metrics to TensorBoard
                if 'metrics' in result:
                    for metric_name, metric_value in result['metrics'].items():
                        if metric_value is not None:
                            tensorboard_logger.add_scalar(f"{method}/{metric_name}", metric_value)
            except Exception as e:
                logger.error(f"Error evaluating metrics for {method}: {e}")
                results[method]['metrics'] = {'error': str(e)}

    def compute_clustering_metrics(self, embeddings_2d):
        """
        Computes clustering metrics for the given embeddings.

        Parameters:
            embeddings_2d (array-like): 2D embeddings.

        Returns:
            dict: Computed metrics.
        """
        metrics = {}
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init='auto'
        )
        predicted_labels = kmeans.fit_predict(embeddings_2d)
        metrics['Silhouette Score'] = silhouette_score(embeddings_2d, predicted_labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(embeddings_2d, predicted_labels)
        metrics['Adjusted Rand Index'] = adjusted_rand_score(self.labels, predicted_labels)
        metrics['Purity Score'] = self.purity_score(self.labels, predicted_labels)
        metrics['Average Entropy'] = self.compute_average_entropy(self.labels, predicted_labels)
        metrics['Adjusted Mutual Information'] = adjusted_mutual_info_score(self.labels, predicted_labels)
        metrics['Trustworthiness'] = trustworthiness(self.embeddings, embeddings_2d, n_neighbors=5)
        metrics['Continuity'] = None
        return metrics

    @staticmethod
    def purity_score(true_labels, predicted_labels):
        """
        Calculates the purity score for the clustering.

        Parameters:
            true_labels (array-like): True labels.
            predicted_labels (array-like): Predicted cluster labels.

        Returns:
            float: Purity score.
        """
        contingency_matrix = np.zeros((len(np.unique(true_labels)), len(np.unique(predicted_labels))))
        for i, true_label in enumerate(np.unique(true_labels)):
            for j, pred_label in enumerate(np.unique(predicted_labels)):
                contingency_matrix[i, j] = np.sum(
                    (true_labels == true_label) & (predicted_labels == pred_label)
                )
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        purity = contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)
        return purity

    @staticmethod
    def compute_average_entropy(true_labels, predicted_labels):
        """
        Computes the average entropy for the clustering.

        Parameters:
            true_labels (array-like): True labels.
            predicted_labels (array-like): Predicted cluster labels.

        Returns:
            float: Average entropy.
        """
        cluster_labels = defaultdict(list)
        for label, cluster in zip(true_labels, predicted_labels):
            cluster_labels[cluster].append(label)
        total_entropy = 0
        for labels_in_cluster in cluster_labels.values():
            label_counts = np.bincount(labels_in_cluster)
            probabilities = label_counts / len(labels_in_cluster)
            probabilities = probabilities[probabilities > 0]
            cluster_entropy = entropy(probabilities, base=2)
            total_entropy += cluster_entropy * len(labels_in_cluster)
        average_entropy = total_entropy / len(true_labels)
        return average_entropy

class EmbeddingVisualizer:
    def __init__(self, labels, n_clusters, output_image_dir, experiment_num, visualization_fraction=0.1):
        """
        Initializes the EmbeddingVisualizer.

        Parameters:
            labels: True labels.
            n_clusters (int): Number of clusters.
            output_image_dir (Path): Directory to save visualization images.
            experiment_num (str): Identifier for the experiment run.
            visualization_fraction (float): Fraction of embeddings to sample for visualization.
        """
        self.labels = labels
        self.n_clusters = n_clusters
        self.output_image_dir = Path(output_image_dir)
        self.experiment_num = experiment_num
        self.visualization_fraction = visualization_fraction

    def visualize_embeddings(self, results):
        """
        Visualizes a subset of the embeddings and saves the plots as image files.
        """
        tensorboard_logger = get_tensorboard_logger()
        logger.info("Visualizing embeddings...")
        palette = sns.color_palette("tab10", n_colors=self.n_clusters)

        for method, result in results.items():
            if result['embeddings'] is None:
                logger.warning(f"Skipping visualization for {method} due to previous errors.")
                continue
            try:
                embeddings_2d = result['embeddings']
                # Sample a fraction of embeddings per class
                sampled_indices = []
                unique_labels = np.unique(self.labels)
                for label in unique_labels:
                    label_indices = np.where(self.labels == label)[0]
                    sample_size = max(1, int(len(label_indices) * self.visualization_fraction))
                    sampled = np.random.choice(label_indices, sample_size, replace=False)
                    sampled_indices.extend(sampled)
                sampled_embeddings = embeddings_2d[sampled_indices]
                sampled_labels = self.labels[sampled_indices]
                
                fig = plt.figure(figsize=(12, 10))
                sns.scatterplot(
                    x=sampled_embeddings[:, 0],
                    y=sampled_embeddings[:, 1],
                    hue=sampled_labels,
                    palette=palette,
                    legend='full',
                    alpha=0.7,
                    edgecolor='k',
                    linewidth=0.5
                )
                plt.title(f"{method} Visualization of Embeddings", fontsize=16)
                plt.xlabel(f"{method} Dimension 1", fontsize=14)
                plt.ylabel(f"{method} Dimension 2", fontsize=14)
                plt.legend(title="Classes", fontsize=12, title_fontsize=13, loc='best')
                plt.tight_layout()

                image_filename = f"{method.lower()}_embeddings_{self.experiment_num}.png"
                image_path = self.output_image_dir / image_filename
                plt.savefig(image_path, dpi=300)
                
                # Log figure to TensorBoard
                tensorboard_logger.add_figure(f"{method}/Embeddings", fig)
                plt.close()
                logger.info(f"Visualization for {method} saved to {image_path}.")
            except Exception as e:
                logger.error(f"Error visualizing embeddings for {method}: {e}")

class LatentSpaceEvaluator:
    def __init__(self, model, dataloader, device='cpu', umap_enabled=True, pca_enabled=True, tsne_enabled=True, visualize=True, compute_metrics=True, n_clusters=5, output_image_dir='visualizations', output_metrics_dir='metrics', experiment_num='default_experiment', visualization_fraction=0.5):
        """
        Initializes the LatentSpaceEvaluator.

        Parameters:
            model: The neural network model to extract embeddings from.
            dataloader: DataLoader providing the data batches.
            device (str): Device to run the model on ('cpu' or 'cuda').
            umap_enabled (bool): Whether to use UMAP for dimensionality reduction.
            pca_enabled (bool): Whether to use PCA for dimensionality reduction.
            tsne_enabled (bool): Whether to use t-SNE for dimensionality reduction.
            visualize (bool): Whether to visualize embeddings.
            compute_metrics (bool): Whether to compute evaluation metrics.
            n_clusters (int): Number of expected clusters/classes.
            output_image_dir (str): Directory path to save visualization images.
            output_metrics_dir (str): Directory path to save metrics CSV files.
            experiment_num (str): Identifier for the experiment run.
            visualization_fraction (float): Fraction of embeddings to sample for visualization.
        """
        self.extractor = EmbeddingExtractor(model, dataloader, device)
        self.reducer = None
        self.evaluator = None
        self.visualizer = None
        self.use_umap = umap_enabled
        self.use_pca = pca_enabled
        self.use_tsne = tsne_enabled
        self.visualize = visualize
        self.compute_metrics = compute_metrics
        self.n_clusters = n_clusters
        self.experiment_num = experiment_num
        self.visualization_fraction = visualization_fraction

        # Set up output directories
        self.output_image_dir = Path(output_image_dir)
        self.output_metrics_dir = Path(output_metrics_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.output_metrics_dir.mkdir(parents=True, exist_ok=True)

        # Set Seaborn style for better aesthetics
        sns.set_theme(style='whitegrid', context='talk')
        matplotlib.rcParams.update({'figure.autolayout': True})

    def run(self):
        """
        Executes the full evaluation pipeline: extraction, dimensionality reduction,
        metric evaluation, and visualization.

        This method orchestrates the entire evaluation process by calling the appropriate methods
        in sequence and handling any exceptions that occur.
        """
        tensorboard_logger = get_tensorboard_logger()
        logger.info("Starting Latent Space Evaluation...")
        try:
            embeddings, labels, unique_labels = self.extractor.extract_embeddings()
            if len(unique_labels) != self.n_clusters:
                raise ValueError(
                    f"Expected {self.n_clusters} unique labels, but found {len(unique_labels)}: {unique_labels}"
                )
            embeddings_scaled = StandardScaler().fit_transform(embeddings)
            self.reducer = DimensionalityReducer(embeddings_scaled, self.use_umap, self.use_pca, self.use_tsne)
            results = self.reducer.apply_dimensionality_reduction()
            if self.compute_metrics:
                self.evaluator = MetricEvaluator(embeddings_scaled, labels, self.n_clusters, self.output_metrics_dir, self.experiment_num)
                self.evaluator.evaluate_metrics(results)
                logger.info("Evaluation metrics computed and saved.")
            if self.visualize:
                self.visualizer = EmbeddingVisualizer(labels, self.n_clusters, self.output_image_dir, self.experiment_num, self.visualization_fraction)
                self.visualizer.visualize_embeddings(results)
                logger.info("Embeddings visualized and saved.")
            logger.info("Latent Space Evaluation Completed.")
        except Exception as e:
            logger.error(f"Error during Latent Space Evaluation: {e}")