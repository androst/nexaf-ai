"""
Cluster validation and comparison utilities.

Provides metrics and methods for evaluating clustering quality.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


class ClusterValidator:
    """Validate clustering quality and stability."""

    def __init__(self):
        """Initialize cluster validator."""
        pass

    def compute_internal_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute internal validation metrics.

        Args:
            X: Feature matrix.
            labels: Cluster labels.

        Returns:
            Dictionary of metric names to values.
        """
        n_clusters = len(np.unique(labels))

        if n_clusters < 2:
            return {
                "silhouette": 0,
                "calinski_harabasz": 0,
                "davies_bouldin": np.inf,
                "n_clusters": n_clusters,
            }

        # Handle noise points (label = -1)
        valid_mask = labels >= 0
        if valid_mask.sum() < n_clusters:
            return {
                "silhouette": 0,
                "calinski_harabasz": 0,
                "davies_bouldin": np.inf,
                "n_clusters": n_clusters,
            }

        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        return {
            "silhouette": silhouette_score(X_valid, labels_valid),
            "calinski_harabasz": calinski_harabasz_score(X_valid, labels_valid),
            "davies_bouldin": davies_bouldin_score(X_valid, labels_valid),
            "n_clusters": n_clusters,
            "n_samples": len(labels),
            "n_noise": np.sum(labels < 0),
        }

    def compute_silhouette_per_sample(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute silhouette score for each sample.

        Args:
            X: Feature matrix.
            labels: Cluster labels.

        Returns:
            Array of silhouette scores per sample.
        """
        valid_mask = labels >= 0
        if valid_mask.sum() < 2:
            return np.zeros(len(labels))

        scores = np.zeros(len(labels))
        scores[valid_mask] = silhouette_samples(X[valid_mask], labels[valid_mask])
        return scores

    def cluster_stability_bootstrap(
        self,
        X: np.ndarray,
        clusterer,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.8,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Assess clustering stability via bootstrap resampling.

        Args:
            X: Feature matrix.
            clusterer: Clustering object with fit_predict method.
            n_bootstrap: Number of bootstrap iterations.
            sample_fraction: Fraction of samples per bootstrap.
            random_state: Random seed.

        Returns:
            Dictionary with stability metrics.
        """
        np.random.seed(random_state)
        n_samples = len(X)
        sample_size = int(n_samples * sample_fraction)

        # Get reference labels
        reference_labels = clusterer.fit_predict(X)

        ari_scores = []
        nmi_scores = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_boot = X[indices]

            # Cluster bootstrap sample
            boot_labels = clusterer.fit_predict(X_boot)

            # Compare to reference (on overlapping samples)
            ref_labels_subset = reference_labels[indices]

            ari = adjusted_rand_score(ref_labels_subset, boot_labels)
            nmi = normalized_mutual_info_score(ref_labels_subset, boot_labels)

            ari_scores.append(ari)
            nmi_scores.append(nmi)

        return {
            "mean_ari": np.mean(ari_scores),
            "std_ari": np.std(ari_scores),
            "mean_nmi": np.mean(nmi_scores),
            "std_nmi": np.std(nmi_scores),
            "stability_score": np.mean(ari_scores),  # ARI as primary stability metric
        }

    def compare_clusterings(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compare two clustering solutions.

        Args:
            labels1: First set of cluster labels.
            labels2: Second set of cluster labels.

        Returns:
            Dictionary with comparison metrics.
        """
        return {
            "adjusted_rand_index": adjusted_rand_score(labels1, labels2),
            "normalized_mutual_info": normalized_mutual_info_score(labels1, labels2),
        }

    def optimal_k_analysis(
        self,
        X: np.ndarray,
        k_range: range,
        method: str = "kmeans",
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Analyze clustering quality across range of k values.

        Args:
            X: Feature matrix.
            k_range: Range of cluster numbers to try.
            method: Clustering method ('kmeans', 'hierarchical').
            random_state: Random seed.

        Returns:
            DataFrame with metrics for each k.
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering

        results = []

        for k in k_range:
            if method == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            elif method == "hierarchical":
                clusterer = AgglomerativeClustering(n_clusters=k)
            else:
                raise ValueError(f"Unknown method: {method}")

            labels = clusterer.fit_predict(X)
            metrics = self.compute_internal_metrics(X, labels)
            metrics["k"] = k
            results.append(metrics)

        df = pd.DataFrame(results)

        # Add elbow/gap statistics
        if "calinski_harabasz" in df.columns:
            # Higher is better for CH
            df["ch_gain"] = df["calinski_harabasz"].diff()

        return df

    def get_cluster_summary(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.

        Args:
            X: Feature matrix.
            labels: Cluster labels.
            feature_names: Optional feature names.

        Returns:
            DataFrame with cluster summaries.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=feature_names)
        df["cluster"] = labels

        # Compute per-cluster statistics
        summaries = []
        for cluster_id in np.unique(labels):
            cluster_data = df[df["cluster"] == cluster_id]
            summary = {
                "cluster": cluster_id,
                "n_samples": len(cluster_data),
                "pct_samples": 100 * len(cluster_data) / len(df),
            }

            # Add mean for each feature
            for feat in feature_names:
                summary[f"{feat}_mean"] = cluster_data[feat].mean()
                summary[f"{feat}_std"] = cluster_data[feat].std()

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def identify_representative_samples(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_per_cluster: int = 3,
    ) -> Dict[int, List[int]]:
        """
        Find samples closest to cluster centroids.

        Args:
            X: Feature matrix.
            labels: Cluster labels.
            n_per_cluster: Number of representative samples per cluster.

        Returns:
            Dictionary mapping cluster ID to list of sample indices.
        """
        from sklearn.metrics import pairwise_distances

        representatives = {}

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_X = X[mask]

            if len(cluster_X) == 0:
                continue

            # Compute centroid
            centroid = cluster_X.mean(axis=0).reshape(1, -1)

            # Find distances to centroid
            distances = pairwise_distances(cluster_X, centroid).flatten()

            # Get indices of closest samples
            n_select = min(n_per_cluster, len(cluster_indices))
            closest_local = np.argsort(distances)[:n_select]
            representatives[cluster_id] = cluster_indices[closest_local].tolist()

        return representatives


def create_clustering_report(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method_name: str = "clustering",
) -> Dict[str, Any]:
    """
    Create comprehensive clustering report.

    Args:
        X: Feature matrix.
        labels: Cluster labels.
        feature_names: Optional feature names.
        method_name: Name of clustering method used.

    Returns:
        Dictionary with full clustering report.
    """
    validator = ClusterValidator()

    report = {
        "method": method_name,
        "n_clusters": len(np.unique(labels[labels >= 0])),
        "n_samples": len(labels),
        "n_noise": np.sum(labels < 0),
    }

    # Internal metrics
    metrics = validator.compute_internal_metrics(X, labels)
    report.update(metrics)

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    report["cluster_sizes"] = dict(zip(unique.tolist(), counts.tolist()))

    # Per-cluster summary
    report["cluster_summary"] = validator.get_cluster_summary(X, labels, feature_names)

    # Silhouette per sample
    report["silhouette_samples"] = validator.compute_silhouette_per_sample(X, labels)

    # Representative samples
    report["representatives"] = validator.identify_representative_samples(X, labels)

    return report
