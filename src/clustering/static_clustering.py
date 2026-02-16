"""
Static clustering methods for AF patient phenotyping.

Clusters patients based on extracted feature vectors.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class StaticClusteringPipeline:
    """Cluster patients based on feature vectors."""

    def __init__(
        self,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        """
        Initialize clustering pipeline.

        Args:
            random_state: Random seed for reproducibility.
            scale_features: Whether to standardize features before clustering.
        """
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self._fitted_scaler = False

    def preprocess(
        self,
        X: np.ndarray,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Preprocess features (scaling).

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            fit: Whether to fit the scaler (True for training, False for new data).

        Returns:
            Preprocessed feature matrix.
        """
        if self.scaler is None:
            return X

        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self._fitted_scaler = True
        else:
            if not self._fitted_scaler:
                raise ValueError("Scaler not fitted. Call preprocess with fit=True first.")
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def kmeans_clustering(
        self,
        X: np.ndarray,
        k_range: range = range(2, 10),
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        K-means clustering with elbow method for k selection.

        Args:
            X: Feature matrix.
            k_range: Range of k values to try.
            return_all: Whether to return results for all k values.

        Returns:
            Dictionary with clustering results and metrics.
        """
        X_processed = self.preprocess(X)

        results = {
            "k_values": [],
            "inertias": [],
            "silhouettes": [],
            "calinski": [],
            "davies_bouldin": [],
        }

        best_k = k_range[0]
        best_silhouette = -1
        best_labels = None
        best_model = None

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(X_processed)

            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X_processed, labels) if k > 1 else 0
            calinski = calinski_harabasz_score(X_processed, labels) if k > 1 else 0
            davies_bouldin_val = davies_bouldin_score(X_processed, labels) if k > 1 else 0

            results["k_values"].append(k)
            results["inertias"].append(inertia)
            results["silhouettes"].append(silhouette)
            results["calinski"].append(calinski)
            results["davies_bouldin"].append(davies_bouldin_val)

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
                best_labels = labels
                best_model = kmeans

        results["best_k"] = best_k
        results["best_silhouette"] = best_silhouette
        results["best_labels"] = best_labels
        results["best_model"] = best_model
        results["cluster_centers"] = best_model.cluster_centers_

        if not return_all:
            # Keep only best results
            results = {
                "k": best_k,
                "labels": best_labels,
                "silhouette": best_silhouette,
                "model": best_model,
                "cluster_centers": best_model.cluster_centers_,
            }

        return results

    def hierarchical_clustering(
        self,
        X: np.ndarray,
        n_clusters: int = 4,
        linkage: str = "ward",
        return_linkage_matrix: bool = False,
    ) -> Dict[str, Any]:
        """
        Agglomerative hierarchical clustering.

        Args:
            X: Feature matrix.
            n_clusters: Number of clusters.
            linkage: Linkage method ('ward', 'complete', 'average', 'single').
            return_linkage_matrix: Whether to compute and return linkage matrix.

        Returns:
            Dictionary with clustering results.
        """
        X_processed = self.preprocess(X)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        labels = model.fit_predict(X_processed)

        # Calculate metrics
        silhouette = silhouette_score(X_processed, labels) if n_clusters > 1 else 0
        calinski = calinski_harabasz_score(X_processed, labels) if n_clusters > 1 else 0
        davies_bouldin_val = davies_bouldin_score(X_processed, labels) if n_clusters > 1 else 0

        results = {
            "n_clusters": n_clusters,
            "labels": labels,
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies_bouldin_val,
            "model": model,
        }

        if return_linkage_matrix:
            from scipy.cluster.hierarchy import linkage as scipy_linkage
            results["linkage_matrix"] = scipy_linkage(X_processed, method=linkage)

        return results

    def hdbscan_clustering(
        self,
        X: np.ndarray,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        HDBSCAN density-based clustering.

        Args:
            X: Feature matrix.
            min_cluster_size: Minimum cluster size.
            min_samples: Minimum samples for core point.

        Returns:
            Dictionary with clustering results.
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan package required. Install with: pip install hdbscan")

        X_processed = self.preprocess(X)

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            gen_min_span_tree=True,
        )
        labels = model.fit_predict(X_processed)

        # Count clusters (excluding noise = -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        # Calculate metrics (only if we have valid clusters)
        if n_clusters > 1:
            # Exclude noise points for silhouette
            mask = labels != -1
            if mask.sum() > n_clusters:
                silhouette = silhouette_score(X_processed[mask], labels[mask])
            else:
                silhouette = 0
        else:
            silhouette = 0

        results = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels": labels,
            "silhouette": silhouette,
            "probabilities": model.probabilities_,
            "model": model,
        }

        return results

    def find_optimal_clusters(
        self,
        X: np.ndarray,
        k_range: range = range(2, 10),
        methods: List[str] = ["kmeans", "hierarchical"],
    ) -> pd.DataFrame:
        """
        Compare clustering solutions across methods and k values.

        Args:
            X: Feature matrix.
            k_range: Range of cluster numbers to try.
            methods: List of methods to compare.

        Returns:
            DataFrame with metrics for each method/k combination.
        """
        results = []
        X_processed = self.preprocess(X)

        for k in k_range:
            for method in methods:
                if method == "kmeans":
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(X_processed)
                elif method == "hierarchical":
                    hc = AgglomerativeClustering(n_clusters=k)
                    labels = hc.fit_predict(X_processed)
                else:
                    continue

                results.append({
                    "method": method,
                    "k": k,
                    "silhouette": silhouette_score(X_processed, labels),
                    "calinski_harabasz": calinski_harabasz_score(X_processed, labels),
                    "davies_bouldin": davies_bouldin_score(X_processed, labels),
                })

        return pd.DataFrame(results)

    def get_cluster_profiles(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create cluster profile summary (mean values per cluster).

        Args:
            X: Feature DataFrame or matrix.
            labels: Cluster labels.
            feature_names: Optional list of feature names.

        Returns:
            DataFrame with mean feature values per cluster.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)

        df["cluster"] = labels

        # Calculate mean and std per cluster
        profiles = df.groupby("cluster").agg(["mean", "std", "count"])

        return profiles

    def get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


def reduce_dimensions(
    X: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce dimensionality for visualization.

    Args:
        X: Feature matrix.
        method: Reduction method ('umap', 'pca', 'tsne').
        n_components: Number of output dimensions.
        random_state: Random seed.

    Returns:
        Reduced feature matrix.
    """
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
            )
            return reducer.fit_transform(X)
        except ImportError:
            print("umap-learn not installed, falling back to PCA")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(X)

    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, random_state=random_state)
        return tsne.fit_transform(X)

    raise ValueError(f"Unknown method: {method}")
