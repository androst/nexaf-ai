"""
Trajectory-based clustering methods.

Clusters patients based on trajectory shape similarity using DTW and k-Shape.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


class TrajectoryClusteringPipeline:
    """Cluster patients based on trajectory similarity."""

    def __init__(self, random_state: int = 42):
        """
        Initialize trajectory clustering pipeline.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state

    def kshape_clustering(
        self,
        time_series: np.ndarray,
        n_clusters: int = 4,
        n_init: int = 10,
        max_iter: int = 100,
    ) -> Dict[str, Any]:
        """
        k-Shape clustering (shape-based, scale-invariant).

        Args:
            time_series: 3D array of shape (n_samples, n_timepoints, n_features).
            n_clusters: Number of clusters.
            n_init: Number of initializations.
            max_iter: Maximum iterations.

        Returns:
            Dictionary with clustering results.
        """
        try:
            from tslearn.clustering import KShape
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        except ImportError:
            raise ImportError("tslearn required. Install with: pip install tslearn")

        # Normalize time series
        scaler = TimeSeriesScalerMeanVariance()
        ts_scaled = scaler.fit_transform(time_series)

        # Fit k-Shape
        ks = KShape(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        labels = ks.fit_predict(ts_scaled)

        # Calculate silhouette (using correlation distance)
        from sklearn.metrics import silhouette_score
        ts_2d = time_series[:, :, 0]
        try:
            silhouette = silhouette_score(ts_2d, labels, metric='correlation')
        except:
            silhouette = 0

        return {
            "n_clusters": n_clusters,
            "labels": labels,
            "silhouette": silhouette,
            "cluster_centers": ks.cluster_centers_,
            "inertia": ks.inertia_,
            "model": ks,
        }

    def dtw_kmeans_clustering(
        self,
        time_series: np.ndarray,
        n_clusters: int = 4,
        metric: str = "dtw",
        max_iter: int = 50,
    ) -> Dict[str, Any]:
        """
        DTW-based k-means clustering.

        Args:
            time_series: 3D array of shape (n_samples, n_timepoints, n_features).
            n_clusters: Number of clusters.
            metric: Distance metric ('dtw', 'softdtw').
            max_iter: Maximum iterations.

        Returns:
            Dictionary with clustering results.
        """
        try:
            from tslearn.clustering import TimeSeriesKMeans
        except ImportError:
            raise ImportError("tslearn required. Install with: pip install tslearn")

        km = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        labels = km.fit_predict(time_series)

        return {
            "n_clusters": n_clusters,
            "labels": labels,
            "cluster_centers": km.cluster_centers_,
            "inertia": km.inertia_,
            "model": km,
        }

    def compute_dtw_distance_matrix(
        self,
        time_series: np.ndarray,
        sakoe_chiba_radius: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute full DTW distance matrix.

        Args:
            time_series: 3D array of shape (n_samples, n_timepoints, n_features).
            sakoe_chiba_radius: Constraint band width for faster computation.

        Returns:
            2D distance matrix of shape (n_samples, n_samples).
        """
        try:
            from tslearn.metrics import cdist_dtw
        except ImportError:
            # Fallback to simple implementation
            return self._compute_dtw_matrix_simple(time_series)

        return cdist_dtw(
            time_series,
            time_series,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )

    def _compute_dtw_matrix_simple(self, time_series: np.ndarray) -> np.ndarray:
        """Simple DTW implementation as fallback."""
        n_samples = len(time_series)
        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = self._dtw_distance(time_series[i, :, 0], time_series[j, :, 0])
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Compute DTW distance between two sequences."""
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )

        return dtw_matrix[n, m]

    def hierarchical_dtw(
        self,
        distance_matrix: np.ndarray,
        n_clusters: int = 4,
        linkage_method: str = "average",
    ) -> Dict[str, Any]:
        """
        Hierarchical clustering with precomputed DTW distances.

        Args:
            distance_matrix: Precomputed distance matrix.
            n_clusters: Number of clusters.
            linkage_method: Linkage method ('average', 'complete', 'single').

        Returns:
            Dictionary with clustering results.
        """
        # Convert to condensed distance matrix
        condensed = squareform(distance_matrix)

        # Compute linkage
        Z = linkage(condensed, method=linkage_method)

        # Cut tree to get clusters
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

        return {
            "n_clusters": n_clusters,
            "labels": labels,
            "linkage_matrix": Z,
        }

    def find_optimal_k(
        self,
        time_series: np.ndarray,
        k_range: range = range(2, 8),
        method: str = "kshape",
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using various metrics.

        Args:
            time_series: 3D array of trajectories.
            k_range: Range of k values to try.
            method: Clustering method ('kshape', 'dtw').

        Returns:
            Dictionary with metrics for each k value.
        """
        results = {
            "k_values": [],
            "inertias": [],
            "silhouettes": [],
        }

        for k in k_range:
            if method == "kshape":
                res = self.kshape_clustering(time_series, n_clusters=k)
            else:
                res = self.dtw_kmeans_clustering(time_series, n_clusters=k)

            results["k_values"].append(k)
            results["inertias"].append(res.get("inertia", 0))
            results["silhouettes"].append(res.get("silhouette", 0))

        # Find best k by silhouette
        best_idx = np.argmax(results["silhouettes"])
        results["best_k"] = results["k_values"][best_idx]
        results["best_silhouette"] = results["silhouettes"][best_idx]

        return results

    def get_cluster_trajectories(
        self,
        time_series: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Get average trajectory for each cluster.

        Args:
            time_series: 3D array of trajectories.
            labels: Cluster labels.

        Returns:
            Dictionary mapping cluster ID to average trajectory.
        """
        cluster_trajectories = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            cluster_ts = time_series[mask]
            cluster_trajectories[label] = np.mean(cluster_ts, axis=0)

        return cluster_trajectories


def compare_trajectory_clustering(
    time_series: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compare different trajectory clustering methods.

    Args:
        time_series: 3D array of trajectories.
        n_clusters: Number of clusters to create.
        random_state: Random seed.

    Returns:
        Dictionary with results from each method.
    """
    pipeline = TrajectoryClusteringPipeline(random_state=random_state)
    results = {}

    # k-Shape
    try:
        results["kshape"] = pipeline.kshape_clustering(time_series, n_clusters=n_clusters)
    except Exception as e:
        results["kshape"] = {"error": str(e)}

    # DTW k-means
    try:
        results["dtw_kmeans"] = pipeline.dtw_kmeans_clustering(time_series, n_clusters=n_clusters)
    except Exception as e:
        results["dtw_kmeans"] = {"error": str(e)}

    # Hierarchical with DTW
    try:
        dist_matrix = pipeline.compute_dtw_distance_matrix(time_series)
        results["hierarchical_dtw"] = pipeline.hierarchical_dtw(dist_matrix, n_clusters=n_clusters)
    except Exception as e:
        results["hierarchical_dtw"] = {"error": str(e)}

    return results
