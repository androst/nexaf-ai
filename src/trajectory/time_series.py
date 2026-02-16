"""
Time series representation for AF trajectories.

Creates standardized time series for trajectory clustering and analysis.
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class TimeSeriesRepresentation:
    """Create time series representations of AF trajectories."""

    def __init__(
        self,
        time_resolution: str = "daily",
        max_days: int = 365,
        representation: str = "cumulative_burden",
        n_timepoints: Optional[int] = None,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        time_column: str = "days_since_implant",
    ):
        """
        Initialize time series representation.

        Args:
            time_resolution: 'daily', 'weekly', or 'monthly'.
            max_days: Maximum follow-up to consider.
            representation: Type of time series to create:
                - 'cumulative_burden': Running total of AF minutes
                - 'burden_rate': AF burden per time unit
                - 'episode_count': Cumulative episode count
            n_timepoints: Fixed number of timepoints (overrides time_resolution).
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            time_column: Time since implant column name.
        """
        self.time_resolution = time_resolution
        self.max_days = max_days
        self.representation = representation
        self.n_timepoints = n_timepoints
        self.id_column = id_column
        self.duration_column = duration_column
        self.time_column = time_column

        # Calculate number of timepoints from resolution
        if self.n_timepoints is None:
            if time_resolution == "daily":
                self.n_timepoints = max_days
            elif time_resolution == "weekly":
                self.n_timepoints = max_days // 7
            elif time_resolution == "monthly":
                self.n_timepoints = max_days // 30
            else:
                self.n_timepoints = 100

    def create_dataset(
        self,
        df: pd.DataFrame,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Create time series dataset for all patients.

        Args:
            df: Episode DataFrame with time alignment.
            normalize: Whether to normalize each trajectory (z-score).

        Returns:
            Tuple of:
            - 3D array of shape (n_patients, n_timepoints, 1)
            - List of patient IDs in same order
        """
        patient_ids = df[self.id_column].unique()
        n_patients = len(patient_ids)

        # Initialize output array
        time_series = np.zeros((n_patients, self.n_timepoints, 1))
        id_list = []

        for i, pid in enumerate(patient_ids):
            patient_df = df[df[self.id_column] == pid]
            ts = self._create_patient_series(patient_df)

            if normalize and np.std(ts) > 0:
                ts = (ts - np.mean(ts)) / np.std(ts)

            time_series[i, :, 0] = ts
            id_list.append(pid)

        return time_series, id_list

    def _create_patient_series(self, df: pd.DataFrame) -> np.ndarray:
        """Create time series for a single patient."""
        if self.representation == "cumulative_burden":
            return self._create_cumulative_burden_series(df)
        elif self.representation == "burden_rate":
            return self._create_burden_rate_series(df)
        elif self.representation == "episode_count":
            return self._create_episode_count_series(df)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")

    def _create_cumulative_burden_series(self, df: pd.DataFrame) -> np.ndarray:
        """Create cumulative burden time series."""
        if len(df) == 0:
            return np.zeros(self.n_timepoints)

        df = df.sort_values(self.time_column)
        times = df[self.time_column].values
        durations = df[self.duration_column].values
        cumsum = np.cumsum(durations)

        # Cap at max_days
        mask = times <= self.max_days
        times = times[mask]
        cumsum = cumsum[mask] if len(cumsum) > len(times) else cumsum[:len(times)]

        if len(times) == 0:
            return np.zeros(self.n_timepoints)

        # Add endpoints
        times_ext = np.concatenate([[0], times, [self.max_days]])
        cumsum_ext = np.concatenate([[0], cumsum, [cumsum[-1]]])

        # Interpolate to fixed grid
        time_grid = np.linspace(0, self.max_days, self.n_timepoints)

        try:
            interp_func = interp1d(
                times_ext, cumsum_ext,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            result = interp_func(time_grid)
            result = np.maximum(result, 0)
        except Exception:
            result = np.zeros(self.n_timepoints)

        return result

    def _create_burden_rate_series(self, df: pd.DataFrame) -> np.ndarray:
        """Create burden rate (per time unit) series."""
        if len(df) == 0:
            return np.zeros(self.n_timepoints)

        df = df.sort_values(self.time_column)
        times = df[self.time_column].values
        durations = df[self.duration_column].values

        # Create time bins
        time_grid = np.linspace(0, self.max_days, self.n_timepoints + 1)
        bin_width = time_grid[1] - time_grid[0]

        # Assign episodes to bins
        result = np.zeros(self.n_timepoints)
        for t, d in zip(times, durations):
            if t <= self.max_days:
                bin_idx = int(t / bin_width)
                if bin_idx >= self.n_timepoints:
                    bin_idx = self.n_timepoints - 1
                result[bin_idx] += d

        # Convert to rate (burden per day)
        result = result / bin_width

        return result

    def _create_episode_count_series(self, df: pd.DataFrame) -> np.ndarray:
        """Create cumulative episode count series."""
        if len(df) == 0:
            return np.zeros(self.n_timepoints)

        df = df.sort_values(self.time_column)
        times = df[self.time_column].values

        # Cap at max_days
        times = times[times <= self.max_days]

        if len(times) == 0:
            return np.zeros(self.n_timepoints)

        # Create cumulative count
        cumcount = np.arange(1, len(times) + 1)

        # Add endpoints
        times_ext = np.concatenate([[0], times, [self.max_days]])
        cumcount_ext = np.concatenate([[0], cumcount, [len(times)]])

        # Interpolate
        time_grid = np.linspace(0, self.max_days, self.n_timepoints)

        try:
            interp_func = interp1d(
                times_ext, cumcount_ext,
                kind='previous',  # Step function
                fill_value='extrapolate',
                bounds_error=False
            )
            result = interp_func(time_grid)
            result = np.maximum(result, 0)
        except Exception:
            result = np.zeros(self.n_timepoints)

        return result

    def get_time_grid(self) -> np.ndarray:
        """Return the time grid used for interpolation."""
        return np.linspace(0, self.max_days, self.n_timepoints)

    def to_dataframe(
        self,
        time_series: np.ndarray,
        patient_ids: List[int],
    ) -> pd.DataFrame:
        """
        Convert time series array to DataFrame format.

        Args:
            time_series: 3D array from create_dataset.
            patient_ids: List of patient IDs.

        Returns:
            DataFrame with columns [ID, time, value].
        """
        time_grid = self.get_time_grid()
        records = []

        for i, pid in enumerate(patient_ids):
            for j, t in enumerate(time_grid):
                records.append({
                    self.id_column: pid,
                    "time": t,
                    "value": time_series[i, j, 0],
                })

        return pd.DataFrame(records)

    def compute_distance_matrix(
        self,
        time_series: np.ndarray,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix between trajectories.

        Args:
            time_series: 3D array from create_dataset.
            metric: Distance metric ('euclidean', 'correlation', 'cosine').

        Returns:
            2D distance matrix of shape (n_patients, n_patients).
        """
        from scipy.spatial.distance import pdist, squareform

        # Flatten to 2D (n_patients, n_timepoints)
        ts_2d = time_series[:, :, 0]

        if metric == "correlation":
            # Use 1 - correlation as distance
            distances = pdist(ts_2d, metric='correlation')
        elif metric == "cosine":
            distances = pdist(ts_2d, metric='cosine')
        else:
            distances = pdist(ts_2d, metric='euclidean')

        return squareform(distances)


class TrajectoryNormalizer:
    """Normalize trajectories for comparison across patients."""

    def __init__(self, method: str = "minmax"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method:
                - 'minmax': Scale to [0, 1]
                - 'zscore': Standard normalization
                - 'endpoint': Divide by final value
                - 'time': Normalize time axis to [0, 1]
        """
        self.method = method

    def normalize(
        self,
        time_series: np.ndarray,
        axis: int = 1,
    ) -> np.ndarray:
        """
        Normalize time series array.

        Args:
            time_series: 3D array of shape (n_patients, n_timepoints, n_features).
            axis: Axis along which to normalize (1 = time axis).

        Returns:
            Normalized array of same shape.
        """
        result = time_series.copy()

        if self.method == "minmax":
            for i in range(len(result)):
                min_val = result[i].min()
                max_val = result[i].max()
                if max_val > min_val:
                    result[i] = (result[i] - min_val) / (max_val - min_val)

        elif self.method == "zscore":
            for i in range(len(result)):
                mean_val = result[i].mean()
                std_val = result[i].std()
                if std_val > 0:
                    result[i] = (result[i] - mean_val) / std_val

        elif self.method == "endpoint":
            for i in range(len(result)):
                end_val = result[i, -1, 0]
                if end_val > 0:
                    result[i] = result[i] / end_val

        return result
