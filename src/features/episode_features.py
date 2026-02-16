"""
Episode pattern feature extraction.

Extracts features describing episode frequency and duration patterns.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class EpisodePatternExtractor:
    """Extract episode frequency and duration patterns per patient."""

    def __init__(
        self,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        time_column: str = "days_since_implant",
        start_column: str = "time_start_ep",
    ):
        """
        Initialize the episode pattern extractor.

        Args:
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            time_column: Time since implant column name.
            start_column: Episode start timestamp column name.
        """
        self.id_column = id_column
        self.duration_column = duration_column
        self.time_column = time_column
        self.start_column = start_column

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all episode pattern features for each patient.

        Args:
            df: Episode DataFrame with time alignment.

        Returns:
            Patient-level DataFrame with episode features.
        """
        features = []

        for pid, group in df.groupby(self.id_column):
            patient_features = self._extract_patient_features(pid, group)
            features.append(patient_features)

        return pd.DataFrame(features)

    def _extract_patient_features(self, patient_id: int, df: pd.DataFrame) -> dict:
        """Extract features for a single patient."""
        features = {self.id_column: patient_id}

        durations = df[self.duration_column].values
        times = df[self.time_column].values
        n_episodes = len(durations)

        # Episode count features
        features["n_episodes"] = n_episodes

        follow_up_days = np.max(times) if len(times) > 0 else 0
        follow_up_weeks = follow_up_days / 7
        follow_up_months = follow_up_days / 30.44

        if follow_up_weeks > 0:
            features["episode_rate_per_week"] = n_episodes / follow_up_weeks
        else:
            features["episode_rate_per_week"] = 0

        if follow_up_months > 0:
            features["episode_rate_per_month"] = n_episodes / follow_up_months
        else:
            features["episode_rate_per_month"] = 0

        # Duration distribution
        if n_episodes > 0:
            features["mean_episode_duration"] = np.mean(durations)
            features["median_episode_duration"] = np.median(durations)
            features["max_episode_duration"] = np.max(durations)
            features["min_episode_duration"] = np.min(durations)
            features["std_episode_duration"] = np.std(durations) if n_episodes > 1 else 0

            # Coefficient of variation
            if features["mean_episode_duration"] > 0:
                features["cv_episode_duration"] = (
                    features["std_episode_duration"] / features["mean_episode_duration"]
                )
            else:
                features["cv_episode_duration"] = 0

            # Skewness (requires at least 3 episodes)
            if n_episodes >= 3:
                features["skew_episode_duration"] = stats.skew(durations)
            else:
                features["skew_episode_duration"] = 0

            # Duration percentiles
            features["duration_p25"] = np.percentile(durations, 25)
            features["duration_p75"] = np.percentile(durations, 75)
            features["duration_p90"] = np.percentile(durations, 90)
            features["duration_iqr"] = features["duration_p75"] - features["duration_p25"]

            # Duration categories
            features["pct_episodes_lt_1h"] = 100 * np.mean(durations < 60)
            features["pct_episodes_1h_24h"] = 100 * np.mean((durations >= 60) & (durations < 1440))
            features["pct_episodes_gt_24h"] = 100 * np.mean(durations >= 1440)
            features["pct_episodes_gt_7d"] = 100 * np.mean(durations >= 10080)

            features["n_episodes_gt_1h"] = np.sum(durations >= 60)
            features["n_episodes_gt_24h"] = np.sum(durations >= 1440)

        else:
            # No episodes - set to zero/nan
            for feat in [
                "mean_episode_duration", "median_episode_duration",
                "max_episode_duration", "min_episode_duration",
                "std_episode_duration", "cv_episode_duration",
                "skew_episode_duration", "duration_p25", "duration_p75",
                "duration_p90", "duration_iqr",
            ]:
                features[feat] = 0

            for feat in [
                "pct_episodes_lt_1h", "pct_episodes_1h_24h",
                "pct_episodes_gt_24h", "pct_episodes_gt_7d",
                "n_episodes_gt_1h", "n_episodes_gt_24h",
            ]:
                features[feat] = 0

        # Inter-episode intervals
        if n_episodes >= 2:
            # Sort by time
            sorted_times = np.sort(times)
            intervals = np.diff(sorted_times)  # In days

            features["mean_inter_episode_interval"] = np.mean(intervals)
            features["median_inter_episode_interval"] = np.median(intervals)
            features["min_inter_episode_interval"] = np.min(intervals)
            features["max_inter_episode_interval"] = np.max(intervals)
            features["std_inter_episode_interval"] = np.std(intervals)

            if features["mean_inter_episode_interval"] > 0:
                features["cv_inter_episode_interval"] = (
                    features["std_inter_episode_interval"] / features["mean_inter_episode_interval"]
                )
            else:
                features["cv_inter_episode_interval"] = 0

            # Burstiness index: (std - mean) / (std + mean)
            # Ranges from -1 (periodic) to 1 (bursty), 0 = Poisson
            std_int = features["std_inter_episode_interval"]
            mean_int = features["mean_inter_episode_interval"]
            if (std_int + mean_int) > 0:
                features["burstiness_index"] = (std_int - mean_int) / (std_int + mean_int)
            else:
                features["burstiness_index"] = 0

            # Clustering metrics
            features["pct_intervals_lt_1d"] = 100 * np.mean(intervals < 1)
            features["pct_intervals_lt_7d"] = 100 * np.mean(intervals < 7)

        else:
            for feat in [
                "mean_inter_episode_interval", "median_inter_episode_interval",
                "min_inter_episode_interval", "max_inter_episode_interval",
                "std_inter_episode_interval", "cv_inter_episode_interval",
                "burstiness_index", "pct_intervals_lt_1d", "pct_intervals_lt_7d",
            ]:
                features[feat] = np.nan if n_episodes == 0 else 0

        # Episode regularity (Fano factor)
        if n_episodes >= 2 and follow_up_days > 7:
            # Count episodes per week
            weeks = (times / 7).astype(int)
            week_counts = pd.Series(weeks).value_counts().values

            if len(week_counts) > 1 and np.mean(week_counts) > 0:
                # Fano factor: variance / mean
                features["fano_factor_weekly"] = np.var(week_counts) / np.mean(week_counts)
            else:
                features["fano_factor_weekly"] = 1.0
        else:
            features["fano_factor_weekly"] = np.nan

        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            "n_episodes",
            "episode_rate_per_week",
            "episode_rate_per_month",
            "mean_episode_duration",
            "median_episode_duration",
            "max_episode_duration",
            "min_episode_duration",
            "std_episode_duration",
            "cv_episode_duration",
            "skew_episode_duration",
            "duration_p25",
            "duration_p75",
            "duration_p90",
            "duration_iqr",
            "pct_episodes_lt_1h",
            "pct_episodes_1h_24h",
            "pct_episodes_gt_24h",
            "pct_episodes_gt_7d",
            "n_episodes_gt_1h",
            "n_episodes_gt_24h",
            "mean_inter_episode_interval",
            "median_inter_episode_interval",
            "min_inter_episode_interval",
            "max_inter_episode_interval",
            "std_inter_episode_interval",
            "cv_inter_episode_interval",
            "burstiness_index",
            "pct_intervals_lt_1d",
            "pct_intervals_lt_7d",
            "fano_factor_weekly",
        ]
