"""
AF burden feature extraction.

Extracts patient-level features related to overall AF burden.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from ..config import DEFAULT_CONFIG


class BurdenFeatureExtractor:
    """Extract AF burden features per patient."""

    def __init__(
        self,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        time_column: str = "days_since_implant",
        time_windows: Optional[List[Tuple[int, int]]] = None,
        max_follow_up_days: int = DEFAULT_CONFIG.max_follow_up_days,
    ):
        """
        Initialize the burden feature extractor.

        Args:
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            time_column: Time since implant column name.
            time_windows: Time windows for windowed burden features.
            max_follow_up_days: Maximum follow-up period for rate calculations.
        """
        self.id_column = id_column
        self.duration_column = duration_column
        self.time_column = time_column
        self.time_windows = time_windows or DEFAULT_CONFIG.time_windows
        self.max_follow_up_days = max_follow_up_days

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all burden features for each patient.

        Args:
            df: Episode DataFrame with time alignment.

        Returns:
            Patient-level DataFrame with burden features.
        """
        features = []

        for pid, group in df.groupby(self.id_column):
            patient_features = self._extract_patient_features(pid, group)
            features.append(patient_features)

        return pd.DataFrame(features)

    def _extract_patient_features(
        self, patient_id: int, df: pd.DataFrame
    ) -> dict:
        """Extract features for a single patient."""
        features = {self.id_column: patient_id}

        durations = df[self.duration_column].values
        times = df[self.time_column].values

        # Total burden
        features["total_af_minutes"] = np.sum(durations)
        features["total_af_hours"] = features["total_af_minutes"] / 60
        features["total_af_days"] = features["total_af_minutes"] / 1440

        # Follow-up duration
        follow_up_days = min(np.max(times), self.max_follow_up_days) if len(times) > 0 else 0
        features["follow_up_days"] = follow_up_days

        # Burden rate (per observation period)
        if follow_up_days > 0:
            total_minutes_in_follow_up = follow_up_days * 1440
            features["af_burden_percent"] = 100 * features["total_af_minutes"] / total_minutes_in_follow_up
            features["af_burden_rate_per_day"] = features["total_af_minutes"] / follow_up_days
            features["af_burden_rate_per_week"] = features["af_burden_rate_per_day"] * 7
        else:
            features["af_burden_percent"] = 0
            features["af_burden_rate_per_day"] = 0
            features["af_burden_rate_per_week"] = 0

        # Time-windowed burden
        for start, end in self.time_windows:
            window_mask = (times >= start) & (times < end)
            window_burden = np.sum(durations[window_mask])
            window_days = end - start

            features[f"burden_minutes_{start}_{end}d"] = window_burden
            features[f"burden_rate_{start}_{end}d"] = window_burden / window_days if window_days > 0 else 0

        # Burden progression (first half vs second half of follow-up)
        if follow_up_days > 0:
            midpoint = follow_up_days / 2
            first_half_mask = times < midpoint
            second_half_mask = times >= midpoint

            features["burden_first_half"] = np.sum(durations[first_half_mask])
            features["burden_second_half"] = np.sum(durations[second_half_mask])

            if features["burden_first_half"] > 0:
                features["burden_ratio_second_first"] = (
                    features["burden_second_half"] / features["burden_first_half"]
                )
            else:
                features["burden_ratio_second_first"] = (
                    np.inf if features["burden_second_half"] > 0 else 1.0
                )
        else:
            features["burden_first_half"] = 0
            features["burden_second_half"] = 0
            features["burden_ratio_second_first"] = 1.0

        # Burden quartiles
        if follow_up_days > 0:
            q1, q2, q3 = follow_up_days * 0.25, follow_up_days * 0.5, follow_up_days * 0.75

            features["burden_q1"] = np.sum(durations[times < q1])
            features["burden_q2"] = np.sum(durations[(times >= q1) & (times < q2)])
            features["burden_q3"] = np.sum(durations[(times >= q2) & (times < q3)])
            features["burden_q4"] = np.sum(durations[times >= q3])

        # Early vs late burden (first 30 days vs rest)
        early_mask = times < 30
        features["burden_first_30d"] = np.sum(durations[early_mask])
        features["burden_after_30d"] = np.sum(durations[~early_mask])

        # Time to burden milestones
        cumsum = np.cumsum(durations)
        total_burden = cumsum[-1] if len(cumsum) > 0 else 0

        if total_burden > 0:
            # Time to 25%, 50%, 75% of total burden
            for pct in [25, 50, 75]:
                threshold = total_burden * pct / 100
                idx = np.searchsorted(cumsum, threshold)
                if idx < len(times):
                    features[f"time_to_{pct}pct_burden"] = times[idx]
                else:
                    features[f"time_to_{pct}pct_burden"] = np.nan
        else:
            features["time_to_25pct_burden"] = np.nan
            features["time_to_50pct_burden"] = np.nan
            features["time_to_75pct_burden"] = np.nan

        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        base_features = [
            "total_af_minutes",
            "total_af_hours",
            "total_af_days",
            "follow_up_days",
            "af_burden_percent",
            "af_burden_rate_per_day",
            "af_burden_rate_per_week",
            "burden_first_half",
            "burden_second_half",
            "burden_ratio_second_first",
            "burden_q1",
            "burden_q2",
            "burden_q3",
            "burden_q4",
            "burden_first_30d",
            "burden_after_30d",
            "time_to_25pct_burden",
            "time_to_50pct_burden",
            "time_to_75pct_burden",
        ]

        window_features = []
        for start, end in self.time_windows:
            window_features.append(f"burden_minutes_{start}_{end}d")
            window_features.append(f"burden_rate_{start}_{end}d")

        return base_features + window_features
