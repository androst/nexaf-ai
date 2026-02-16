"""
RR interval feature extraction.

Extracts features related to heart rate during AF episodes.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class RRIntervalExtractor:
    """Extract RR interval (heart rate) features during AF per patient."""

    def __init__(
        self,
        id_column: str = "ID",
        rr_column: str = "AF_MEAN_RR_INTERVAL_msec",
        rate_column: str = "AF_MEAN_RR_RATE_bpm",
        duration_column: str = "af_episode_minutes",
    ):
        """
        Initialize the RR interval feature extractor.

        Args:
            id_column: Patient ID column name.
            rr_column: RR interval column name (in milliseconds).
            rate_column: Heart rate column name (in bpm).
            duration_column: Episode duration column name.
        """
        self.id_column = id_column
        self.rr_column = rr_column
        self.rate_column = rate_column
        self.duration_column = duration_column

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all RR interval features for each patient.

        Args:
            df: Episode DataFrame with RR interval data.

        Returns:
            Patient-level DataFrame with RR features.
        """
        features = []

        for pid, group in df.groupby(self.id_column):
            patient_features = self._extract_patient_features(pid, group)
            features.append(patient_features)

        return pd.DataFrame(features)

    def _extract_patient_features(self, patient_id: int, df: pd.DataFrame) -> dict:
        """Extract features for a single patient."""
        features = {self.id_column: patient_id}

        # Get RR intervals (handle missing values and string types)
        if self.rr_column in df.columns:
            rr_series = pd.to_numeric(df[self.rr_column], errors='coerce')
            rr_values = rr_series.dropna().values.astype(float)
        else:
            rr_values = np.array([])

        # Get heart rates (calculate from RR if not available)
        if self.rate_column in df.columns:
            rate_series = pd.to_numeric(df[self.rate_column], errors='coerce')
            rate_values = rate_series.dropna().values.astype(float)
        elif len(rr_values) > 0:
            # Convert RR (ms) to rate (bpm): rate = 60000 / RR
            valid_rr = rr_values[rr_values > 0]
            rate_values = 60000 / valid_rr if len(valid_rr) > 0 else np.array([])
        else:
            rate_values = np.array([])

        n_with_rr = len(rr_values)
        n_episodes = len(df)

        features["n_episodes_with_rr"] = n_with_rr
        features["pct_episodes_with_rr"] = 100 * n_with_rr / n_episodes if n_episodes > 0 else 0

        if n_with_rr > 0:
            # RR interval statistics
            features["mean_rr_interval"] = np.mean(rr_values)
            features["median_rr_interval"] = np.median(rr_values)
            features["std_rr_interval"] = np.std(rr_values) if n_with_rr > 1 else 0
            features["min_rr_interval"] = np.min(rr_values)
            features["max_rr_interval"] = np.max(rr_values)

            # Coefficient of variation
            if features["mean_rr_interval"] > 0:
                features["cv_rr_interval"] = features["std_rr_interval"] / features["mean_rr_interval"]
            else:
                features["cv_rr_interval"] = 0

            # Range
            features["range_rr_interval"] = features["max_rr_interval"] - features["min_rr_interval"]

            # Heart rate statistics
            if len(rate_values) > 0:
                features["mean_ventricular_rate"] = np.mean(rate_values)
                features["median_ventricular_rate"] = np.median(rate_values)
                features["std_ventricular_rate"] = np.std(rate_values) if len(rate_values) > 1 else 0
                features["min_ventricular_rate"] = np.min(rate_values)
                features["max_ventricular_rate"] = np.max(rate_values)
            else:
                # Calculate from RR
                features["mean_ventricular_rate"] = 60000 / features["mean_rr_interval"]
                features["median_ventricular_rate"] = 60000 / features["median_rr_interval"]
                features["std_ventricular_rate"] = np.nan
                features["min_ventricular_rate"] = 60000 / features["max_rr_interval"]
                features["max_ventricular_rate"] = 60000 / features["min_rr_interval"]

            # Rate categories
            # Rapid AF: HR > 110 bpm (RR < 545 ms)
            # Controlled AF: HR 60-110 bpm (RR 545-1000 ms)
            # Slow AF: HR < 60 bpm (RR > 1000 ms)
            features["pct_rapid_af"] = 100 * np.mean(rr_values < 545)
            features["pct_controlled_af"] = 100 * np.mean((rr_values >= 545) & (rr_values <= 1000))
            features["pct_slow_af"] = 100 * np.mean(rr_values > 1000)

            # Very rapid AF: HR > 150 bpm (RR < 400 ms)
            features["pct_very_rapid_af"] = 100 * np.mean(rr_values < 400)

            # Duration-weighted mean RR (if durations available)
            if self.duration_column in df.columns:
                df_with_rr = df.dropna(subset=[self.rr_column])
                if len(df_with_rr) > 0:
                    durations = df_with_rr[self.duration_column].values
                    rr_weighted = df_with_rr[self.rr_column].values
                    total_duration = np.sum(durations)
                    if total_duration > 0:
                        features["duration_weighted_mean_rr"] = (
                            np.sum(rr_weighted * durations) / total_duration
                        )
                    else:
                        features["duration_weighted_mean_rr"] = features["mean_rr_interval"]
                else:
                    features["duration_weighted_mean_rr"] = np.nan
            else:
                features["duration_weighted_mean_rr"] = features["mean_rr_interval"]

            # RR-duration correlation
            if self.duration_column in df.columns:
                df_with_rr = df.dropna(subset=[self.rr_column])
                if len(df_with_rr) >= 3:
                    corr, p_value = stats.pearsonr(
                        df_with_rr[self.rr_column].values,
                        df_with_rr[self.duration_column].values
                    )
                    features["rr_duration_correlation"] = corr
                    features["rr_duration_correlation_pvalue"] = p_value
                else:
                    features["rr_duration_correlation"] = np.nan
                    features["rr_duration_correlation_pvalue"] = np.nan
            else:
                features["rr_duration_correlation"] = np.nan
                features["rr_duration_correlation_pvalue"] = np.nan

        else:
            # No RR data
            for feat in [
                "mean_rr_interval", "median_rr_interval", "std_rr_interval",
                "min_rr_interval", "max_rr_interval", "cv_rr_interval",
                "range_rr_interval", "mean_ventricular_rate", "median_ventricular_rate",
                "std_ventricular_rate", "min_ventricular_rate", "max_ventricular_rate",
                "pct_rapid_af", "pct_controlled_af", "pct_slow_af", "pct_very_rapid_af",
                "duration_weighted_mean_rr", "rr_duration_correlation",
                "rr_duration_correlation_pvalue",
            ]:
                features[feat] = np.nan

        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            "n_episodes_with_rr",
            "pct_episodes_with_rr",
            "mean_rr_interval",
            "median_rr_interval",
            "std_rr_interval",
            "min_rr_interval",
            "max_rr_interval",
            "cv_rr_interval",
            "range_rr_interval",
            "mean_ventricular_rate",
            "median_ventricular_rate",
            "std_ventricular_rate",
            "min_ventricular_rate",
            "max_ventricular_rate",
            "pct_rapid_af",
            "pct_controlled_af",
            "pct_slow_af",
            "pct_very_rapid_af",
            "duration_weighted_mean_rr",
            "rr_duration_correlation",
            "rr_duration_correlation_pvalue",
        ]
