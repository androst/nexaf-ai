"""
Temporal pattern feature extraction.

Extracts features describing circadian and weekly rhythms of AF episodes.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class TemporalPatternExtractor:
    """Extract circadian and weekly patterns per patient."""

    def __init__(
        self,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        start_column: str = "time_start_ep",
        daytime_column: str = "episode_start_during_day",
    ):
        """
        Initialize the temporal pattern extractor.

        Args:
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            start_column: Episode start timestamp column name.
            daytime_column: Binary day/night indicator column name.
        """
        self.id_column = id_column
        self.duration_column = duration_column
        self.start_column = start_column
        self.daytime_column = daytime_column

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all temporal pattern features for each patient.

        Args:
            df: Episode DataFrame with timestamps.

        Returns:
            Patient-level DataFrame with temporal features.
        """
        features = []

        for pid, group in df.groupby(self.id_column):
            patient_features = self._extract_patient_features(pid, group)
            features.append(patient_features)

        return pd.DataFrame(features)

    def _extract_patient_features(self, patient_id: int, df: pd.DataFrame) -> dict:
        """Extract features for a single patient."""
        features = {self.id_column: patient_id}
        n_episodes = len(df)

        # Day/night ratio using provided column
        if self.daytime_column in df.columns:
            daytime_values = df[self.daytime_column].values
            # Handle different encodings:
            # 1. Categorical strings like "night 00-05:59", "morning 6-11:59", etc.
            # 2. Binary (1/0, True/False)
            if df[self.daytime_column].dtype == 'object' or str(df[self.daytime_column].dtype) == 'category':
                # String-based: night contains "night", daytime is everything else
                daytime_mask = ~df[self.daytime_column].astype(str).str.lower().str.contains('night', na=False)
            else:
                # Binary encoding
                daytime_mask = pd.notna(daytime_values) & (daytime_values != 0)
            n_daytime = daytime_mask.sum()
            n_nighttime = n_episodes - n_daytime

            features["n_daytime_episodes"] = n_daytime
            features["n_nighttime_episodes"] = n_nighttime
            features["pct_daytime_episodes"] = 100 * n_daytime / n_episodes if n_episodes > 0 else 0

            if n_nighttime > 0:
                features["day_night_ratio"] = n_daytime / n_nighttime
            else:
                features["day_night_ratio"] = np.inf if n_daytime > 0 else 1.0

            # Burden-weighted day/night
            if self.duration_column in df.columns:
                # daytime_mask is already a boolean Series/array from above
                daytime_burden = df.loc[daytime_mask, self.duration_column].sum()
                nighttime_burden = df.loc[~daytime_mask, self.duration_column].sum()
                total_burden = df[self.duration_column].sum()
                features["burden_pct_daytime"] = (
                    100 * daytime_burden / total_burden if total_burden > 0 else 0
                )
                features["burden_pct_nighttime"] = (
                    100 * nighttime_burden / total_burden if total_burden > 0 else 0
                )
            else:
                features["burden_pct_daytime"] = features["pct_daytime_episodes"]

        else:
            features["n_daytime_episodes"] = np.nan
            features["n_nighttime_episodes"] = np.nan
            features["pct_daytime_episodes"] = np.nan
            features["day_night_ratio"] = np.nan
            features["burden_pct_daytime"] = np.nan

        # Hourly distribution (if timestamps available)
        if self.start_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.start_column]):
            hours = df[self.start_column].dt.hour.values
            hours = hours[~np.isnan(hours)]

            if len(hours) > 0:
                # Convert hours to radians for circular statistics
                hours_rad = hours * 2 * np.pi / 24

                # Circular mean (peak hour)
                sin_mean = np.mean(np.sin(hours_rad))
                cos_mean = np.mean(np.cos(hours_rad))
                mean_angle = np.arctan2(sin_mean, cos_mean)
                features["peak_hour"] = (mean_angle * 24 / (2 * np.pi)) % 24

                # Circular variance (concentration)
                r = np.sqrt(sin_mean**2 + cos_mean**2)
                features["hourly_concentration"] = r  # 0 = uniform, 1 = all same hour

                # Hourly entropy
                hour_counts = np.bincount(hours.astype(int), minlength=24)
                hour_probs = hour_counts / hour_counts.sum()
                hour_probs = hour_probs[hour_probs > 0]
                features["hourly_entropy"] = -np.sum(hour_probs * np.log2(hour_probs))
                features["hourly_entropy_normalized"] = features["hourly_entropy"] / np.log2(24)

            else:
                features["peak_hour"] = np.nan
                features["hourly_concentration"] = np.nan
                features["hourly_entropy"] = np.nan
                features["hourly_entropy_normalized"] = np.nan
        else:
            features["peak_hour"] = np.nan
            features["hourly_concentration"] = np.nan
            features["hourly_entropy"] = np.nan
            features["hourly_entropy_normalized"] = np.nan

        # Weekly patterns (if timestamps available)
        if self.start_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.start_column]):
            days_of_week = df[self.start_column].dt.dayofweek.values  # 0=Monday, 6=Sunday
            days_of_week = days_of_week[~np.isnan(days_of_week)]

            if len(days_of_week) > 0:
                # Weekend vs weekday
                is_weekend = (days_of_week >= 5)
                features["pct_weekend_episodes"] = 100 * np.mean(is_weekend)

                n_weekend = is_weekend.sum()
                n_weekday = len(days_of_week) - n_weekend

                if n_weekend > 0:
                    # Normalize by number of weekend vs weekday days (2 vs 5)
                    weekend_rate = n_weekend / 2
                    weekday_rate = n_weekday / 5
                    features["weekday_weekend_ratio"] = weekday_rate / weekend_rate if weekend_rate > 0 else np.inf
                else:
                    features["weekday_weekend_ratio"] = np.inf

                # Day of week entropy
                dow_counts = np.bincount(days_of_week.astype(int), minlength=7)
                dow_probs = dow_counts / dow_counts.sum()
                dow_probs = dow_probs[dow_probs > 0]
                features["dow_entropy"] = -np.sum(dow_probs * np.log2(dow_probs))
                features["dow_entropy_normalized"] = features["dow_entropy"] / np.log2(7)

            else:
                features["pct_weekend_episodes"] = np.nan
                features["weekday_weekend_ratio"] = np.nan
                features["dow_entropy"] = np.nan
                features["dow_entropy_normalized"] = np.nan
        else:
            features["pct_weekend_episodes"] = np.nan
            features["weekday_weekend_ratio"] = np.nan
            features["dow_entropy"] = np.nan
            features["dow_entropy_normalized"] = np.nan

        # Monthly patterns (seasonality)
        if self.start_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.start_column]):
            months = df[self.start_column].dt.month.values
            months = months[~np.isnan(months)]

            if len(months) > 0:
                # Month entropy
                month_counts = np.bincount(months.astype(int), minlength=13)[1:]  # Skip 0
                month_probs = month_counts / month_counts.sum()
                month_probs = month_probs[month_probs > 0]
                features["monthly_entropy"] = -np.sum(month_probs * np.log2(month_probs))

            else:
                features["monthly_entropy"] = np.nan
        else:
            features["monthly_entropy"] = np.nan

        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            "n_daytime_episodes",
            "n_nighttime_episodes",
            "pct_daytime_episodes",
            "day_night_ratio",
            "burden_pct_daytime",
            "peak_hour",
            "hourly_concentration",
            "hourly_entropy",
            "hourly_entropy_normalized",
            "pct_weekend_episodes",
            "weekday_weekend_ratio",
            "dow_entropy",
            "dow_entropy_normalized",
            "monthly_entropy",
        ]
