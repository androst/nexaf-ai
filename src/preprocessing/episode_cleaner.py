"""
Episode data cleaning and validation utilities.

Handles removal of invalid episodes and generates cleaning reports.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from ..config import DEFAULT_CONFIG


@dataclass
class EpisodeCleaningReport:
    """Report of cleaning operations performed."""

    n_original: int = 0
    n_removed_negative_duration: int = 0
    n_removed_zero_duration: int = 0
    n_removed_short_duration: int = 0
    n_removed_excessive_duration: int = 0
    n_removed_missing_timestamp: int = 0
    n_removed_invalid_rr: int = 0
    n_removed_overlapping: int = 0
    n_final: int = 0
    patients_with_removals: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        """Format report as readable string."""
        lines = [
            "Episode Cleaning Report",
            "=" * 40,
            f"Original episodes:         {self.n_original:,}",
            f"Removed (negative dur):    {self.n_removed_negative_duration:,}",
            f"Removed (zero duration):   {self.n_removed_zero_duration:,}",
            f"Removed (short duration):  {self.n_removed_short_duration:,}",
            f"Removed (excessive dur):   {self.n_removed_excessive_duration:,}",
            f"Removed (missing time):    {self.n_removed_missing_timestamp:,}",
            f"Removed (invalid RR):      {self.n_removed_invalid_rr:,}",
            f"Removed (overlapping):     {self.n_removed_overlapping:,}",
            f"Final episodes:            {self.n_final:,}",
            f"Retention rate:            {100 * self.n_final / self.n_original:.1f}%",
            f"Patients affected:         {len(self.patients_with_removals):,}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "n_original": self.n_original,
            "n_removed_negative_duration": self.n_removed_negative_duration,
            "n_removed_zero_duration": self.n_removed_zero_duration,
            "n_removed_short_duration": self.n_removed_short_duration,
            "n_removed_excessive_duration": self.n_removed_excessive_duration,
            "n_removed_missing_timestamp": self.n_removed_missing_timestamp,
            "n_removed_invalid_rr": self.n_removed_invalid_rr,
            "n_removed_overlapping": self.n_removed_overlapping,
            "n_final": self.n_final,
            "retention_rate": self.n_final / self.n_original if self.n_original > 0 else 0,
            "n_patients_affected": len(self.patients_with_removals),
        }


class EpisodeCleaner:
    """Clean and validate AF episode data."""

    def __init__(
        self,
        min_duration_minutes: float = DEFAULT_CONFIG.min_episode_duration_minutes,
        max_duration_minutes: float = DEFAULT_CONFIG.max_episode_duration_minutes,
        min_rr_interval_msec: float = DEFAULT_CONFIG.min_rr_interval_msec,
        max_rr_interval_msec: float = DEFAULT_CONFIG.max_rr_interval_msec,
        remove_zero_duration: bool = True,
        remove_overlapping: bool = False,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        rr_column: str = "AF_MEAN_RR_INTERVAL_msec",
        start_column: str = "time_start_ep",
    ):
        """
        Initialize the episode cleaner.

        Args:
            min_duration_minutes: Minimum valid episode duration.
            max_duration_minutes: Maximum valid episode duration (7 days default).
            min_rr_interval_msec: Minimum physiological RR interval (200ms = 300bpm).
            max_rr_interval_msec: Maximum physiological RR interval (2000ms = 30bpm).
            remove_zero_duration: Whether to remove zero-duration episodes.
            remove_overlapping: Whether to detect and remove overlapping episodes.
            id_column: Name of patient ID column.
            duration_column: Name of episode duration column.
            rr_column: Name of RR interval column.
            start_column: Name of episode start time column.
        """
        self.min_duration_minutes = min_duration_minutes
        self.max_duration_minutes = max_duration_minutes
        self.min_rr_interval_msec = min_rr_interval_msec
        self.max_rr_interval_msec = max_rr_interval_msec
        self.remove_zero_duration = remove_zero_duration
        self.remove_overlapping = remove_overlapping
        self.id_column = id_column
        self.duration_column = duration_column
        self.rr_column = rr_column
        self.start_column = start_column

    def clean(
        self, df: pd.DataFrame, verbose: bool = True
    ) -> Tuple[pd.DataFrame, EpisodeCleaningReport]:
        """
        Apply all cleaning steps and return cleaned data with report.

        Args:
            df: DataFrame with episode data.
            verbose: Whether to print progress messages.

        Returns:
            Tuple of (cleaned DataFrame, cleaning report).
        """
        report = EpisodeCleaningReport(n_original=len(df))
        df = df.copy()
        removed_patients = set()

        # Track original indices for each patient
        original_counts = df.groupby(self.id_column).size()

        # 1. Remove negative durations
        if self.duration_column in df.columns:
            mask = df[self.duration_column] < 0
            n_neg = mask.sum()
            if n_neg > 0:
                removed_patients.update(df.loc[mask, self.id_column].unique())
                df = df[~mask]
                report.n_removed_negative_duration = n_neg
                if verbose:
                    print(f"Removed {n_neg} episodes with negative duration")

        # 2. Remove zero durations (optional)
        if self.remove_zero_duration and self.duration_column in df.columns:
            mask = df[self.duration_column] == 0
            n_zero = mask.sum()
            if n_zero > 0:
                removed_patients.update(df.loc[mask, self.id_column].unique())
                df = df[~mask]
                report.n_removed_zero_duration = n_zero
                if verbose:
                    print(f"Removed {n_zero} episodes with zero duration")

        # 3. Remove short durations (below minimum)
        if self.duration_column in df.columns and self.min_duration_minutes > 0:
            mask = df[self.duration_column] < self.min_duration_minutes
            n_short = mask.sum()
            if n_short > 0:
                removed_patients.update(df.loc[mask, self.id_column].unique())
                df = df[~mask]
                report.n_removed_short_duration = n_short
                if verbose:
                    print(
                        f"Removed {n_short} episodes with duration < {self.min_duration_minutes} minutes"
                    )

        # 4. Remove excessive durations
        if self.duration_column in df.columns:
            mask = df[self.duration_column] > self.max_duration_minutes
            n_excess = mask.sum()
            if n_excess > 0:
                removed_patients.update(df.loc[mask, self.id_column].unique())
                df = df[~mask]
                report.n_removed_excessive_duration = n_excess
                if verbose:
                    print(
                        f"Removed {n_excess} episodes with duration > {self.max_duration_minutes} minutes"
                    )

        # 5. Remove missing timestamps
        if self.start_column in df.columns:
            mask = df[self.start_column].isna()
            n_missing = mask.sum()
            if n_missing > 0:
                removed_patients.update(df.loc[mask, self.id_column].unique())
                df = df[~mask]
                report.n_removed_missing_timestamp = n_missing
                if verbose:
                    print(f"Removed {n_missing} episodes with missing timestamp")

        # 6. Remove invalid RR intervals (if column exists)
        if self.rr_column in df.columns:
            # Convert to numeric if needed (handle string/object columns)
            if not pd.api.types.is_numeric_dtype(df[self.rr_column]):
                df = df.copy()
                df[self.rr_column] = pd.to_numeric(df[self.rr_column], errors='coerce')

            # Only check non-null values
            rr_valid = df[self.rr_column].notna()
            rr_values = df[self.rr_column].astype(float)
            mask_invalid = rr_valid & (
                (rr_values < self.min_rr_interval_msec)
                | (rr_values > self.max_rr_interval_msec)
            )
            n_invalid_rr = mask_invalid.sum()
            if n_invalid_rr > 0:
                removed_patients.update(df.loc[mask_invalid, self.id_column].unique())
                df = df[~mask_invalid]
                report.n_removed_invalid_rr = n_invalid_rr
                if verbose:
                    print(f"Removed {n_invalid_rr} episodes with invalid RR interval")

        # 7. Handle overlapping episodes (optional, expensive)
        if self.remove_overlapping:
            df, n_overlapping = self._remove_overlapping_episodes(df)
            report.n_removed_overlapping = n_overlapping
            if verbose and n_overlapping > 0:
                print(f"Removed {n_overlapping} overlapping episodes")

        report.n_final = len(df)
        report.patients_with_removals = list(removed_patients)

        if verbose:
            print(f"\nFinal: {report.n_final:,} episodes ({100*report.n_final/report.n_original:.1f}% retained)")

        return df, report

    def _remove_overlapping_episodes(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int]:
        """
        Detect and remove overlapping episodes within each patient.

        Episodes are considered overlapping if one starts before the previous ends.

        Args:
            df: Episode DataFrame.

        Returns:
            Tuple of (cleaned DataFrame, number removed).
        """
        if self.start_column not in df.columns or self.duration_column not in df.columns:
            return df, 0

        df = df.sort_values([self.id_column, self.start_column])

        # Calculate end time
        df["_end_time"] = df[self.start_column] + pd.to_timedelta(
            df[self.duration_column], unit="min"
        )

        # Find overlapping (within each patient)
        overlap_mask = pd.Series(False, index=df.index)

        for pid, group in df.groupby(self.id_column):
            if len(group) < 2:
                continue

            # Check if current episode starts before previous ends
            starts = group[self.start_column].values
            ends = group["_end_time"].values

            for i in range(1, len(group)):
                if starts[i] < ends[i - 1]:
                    overlap_mask.iloc[group.index[i]] = True

        df = df.drop(columns=["_end_time"])
        n_removed = overlap_mask.sum()

        return df[~overlap_mask], n_removed

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate episode data without removing rows.

        Returns DataFrame with validation flags added.

        Args:
            df: Episode DataFrame.

        Returns:
            DataFrame with added validation columns.
        """
        df = df.copy()

        # Duration flags
        if self.duration_column in df.columns:
            df["_valid_duration"] = (
                (df[self.duration_column] > self.min_duration_minutes)
                & (df[self.duration_column] <= self.max_duration_minutes)
            )
        else:
            df["_valid_duration"] = True

        # Timestamp flags
        if self.start_column in df.columns:
            df["_valid_timestamp"] = df[self.start_column].notna()
        else:
            df["_valid_timestamp"] = True

        # RR interval flags
        if self.rr_column in df.columns:
            df["_valid_rr"] = (
                df[self.rr_column].isna()  # Missing is OK
                | (
                    (df[self.rr_column] >= self.min_rr_interval_msec)
                    & (df[self.rr_column] <= self.max_rr_interval_msec)
                )
            )
        else:
            df["_valid_rr"] = True

        # Overall validity
        df["_valid_overall"] = (
            df["_valid_duration"] & df["_valid_timestamp"] & df["_valid_rr"]
        )

        return df

    def get_invalid_episodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only invalid episodes for inspection.

        Args:
            df: Episode DataFrame.

        Returns:
            DataFrame with only invalid episodes.
        """
        validated = self.validate(df)
        return validated[~validated["_valid_overall"]]

    def summary_by_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate episode summary statistics per patient.

        Args:
            df: Episode DataFrame.

        Returns:
            DataFrame with per-patient summary statistics.
        """
        validated = self.validate(df)

        summary = validated.groupby(self.id_column).agg(
            n_episodes=(self.duration_column, "count"),
            n_valid=("_valid_overall", "sum"),
            n_invalid=("_valid_overall", lambda x: (~x).sum()),
            total_duration=(self.duration_column, "sum"),
            mean_duration=(self.duration_column, "mean"),
            max_duration=(self.duration_column, "max"),
        )

        summary["pct_valid"] = 100 * summary["n_valid"] / summary["n_episodes"]

        return summary.reset_index()
