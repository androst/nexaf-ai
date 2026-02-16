"""
Time alignment utilities for AF episode data.

Aligns episode timestamps relative to ILR implant date.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from ..config import DEFAULT_CONFIG


class TimeAligner:
    """Align episode times relative to implant date."""

    def __init__(
        self,
        time_unit: str = "days",
        exclude_pre_implant: bool = DEFAULT_CONFIG.exclude_pre_implant,
        max_follow_up_days: Optional[int] = DEFAULT_CONFIG.max_follow_up_days,
        id_column: str = "ID",
        implant_column: str = "date_ilr_implant",
        start_column: str = "time_start_ep",
    ):
        """
        Initialize the time aligner.

        Args:
            time_unit: Primary time unit for output ('days', 'hours', 'weeks', 'months').
            exclude_pre_implant: Whether to exclude episodes before implant date.
            max_follow_up_days: Maximum follow-up period in days (None for unlimited).
            id_column: Name of patient ID column.
            implant_column: Name of implant date column.
            start_column: Name of episode start time column.
        """
        self.time_unit = time_unit
        self.exclude_pre_implant = exclude_pre_implant
        self.max_follow_up_days = max_follow_up_days
        self.id_column = id_column
        self.implant_column = implant_column
        self.start_column = start_column

    def align(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Add relative time columns to episode data.

        Adds columns:
        - time_since_implant: Timedelta since implant
        - days_since_implant: Days since implant (float)
        - weeks_since_implant: Weeks since implant (float)
        - months_since_implant: Approximate months since implant (float)

        Args:
            df: Episode DataFrame with implant date and episode start columns.
            verbose: Whether to print summary information.

        Returns:
            DataFrame with added time columns and filtered rows.
        """
        df = df.copy()
        n_original = len(df)

        # Ensure datetime types
        if not pd.api.types.is_datetime64_any_dtype(df[self.implant_column]):
            df[self.implant_column] = pd.to_datetime(df[self.implant_column], errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(df[self.start_column]):
            df[self.start_column] = pd.to_datetime(df[self.start_column], errors="coerce")

        # Calculate time since implant
        df["time_since_implant"] = df[self.start_column] - df[self.implant_column]

        # Convert to numeric units
        df["days_since_implant"] = df["time_since_implant"].dt.total_seconds() / 86400
        df["weeks_since_implant"] = df["days_since_implant"] / 7
        df["months_since_implant"] = df["days_since_implant"] / 30.44  # Average month

        # Exclude pre-implant episodes
        if self.exclude_pre_implant:
            pre_implant_mask = df["days_since_implant"] < 0
            n_pre = pre_implant_mask.sum()
            df = df[~pre_implant_mask]
            if verbose and n_pre > 0:
                print(f"Excluded {n_pre} episodes occurring before implant")

        # Exclude episodes beyond max follow-up
        if self.max_follow_up_days is not None:
            beyond_mask = df["days_since_implant"] > self.max_follow_up_days
            n_beyond = beyond_mask.sum()
            df = df[~beyond_mask]
            if verbose and n_beyond > 0:
                print(f"Excluded {n_beyond} episodes beyond {self.max_follow_up_days} days follow-up")

        if verbose:
            print(f"Aligned {len(df):,} episodes ({100*len(df)/n_original:.1f}% of original)")

        return df

    def create_time_windows(
        self,
        df: pd.DataFrame,
        windows: Optional[List[Tuple[int, int]]] = None,
    ) -> pd.DataFrame:
        """
        Assign episodes to predefined time windows.

        Args:
            df: Episode DataFrame with days_since_implant column.
            windows: List of (start_day, end_day) tuples defining windows.
                    If None, uses default windows from config.

        Returns:
            DataFrame with added time_window column.
        """
        if windows is None:
            windows = DEFAULT_CONFIG.time_windows

        df = df.copy()

        # Ensure alignment has been done
        if "days_since_implant" not in df.columns:
            df = self.align(df, verbose=False)

        # Create window labels
        def assign_window(days):
            for start, end in windows:
                if start <= days < end:
                    return f"{start}-{end}d"
            return "other"

        df["time_window"] = df["days_since_implant"].apply(assign_window)

        return df

    def get_follow_up_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get follow-up duration summary per patient.

        Args:
            df: Episode DataFrame with days_since_implant column.

        Returns:
            DataFrame with follow-up statistics per patient.
        """
        if "days_since_implant" not in df.columns:
            df = self.align(df, verbose=False)

        summary = df.groupby(self.id_column).agg(
            first_episode_day=("days_since_implant", "min"),
            last_episode_day=("days_since_implant", "max"),
            n_episodes=("days_since_implant", "count"),
            follow_up_span=("days_since_implant", lambda x: x.max() - x.min()),
        ).reset_index()

        summary["follow_up_days"] = summary["last_episode_day"]

        return summary

    def resample_to_daily(
        self,
        df: pd.DataFrame,
        duration_column: str = "af_episode_minutes",
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """
        Resample episode data to daily resolution.

        Args:
            df: Episode DataFrame with time alignment.
            duration_column: Column to aggregate.
            agg_func: Aggregation function ('sum', 'count', 'mean').

        Returns:
            DataFrame with daily aggregated data per patient.
        """
        if "days_since_implant" not in df.columns:
            df = self.align(df, verbose=False)

        df = df.copy()
        df["day"] = df["days_since_implant"].astype(int)

        # Aggregate by patient and day
        if agg_func == "sum":
            daily = df.groupby([self.id_column, "day"])[duration_column].sum()
        elif agg_func == "count":
            daily = df.groupby([self.id_column, "day"])[duration_column].count()
        elif agg_func == "mean":
            daily = df.groupby([self.id_column, "day"])[duration_column].mean()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

        return daily.reset_index()

    def create_cumulative_burden(
        self,
        df: pd.DataFrame,
        duration_column: str = "af_episode_minutes",
    ) -> pd.DataFrame:
        """
        Create cumulative AF burden time series per patient.

        Args:
            df: Episode DataFrame with time alignment.
            duration_column: Column containing episode durations.

        Returns:
            DataFrame with cumulative burden per patient.
        """
        if "days_since_implant" not in df.columns:
            df = self.align(df, verbose=False)

        # Sort by patient and time
        df = df.sort_values([self.id_column, "days_since_implant"])

        # Calculate cumulative sum within each patient
        df = df.copy()
        df["cumulative_burden_minutes"] = df.groupby(self.id_column)[duration_column].cumsum()
        df["cumulative_burden_hours"] = df["cumulative_burden_minutes"] / 60
        df["cumulative_burden_days"] = df["cumulative_burden_minutes"] / 1440

        return df

    def normalize_time(
        self,
        df: pd.DataFrame,
        reference_period: int = 365,
    ) -> pd.DataFrame:
        """
        Normalize time to 0-1 scale based on reference period.

        Useful for comparing trajectories across patients with different follow-up.

        Args:
            df: Episode DataFrame with days_since_implant column.
            reference_period: Reference period in days for normalization.

        Returns:
            DataFrame with added normalized_time column.
        """
        if "days_since_implant" not in df.columns:
            df = self.align(df, verbose=False)

        df = df.copy()
        df["normalized_time"] = df["days_since_implant"] / reference_period

        return df
