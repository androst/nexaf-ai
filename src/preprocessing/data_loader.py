"""
Data loading utilities for AF trajectory analysis.

Handles loading of SPSS files with proper type conversion and metadata extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import pyreadstat

from ..config import (
    EPISODE_FILE,
    BURDEN_FILE,
    BASELINE_FILE,
    VALIDATION_FILE,
    EPISODE_COLUMNS,
    OUTCOME_COLUMNS,
)


@dataclass
class DatasetConfig:
    """Configuration for dataset file paths."""

    episode_file: Path = EPISODE_FILE
    burden_file: Path = BURDEN_FILE
    baseline_file: Path = BASELINE_FILE
    validation_file: Optional[Path] = VALIDATION_FILE


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""

    n_rows: int
    n_columns: int
    columns: list
    memory_mb: float
    variable_labels: Dict[str, str]
    value_labels: Dict[str, Dict[Any, str]]


class AFDataLoader:
    """Load and validate AF study datasets from SPSS files."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize the data loader.

        Args:
            config: Dataset configuration with file paths.
                   If None, uses default paths from config module.
        """
        self.config = config or DatasetConfig()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, DatasetInfo] = {}

    def load_episodes(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load episode-level data with ~170K AF episodes.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with episode data including timestamps and durations.
        """
        cache_key = "episodes"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        df, meta = pyreadstat.read_sav(str(self.config.episode_file))

        # Convert date columns to proper datetime
        date_columns = ["date_ilr_implant", "date_randomization"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert episode timestamps
        if "time_start_ep" in df.columns:
            df["time_start_ep"] = pd.to_datetime(df["time_start_ep"], errors="coerce")
        if "time_stop_ep" in df.columns:
            df["time_stop_ep"] = pd.to_datetime(df["time_stop_ep"], errors="coerce")

        # Store metadata
        self._metadata[cache_key] = DatasetInfo(
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            variable_labels=meta.column_names_to_labels,
            value_labels=meta.variable_value_labels,
        )

        self._cache[cache_key] = df
        return df

    def load_burden(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load patient-level burden and outcome data.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with patient outcomes and AF burden measures.
        """
        cache_key = "burden"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        df, meta = pyreadstat.read_sav(str(self.config.burden_file))

        # Store metadata
        self._metadata[cache_key] = DatasetInfo(
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            variable_labels=meta.column_names_to_labels,
            value_labels=meta.variable_value_labels,
        )

        self._cache[cache_key] = df
        return df

    def load_baseline(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load baseline clinical data with 714+ variables.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with baseline patient characteristics.
        """
        cache_key = "baseline"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        df, meta = pyreadstat.read_sav(str(self.config.baseline_file))

        # Store metadata
        self._metadata[cache_key] = DatasetInfo(
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            variable_labels=meta.column_names_to_labels,
            value_labels=meta.variable_value_labels,
        )

        self._cache[cache_key] = df
        return df

    def load_validation(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load validation dataset with manually validated episodes.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with validation flags for AF episodes.
        """
        if self.config.validation_file is None:
            raise ValueError("Validation file path not configured")

        cache_key = "validation"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        df, meta = pyreadstat.read_sav(str(self.config.validation_file))

        # Convert timestamp
        if "time_start_ep" in df.columns:
            df["time_start_ep"] = pd.to_datetime(df["time_start_ep"], errors="coerce")

        # Store metadata
        self._metadata[cache_key] = DatasetInfo(
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            variable_labels=meta.column_names_to_labels,
            value_labels=meta.variable_value_labels,
        )

        self._cache[cache_key] = df
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets at once.

        Returns:
            Dictionary with all loaded DataFrames.
        """
        return {
            "episodes": self.load_episodes(),
            "burden": self.load_burden(),
            "baseline": self.load_baseline(),
            "validation": self.load_validation(),
        }

    def get_metadata(self, dataset: str) -> DatasetInfo:
        """
        Get metadata for a loaded dataset.

        Args:
            dataset: Name of the dataset ('episodes', 'burden', 'baseline', 'validation')

        Returns:
            DatasetInfo with row count, columns, variable labels, etc.
        """
        if dataset not in self._metadata:
            # Load the dataset to generate metadata
            load_methods = {
                "episodes": self.load_episodes,
                "burden": self.load_burden,
                "baseline": self.load_baseline,
                "validation": self.load_validation,
            }
            if dataset not in load_methods:
                raise ValueError(f"Unknown dataset: {dataset}")
            load_methods[dataset]()

        return self._metadata[dataset]

    def get_outcome_subset(self, df_burden: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get a subset of burden data with key outcome columns.

        Args:
            df_burden: Burden DataFrame, or None to load fresh.

        Returns:
            DataFrame with ID and key outcome columns only.
        """
        if df_burden is None:
            df_burden = self.load_burden()

        outcome_cols = [
            "ID",
            "BL_AF_type",
            "BL_Afeqt_symptoms_score",
            "Six_months_Afeqt_symptoms_score",
            "Post_Afeqt_symptoms_score",
            "Post_CVD_hospi",
            "Post_hospi",
            "Post_AF_hosp",
            "Post_non_AF_hospi",
            "Post_trauma_hospi",
            "Post_hosp_komm",
        ]

        available_cols = [c for c in outcome_cols if c in df_burden.columns]
        return df_burden[available_cols].copy()

    def validate_ids(self) -> Dict[str, Any]:
        """
        Check ID consistency across all datasets.

        Returns:
            Dictionary with validation results including overlaps and mismatches.
        """
        episodes_df = self.load_episodes()
        burden_df = self.load_burden()
        baseline_df = self.load_baseline()

        episode_ids = set(episodes_df["ID"].unique())
        burden_ids = set(burden_df["ID"].unique())
        baseline_ids = set(baseline_df["ID"].unique())

        return {
            "n_episode_patients": len(episode_ids),
            "n_burden_patients": len(burden_ids),
            "n_baseline_patients": len(baseline_ids),
            "in_all_three": len(episode_ids & burden_ids & baseline_ids),
            "in_episodes_only": len(episode_ids - burden_ids - baseline_ids),
            "in_burden_only": len(burden_ids - episode_ids - baseline_ids),
            "in_baseline_only": len(baseline_ids - episode_ids - burden_ids),
            "episode_burden_overlap": len(episode_ids & burden_ids),
            "episode_baseline_overlap": len(episode_ids & baseline_ids),
            "burden_baseline_overlap": len(burden_ids & baseline_ids),
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def summary(self) -> pd.DataFrame:
        """
        Generate summary table of all datasets.

        Returns:
            DataFrame with dataset summaries.
        """
        summaries = []
        for name in ["episodes", "burden", "baseline", "validation"]:
            try:
                meta = self.get_metadata(name)
                summaries.append(
                    {
                        "dataset": name,
                        "rows": meta.n_rows,
                        "columns": meta.n_columns,
                        "memory_mb": round(meta.memory_mb, 2),
                    }
                )
            except Exception as e:
                summaries.append(
                    {
                        "dataset": name,
                        "rows": None,
                        "columns": None,
                        "memory_mb": None,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(summaries)
