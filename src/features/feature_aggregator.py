"""
Feature aggregation module.

Combines all feature extractors into a unified pipeline.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .burden_features import BurdenFeatureExtractor
from .episode_features import EpisodePatternExtractor
from .temporal_features import TemporalPatternExtractor
from .rr_features import RRIntervalExtractor
from .trajectory_features import TrajectoryShapeExtractor


# Curated feature set: interpretable, non-redundant features for clustering
CURATED_FEATURES = {
    # === BURDEN (PRIMARY) ===
    'total_af_hours': 'Total hours in AF over follow-up period',
    'af_burden_percent': 'Percentage of time in AF (burden / total time)',

    # === EPISODE COUNT & PATTERN ===
    'n_episodes': 'Total number of AF episodes',
    'mean_episode_duration': 'Average episode duration (minutes)',
    'max_episode_duration': 'Longest episode duration (minutes)',
    'cv_episode_duration': 'Coefficient of variation of episode duration (variability)',

    # === EPISODE REGULARITY ===
    'burstiness_index': 'Temporal clustering of episodes (-1=periodic, 0=random, 1=bursty)',

    # === TEMPORAL PATTERN ===
    'pct_daytime_episodes': 'Percentage of episodes starting during daytime',
    'hourly_entropy': 'Spread of episodes across hours (higher=more distributed)',

    # === HEART RATE / RATE CONTROL ===
    'mean_ventricular_rate': 'Average heart rate during AF episodes (bpm)',
    'pct_rapid_af': 'Percentage of episodes with rapid ventricular response (>110 bpm)',
    'pct_controlled_af': 'Percentage of episodes with controlled rate (60-110 bpm)',

    # === TRAJECTORY / PROGRESSION ===
    'trajectory_slope': 'Rate of burden accumulation over time (min/day)',
    'longest_plateau_episodes': 'Longest period with minimal AF (stability indicator)',
}


class FeatureAggregator:
    """Aggregate all feature extractors into unified pipeline."""

    def __init__(
        self,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        time_column: str = "days_since_implant",
        start_column: str = "time_start_ep",
        rr_column: str = "AF_MEAN_RR_INTERVAL_msec",
        rate_column: str = "AF_MEAN_RR_RATE_bpm",
        daytime_column: str = "episode_start_during_day",
    ):
        """
        Initialize the feature aggregator with all extractors.

        Args:
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            time_column: Time since implant column name.
            start_column: Episode start timestamp column name.
            rr_column: RR interval column name.
            rate_column: Heart rate column name.
            daytime_column: Day/night indicator column name.
        """
        self.id_column = id_column

        self.extractors = {
            "burden": BurdenFeatureExtractor(
                id_column=id_column,
                duration_column=duration_column,
                time_column=time_column,
            ),
            "episode": EpisodePatternExtractor(
                id_column=id_column,
                duration_column=duration_column,
                time_column=time_column,
                start_column=start_column,
            ),
            "temporal": TemporalPatternExtractor(
                id_column=id_column,
                duration_column=duration_column,
                start_column=start_column,
                daytime_column=daytime_column,
            ),
            "rr": RRIntervalExtractor(
                id_column=id_column,
                rr_column=rr_column,
                rate_column=rate_column,
                duration_column=duration_column,
            ),
            "trajectory": TrajectoryShapeExtractor(
                id_column=id_column,
                duration_column=duration_column,
                time_column=time_column,
            ),
        }

    def extract_all(
        self,
        episode_df: pd.DataFrame,
        extractors: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract all features and combine into single DataFrame.

        Args:
            episode_df: Episode-level DataFrame with time alignment.
            extractors: List of extractor names to use.
                       If None, uses all extractors.
            verbose: Whether to print progress.

        Returns:
            Patient-level DataFrame with all features.
        """
        if extractors is None:
            extractors = list(self.extractors.keys())

        # Extract features from each extractor
        feature_dfs = {}
        for name in extractors:
            if name not in self.extractors:
                print(f"Warning: Unknown extractor '{name}', skipping")
                continue

            if verbose:
                print(f"Extracting {name} features...")

            extractor = self.extractors[name]
            feature_dfs[name] = extractor.extract(episode_df)

        if not feature_dfs:
            raise ValueError("No features extracted")

        # Merge all feature DataFrames on patient ID
        result = feature_dfs[extractors[0]]
        for name in extractors[1:]:
            if name in feature_dfs:
                # Drop duplicate ID column before merge
                other_df = feature_dfs[name].drop(columns=[self.id_column], errors="ignore")
                # Get patient IDs from current result
                other_df[self.id_column] = feature_dfs[name][self.id_column]
                result = result.merge(
                    other_df,
                    on=self.id_column,
                    how="outer",
                    suffixes=("", f"_{name}"),
                )

        if verbose:
            print(f"Extracted {len(result.columns) - 1} features for {len(result)} patients")

        return result

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Return feature names grouped by extractor category.

        Returns:
            Dictionary mapping extractor name to list of feature names.
        """
        return {
            name: extractor.get_feature_names()
            for name, extractor in self.extractors.items()
        }

    def get_all_feature_names(self) -> List[str]:
        """
        Return flat list of all feature names.

        Returns:
            List of all feature names (excluding ID column).
        """
        all_features = []
        for extractor in self.extractors.values():
            all_features.extend(extractor.get_feature_names())
        return list(set(all_features))

    def describe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics for all features.

        Args:
            df: Feature DataFrame.

        Returns:
            DataFrame with summary statistics per feature.
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.id_column]

        stats_list = []
        for col in numeric_cols:
            series = df[col]
            stats_list.append({
                "feature": col,
                "count": series.count(),
                "missing": series.isna().sum(),
                "missing_pct": 100 * series.isna().mean(),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p25": series.quantile(0.25),
                "median": series.median(),
                "p75": series.quantile(0.75),
                "max": series.max(),
            })

        return pd.DataFrame(stats_list)

    def get_feature_correlations(
        self,
        df: pd.DataFrame,
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """
        Find highly correlated feature pairs.

        Args:
            df: Feature DataFrame.
            threshold: Correlation threshold for flagging.

        Returns:
            DataFrame with correlated feature pairs.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.id_column]

        corr_matrix = df[numeric_cols].corr()

        # Find pairs above threshold
        pairs = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) >= threshold:
                        pairs.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                        })

        return pd.DataFrame(pairs).sort_values("correlation", ascending=False, key=abs)

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = "variance",
        n_features: Optional[int] = None,
        variance_threshold: float = 0.01,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select subset of features based on specified method.

        Args:
            df: Feature DataFrame.
            method: Selection method ('variance', 'correlation', 'curated', 'all').
            n_features: Target number of features (for some methods).
            variance_threshold: Minimum variance to keep (for variance method).

        Returns:
            Tuple of (filtered DataFrame, list of selected feature names).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != self.id_column]

        if method == "all":
            return df, feature_cols

        elif method == "curated":
            # Use the curated feature set (interpretable, non-redundant)
            selected = [f for f in CURATED_FEATURES.keys() if f in feature_cols]
            if len(selected) < len(CURATED_FEATURES):
                missing = [f for f in CURATED_FEATURES.keys() if f not in feature_cols]
                print(f"Warning: {len(missing)} curated features not available: {missing}")

        elif method == "variance":
            # Remove low-variance features
            variances = df[feature_cols].var()
            # Normalize by mean to get coefficient of variation squared
            means = df[feature_cols].mean().abs()
            cv_squared = variances / (means**2 + 1e-10)
            selected = cv_squared[cv_squared > variance_threshold].index.tolist()

        elif method == "correlation":
            # Remove highly correlated features (keep one from each group)
            corr_matrix = df[feature_cols].corr().abs()
            selected = []
            remaining = set(feature_cols)

            for col in feature_cols:
                if col in remaining:
                    selected.append(col)
                    # Remove correlated features
                    correlated = corr_matrix.index[corr_matrix[col] > 0.8].tolist()
                    remaining -= set(correlated)

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Limit to n_features if specified
        if n_features is not None and len(selected) > n_features:
            selected = selected[:n_features]

        return df[[self.id_column] + selected], selected

    @staticmethod
    def get_curated_features() -> Dict[str, str]:
        """
        Return the curated feature set with descriptions.

        The curated set contains ~14 interpretable, non-redundant features
        suitable for clinical phenotyping.

        Returns:
            Dictionary mapping feature name to description.
        """
        return CURATED_FEATURES.copy()

    @staticmethod
    def get_curated_feature_names() -> List[str]:
        """
        Return list of curated feature names.

        Returns:
            List of feature names in the curated set.
        """
        return list(CURATED_FEATURES.keys())


def create_feature_pipeline(
    episode_df: pd.DataFrame,
    id_column: str = "ID",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, FeatureAggregator]:
    """
    Convenience function to create features with default settings.

    Args:
        episode_df: Episode DataFrame with time alignment.
        id_column: Patient ID column name.
        verbose: Whether to print progress.

    Returns:
        Tuple of (feature DataFrame, FeatureAggregator instance).
    """
    aggregator = FeatureAggregator(id_column=id_column)
    features = aggregator.extract_all(episode_df, verbose=verbose)
    return features, aggregator
