"""
Data integration utilities for combining episode, burden, and baseline data.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


class DataIntegrator:
    """Integrate all data sources at patient level."""

    def __init__(self, id_column: str = "ID"):
        """
        Initialize the data integrator.

        Args:
            id_column: Name of the patient ID column used for merging.
        """
        self.id_column = id_column

    def integrate_episode_with_outcomes(
        self,
        episodes: pd.DataFrame,
        burden: pd.DataFrame,
        outcome_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Merge episode data with patient-level outcomes.

        Args:
            episodes: Episode-level DataFrame.
            burden: Patient-level burden/outcome DataFrame.
            outcome_columns: Specific columns to include from burden.
                           If None, includes all columns.

        Returns:
            Episode DataFrame with outcome columns added.
        """
        if outcome_columns is not None:
            # Always include ID column
            cols_to_merge = [self.id_column] + [
                c for c in outcome_columns if c != self.id_column and c in burden.columns
            ]
            burden_subset = burden[cols_to_merge]
        else:
            burden_subset = burden

        return episodes.merge(burden_subset, on=self.id_column, how="left")

    def integrate_episode_with_baseline(
        self,
        episodes: pd.DataFrame,
        baseline: pd.DataFrame,
        baseline_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Merge episode data with baseline patient characteristics.

        Args:
            episodes: Episode-level DataFrame.
            baseline: Baseline characteristics DataFrame.
            baseline_columns: Specific columns to include from baseline.
                            If None, includes all columns.

        Returns:
            Episode DataFrame with baseline columns added.
        """
        if baseline_columns is not None:
            cols_to_merge = [self.id_column] + [
                c for c in baseline_columns if c != self.id_column and c in baseline.columns
            ]
            baseline_subset = baseline[cols_to_merge]
        else:
            baseline_subset = baseline

        return episodes.merge(baseline_subset, on=self.id_column, how="left")

    def create_patient_dataset(
        self,
        episode_features: pd.DataFrame,
        burden: pd.DataFrame,
        baseline: Optional[pd.DataFrame] = None,
        outcome_columns: Optional[List[str]] = None,
        baseline_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create unified patient-level dataset for analysis.

        Combines extracted features from episodes with outcomes and baseline data.

        Args:
            episode_features: Patient-level features extracted from episodes.
            burden: Patient-level burden/outcome DataFrame.
            baseline: Optional baseline characteristics DataFrame.
            outcome_columns: Specific outcome columns to include.
            baseline_columns: Specific baseline columns to include.

        Returns:
            Unified patient-level DataFrame.
        """
        # Start with episode features
        df = episode_features.copy()

        # Add outcomes
        if outcome_columns is None:
            outcome_columns = [
                "BL_AF_type",
                "BL_Afeqt_symptoms_score",
                "Six_months_Afeqt_symptoms_score",
                "Post_Afeqt_symptoms_score",
                "Post_AF_hosp",
                "Post_hospi",
                "Post_CVD_hospi",
            ]

        available_outcome_cols = [
            c for c in outcome_columns if c in burden.columns
        ]
        if available_outcome_cols:
            burden_subset = burden[[self.id_column] + available_outcome_cols]
            df = df.merge(burden_subset, on=self.id_column, how="left")

        # Add baseline if provided
        if baseline is not None and baseline_columns is not None:
            available_baseline_cols = [
                c for c in baseline_columns if c in baseline.columns
            ]
            if available_baseline_cols:
                baseline_subset = baseline[[self.id_column] + available_baseline_cols]
                df = df.merge(baseline_subset, on=self.id_column, how="left")

        return df

    def validate_integration(
        self,
        episodes: pd.DataFrame,
        burden: pd.DataFrame,
        baseline: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Validate data integration and check for ID mismatches.

        Args:
            episodes: Episode DataFrame.
            burden: Burden DataFrame.
            baseline: Optional baseline DataFrame.

        Returns:
            Dictionary with validation results.
        """
        episode_ids = set(episodes[self.id_column].unique())
        burden_ids = set(burden[self.id_column].unique())

        results = {
            "n_episode_patients": len(episode_ids),
            "n_burden_patients": len(burden_ids),
            "overlap_episode_burden": len(episode_ids & burden_ids),
            "in_episodes_not_burden": len(episode_ids - burden_ids),
            "in_burden_not_episodes": len(burden_ids - episode_ids),
            "ids_in_episodes_not_burden": list(episode_ids - burden_ids),
            "ids_in_burden_not_episodes": list(burden_ids - episode_ids),
        }

        if baseline is not None:
            baseline_ids = set(baseline[self.id_column].unique())
            results.update({
                "n_baseline_patients": len(baseline_ids),
                "overlap_all_three": len(episode_ids & burden_ids & baseline_ids),
                "in_episodes_not_baseline": len(episode_ids - baseline_ids),
                "in_baseline_not_episodes": len(baseline_ids - episode_ids),
            })

        return results

    def get_complete_cases(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
    ) -> pd.DataFrame:
        """
        Filter to patients with complete data for required columns.

        Args:
            df: Patient-level DataFrame.
            required_columns: Columns that must be non-null.

        Returns:
            DataFrame with only complete cases.
        """
        mask = df[required_columns].notna().all(axis=1)
        return df[mask].copy()

    def summarize_missingness(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Summarize missing data patterns.

        Args:
            df: DataFrame to analyze.
            columns: Specific columns to check. If None, checks all.

        Returns:
            DataFrame with missingness statistics per column.
        """
        if columns is None:
            columns = df.columns.tolist()

        summary = []
        for col in columns:
            if col not in df.columns:
                continue
            n_missing = df[col].isna().sum()
            n_total = len(df)
            summary.append({
                "column": col,
                "n_missing": n_missing,
                "n_present": n_total - n_missing,
                "pct_missing": 100 * n_missing / n_total,
            })

        return pd.DataFrame(summary).sort_values("pct_missing", ascending=False)

    def create_analysis_cohort(
        self,
        episodes: pd.DataFrame,
        burden: pd.DataFrame,
        baseline: Optional[pd.DataFrame] = None,
        min_episodes: int = 1,
        require_outcome: bool = True,
        outcome_column: str = "Post_AF_hosp",
    ) -> Dict[str, Any]:
        """
        Create analysis cohort with specified criteria.

        Args:
            episodes: Episode DataFrame.
            burden: Burden DataFrame.
            baseline: Optional baseline DataFrame.
            min_episodes: Minimum number of episodes required.
            require_outcome: Whether to require non-null outcome.
            outcome_column: Outcome column to check.

        Returns:
            Dictionary with:
            - 'episodes': Filtered episode DataFrame
            - 'patients': Patient IDs in cohort
            - 'criteria': Applied selection criteria
            - 'attrition': Attrition flow
        """
        attrition = []

        # Start with all patients
        all_episode_patients = set(episodes[self.id_column].unique())
        attrition.append(("All patients with episodes", len(all_episode_patients)))

        # Apply minimum episodes criterion
        episode_counts = episodes.groupby(self.id_column).size()
        patients_with_min_eps = set(episode_counts[episode_counts >= min_episodes].index)
        attrition.append((f"Patients with >= {min_episodes} episodes", len(patients_with_min_eps)))

        # Apply outcome availability criterion
        if require_outcome:
            patients_with_outcome = set(
                burden.loc[burden[outcome_column].notna(), self.id_column].unique()
            )
            final_patients = patients_with_min_eps & patients_with_outcome
            attrition.append(
                (f"Patients with {outcome_column} available", len(final_patients))
            )
        else:
            final_patients = patients_with_min_eps

        # Filter episodes
        filtered_episodes = episodes[episodes[self.id_column].isin(final_patients)]

        return {
            "episodes": filtered_episodes,
            "patients": list(final_patients),
            "n_patients": len(final_patients),
            "n_episodes": len(filtered_episodes),
            "criteria": {
                "min_episodes": min_episodes,
                "require_outcome": require_outcome,
                "outcome_column": outcome_column if require_outcome else None,
            },
            "attrition": attrition,
        }
