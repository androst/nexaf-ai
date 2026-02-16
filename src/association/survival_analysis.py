"""
Survival analysis for AF-related outcomes.

Provides Kaplan-Meier curves, log-rank tests, and Cox proportional hazards regression.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class AFSurvivalAnalysis:
    """Survival analysis for AF-related outcomes."""

    def __init__(
        self,
        event_column: str = "Post_AF_hosp",
        time_column: str = "follow_up_days",
        id_column: str = "ID",
    ):
        """
        Initialize survival analysis.

        Args:
            event_column: Column indicating event occurrence (1=event, 0=censored).
            time_column: Column with time to event or censoring.
            id_column: Patient ID column.
        """
        self.event_column = event_column
        self.time_column = time_column
        self.id_column = id_column

    def prepare_survival_data(
        self,
        df: pd.DataFrame,
        max_follow_up: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Prepare data for survival analysis.

        Args:
            df: DataFrame with event and time columns.
            max_follow_up: Optional maximum follow-up time for censoring.

        Returns:
            DataFrame ready for survival analysis.
        """
        df = df.copy()

        # Ensure event column is binary
        if self.event_column in df.columns:
            # Handle various encodings
            if df[self.event_column].dtype == 'object':
                df[self.event_column] = df[self.event_column].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

            # Convert NaN to 0 (no event = censored)
            df[self.event_column] = df[self.event_column].fillna(0).astype(int)

        # Apply max follow-up censoring
        if max_follow_up is not None and self.time_column in df.columns:
            exceeded = df[self.time_column] > max_follow_up
            df.loc[exceeded, self.time_column] = max_follow_up
            df.loc[exceeded, self.event_column] = 0  # Censor at max follow-up

        return df

    def kaplan_meier(
        self,
        df: pd.DataFrame,
        group_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Kaplan-Meier survival estimates.

        Args:
            df: DataFrame with survival data.
            group_column: Optional column for stratification.

        Returns:
            Dictionary with KM results per group.
        """
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            raise ImportError("lifelines required. Install with: pip install lifelines")

        results = {}

        if group_column is None:
            # Single KM curve
            kmf = KaplanMeierFitter()
            kmf.fit(
                df[self.time_column],
                event_observed=df[self.event_column],
            )
            results["overall"] = {
                "survival_function": kmf.survival_function_,
                "median_survival": kmf.median_survival_time_,
                "confidence_interval": kmf.confidence_interval_survival_function_,
                "fitter": kmf,
            }
        else:
            # Stratified KM curves
            for group in df[group_column].unique():
                group_df = df[df[group_column] == group]
                if len(group_df) < 2:
                    continue

                kmf = KaplanMeierFitter()
                kmf.fit(
                    group_df[self.time_column],
                    event_observed=group_df[self.event_column],
                    label=f"{group_column}={group}",
                )
                results[group] = {
                    "survival_function": kmf.survival_function_,
                    "median_survival": kmf.median_survival_time_,
                    "n_events": group_df[self.event_column].sum(),
                    "n_total": len(group_df),
                    "fitter": kmf,
                }

        return results

    def logrank_test(
        self,
        df: pd.DataFrame,
        group_column: str,
    ) -> Dict[str, float]:
        """
        Perform log-rank test comparing survival between groups.

        Args:
            df: DataFrame with survival data.
            group_column: Column defining groups to compare.

        Returns:
            Dictionary with test statistic and p-value.
        """
        try:
            from lifelines.statistics import multivariate_logrank_test
        except ImportError:
            raise ImportError("lifelines required. Install with: pip install lifelines")

        results = multivariate_logrank_test(
            df[self.time_column],
            df[group_column],
            df[self.event_column],
        )

        return {
            "test_statistic": results.test_statistic,
            "p_value": results.p_value,
            "n_groups": df[group_column].nunique(),
        }

    def cox_regression(
        self,
        df: pd.DataFrame,
        covariates: List[str],
        penalizer: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Fit Cox proportional hazards model.

        Args:
            df: DataFrame with survival data and covariates.
            covariates: List of covariate column names.
            penalizer: L2 penalization (regularization) strength.

        Returns:
            Dictionary with Cox model results.
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            raise ImportError("lifelines required. Install with: pip install lifelines")

        # Prepare data
        cox_df = df[[self.time_column, self.event_column] + covariates].dropna()

        # Fit model
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(
            cox_df,
            duration_col=self.time_column,
            event_col=self.event_column,
        )

        return {
            "summary": cph.summary,
            "hazard_ratios": np.exp(cph.params_),
            "confidence_intervals": np.exp(cph.confidence_intervals_),
            "p_values": cph.summary["p"],
            "concordance": cph.concordance_index_,
            "log_likelihood": cph.log_likelihood_,
            "aic": cph.AIC_,
            "fitter": cph,
        }

    def cox_with_phenotype(
        self,
        df: pd.DataFrame,
        phenotype_column: str = "phenotype",
        reference_phenotype: Optional[int] = None,
        covariates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fit Cox model with phenotype as primary predictor.

        Args:
            df: DataFrame with survival data.
            phenotype_column: Column with phenotype assignments.
            reference_phenotype: Reference phenotype for dummy coding.
            covariates: Additional covariates to adjust for.

        Returns:
            Dictionary with Cox model results.
        """
        df = df.copy()

        # Create dummy variables for phenotype
        if reference_phenotype is None:
            reference_phenotype = df[phenotype_column].min()

        phenotype_dummies = pd.get_dummies(
            df[phenotype_column],
            prefix="phenotype",
            drop_first=False,
        )

        # Drop reference category
        ref_col = f"phenotype_{reference_phenotype}"
        if ref_col in phenotype_dummies.columns:
            phenotype_dummies = phenotype_dummies.drop(columns=[ref_col])

        # Combine with original data
        df = pd.concat([df, phenotype_dummies], axis=1)

        # Define covariates
        phenotype_cols = phenotype_dummies.columns.tolist()
        all_covariates = phenotype_cols + (covariates or [])

        return self.cox_regression(df, all_covariates)

    def check_proportional_hazards(
        self,
        cph,
    ) -> pd.DataFrame:
        """
        Check proportional hazards assumption using Schoenfeld residuals.

        Args:
            cph: Fitted CoxPHFitter object.

        Returns:
            DataFrame with test results for each covariate.
        """
        try:
            return cph.check_assumptions(show_plots=False)
        except Exception as e:
            print(f"PH assumption check failed: {e}")
            return None

    def survival_summary_by_group(
        self,
        df: pd.DataFrame,
        group_column: str,
        timepoints: List[float] = [30, 90, 180, 365],
    ) -> pd.DataFrame:
        """
        Summarize survival at specific timepoints by group.

        Args:
            df: DataFrame with survival data.
            group_column: Column defining groups.
            timepoints: Times at which to estimate survival.

        Returns:
            DataFrame with survival estimates at each timepoint.
        """
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            raise ImportError("lifelines required")

        results = []

        for group in df[group_column].unique():
            group_df = df[df[group_column] == group]

            kmf = KaplanMeierFitter()
            kmf.fit(
                group_df[self.time_column],
                event_observed=group_df[self.event_column],
            )

            row = {
                "group": group,
                "n_total": len(group_df),
                "n_events": group_df[self.event_column].sum(),
                "event_rate": group_df[self.event_column].mean(),
            }

            # Survival at each timepoint
            for t in timepoints:
                try:
                    surv = kmf.predict(t)
                    row[f"survival_{t}d"] = surv
                except:
                    row[f"survival_{t}d"] = np.nan

            results.append(row)

        return pd.DataFrame(results)


def plot_kaplan_meier(
    km_results: Dict[str, Any],
    title: str = "Kaplan-Meier Survival Curves",
    xlabel: str = "Time (days)",
    ylabel: str = "Survival Probability",
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot Kaplan-Meier curves.

    Args:
        km_results: Results from kaplan_meier method.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.

    Returns:
        matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for group, data in km_results.items():
        if "fitter" in data:
            data["fitter"].plot_survival_function(ax=ax, ci_show=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Group")

    return fig, ax
