"""
Outcome prediction models for AF phenotypes.

Provides logistic and linear regression for binary and continuous outcomes.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


class OutcomePrediction:
    """Predict clinical outcomes from features and phenotypes."""

    def __init__(
        self,
        id_column: str = "ID",
        random_state: int = 42,
    ):
        """
        Initialize outcome prediction.

        Args:
            id_column: Patient ID column name.
            random_state: Random seed for reproducibility.
        """
        self.id_column = id_column
        self.random_state = random_state
        self.scaler = StandardScaler()

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        outcome_column: str,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for modeling.

        Args:
            df: DataFrame with features and outcome.
            feature_columns: List of feature column names.
            outcome_column: Outcome column name.
            scale: Whether to standardize features.

        Returns:
            Tuple of (X, y, valid_feature_names).
        """
        # Get complete cases
        valid_cols = [c for c in feature_columns if c in df.columns]
        df_valid = df[valid_cols + [outcome_column]].dropna()

        X = df_valid[valid_cols].values
        y = df_valid[outcome_column].values

        if scale:
            X = self.scaler.fit_transform(X)

        return X, y, valid_cols

    def logistic_regression(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        outcome_column: str,
        cv_folds: int = 5,
        return_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Logistic regression for binary outcomes.

        Args:
            df: DataFrame with features and outcome.
            feature_columns: Feature columns to use.
            outcome_column: Binary outcome column.
            cv_folds: Number of cross-validation folds.
            return_model: Whether to return fitted model.

        Returns:
            Dictionary with model results.
        """
        X, y, valid_cols = self.prepare_data(df, feature_columns, outcome_column)

        # Check if binary
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError(f"Outcome must be binary, found {len(unique_y)} unique values")

        # Fit model
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced',
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        # Fit on full data for coefficients
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        results = {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "cv_auc_scores": cv_scores,
            "train_auc": roc_auc_score(y, y_proba),
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "coefficients": dict(zip(valid_cols, model.coef_[0])),
            "intercept": model.intercept_[0],
            "n_samples": len(y),
            "n_events": int(y.sum()),
            "feature_names": valid_cols,
        }

        if return_model:
            results["model"] = model
            results["scaler"] = self.scaler

        return results

    def linear_regression(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        outcome_column: str,
        cv_folds: int = 5,
        return_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Linear regression for continuous outcomes.

        Args:
            df: DataFrame with features and outcome.
            feature_columns: Feature columns to use.
            outcome_column: Continuous outcome column.
            cv_folds: Number of cross-validation folds.
            return_model: Whether to return fitted model.

        Returns:
            Dictionary with model results.
        """
        X, y, valid_cols = self.prepare_data(df, feature_columns, outcome_column)

        # Fit model
        model = LinearRegression()

        # Cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_mse = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')

        # Fit on full data
        model.fit(X, y)
        y_pred = model.predict(X)

        results = {
            "cv_r2_mean": cv_r2.mean(),
            "cv_r2_std": cv_r2.std(),
            "cv_r2_scores": cv_r2,
            "cv_rmse_mean": np.sqrt(cv_mse.mean()),
            "train_r2": r2_score(y, y_pred),
            "train_mse": mean_squared_error(y, y_pred),
            "train_rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "train_mae": mean_absolute_error(y, y_pred),
            "coefficients": dict(zip(valid_cols, model.coef_)),
            "intercept": model.intercept_,
            "n_samples": len(y),
            "feature_names": valid_cols,
        }

        if return_model:
            results["model"] = model
            results["scaler"] = self.scaler

        return results

    def random_forest_classifier(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        outcome_column: str,
        cv_folds: int = 5,
        n_estimators: int = 100,
    ) -> Dict[str, Any]:
        """
        Random forest for binary classification.

        Args:
            df: DataFrame with features and outcome.
            feature_columns: Feature columns to use.
            outcome_column: Binary outcome column.
            cv_folds: Number of cross-validation folds.
            n_estimators: Number of trees.

        Returns:
            Dictionary with model results.
        """
        X, y, valid_cols = self.prepare_data(df, feature_columns, outcome_column, scale=False)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1,
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        # Fit on full data
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]

        # Feature importance
        feature_importance = dict(zip(valid_cols, model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "train_auc": roc_auc_score(y, y_proba),
            "feature_importance": sorted_importance,
            "model": model,
            "n_samples": len(y),
            "feature_names": valid_cols,
        }

    def phenotype_outcome_table(
        self,
        df: pd.DataFrame,
        phenotype_column: str,
        outcome_columns: List[str],
    ) -> pd.DataFrame:
        """
        Create summary table of outcomes by phenotype.

        Args:
            df: DataFrame with phenotypes and outcomes.
            phenotype_column: Phenotype column name.
            outcome_columns: List of outcome columns.

        Returns:
            DataFrame with outcome summaries per phenotype.
        """
        results = []

        for phenotype in sorted(df[phenotype_column].unique()):
            pheno_df = df[df[phenotype_column] == phenotype]
            row = {
                "phenotype": phenotype,
                "n_patients": len(pheno_df),
            }

            for outcome in outcome_columns:
                if outcome not in df.columns:
                    continue

                values = pheno_df[outcome].dropna()

                if len(values) == 0:
                    continue

                # Check if binary or continuous
                unique_vals = values.unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                    # Binary outcome
                    row[f"{outcome}_n"] = len(values)
                    row[f"{outcome}_events"] = int(values.sum())
                    row[f"{outcome}_rate"] = values.mean()
                else:
                    # Continuous outcome
                    row[f"{outcome}_n"] = len(values)
                    row[f"{outcome}_mean"] = values.mean()
                    row[f"{outcome}_std"] = values.std()
                    row[f"{outcome}_median"] = values.median()

            results.append(row)

        return pd.DataFrame(results)

    def compare_phenotypes(
        self,
        df: pd.DataFrame,
        phenotype_column: str,
        outcome_column: str,
        test_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Statistical comparison of outcome between phenotypes.

        Args:
            df: DataFrame with phenotypes and outcome.
            phenotype_column: Phenotype column name.
            outcome_column: Outcome column name.
            test_type: Type of test ('auto', 'chi2', 'anova', 'kruskal').

        Returns:
            Dictionary with test results.
        """
        from scipy import stats

        df_valid = df[[phenotype_column, outcome_column]].dropna()
        groups = [group[outcome_column].values for name, group in df_valid.groupby(phenotype_column)]

        # Determine test type
        outcome_values = df_valid[outcome_column]
        is_binary = len(outcome_values.unique()) <= 2

        if test_type == "auto":
            test_type = "chi2" if is_binary else "kruskal"

        if test_type == "chi2":
            # Chi-square test for binary outcomes
            contingency = pd.crosstab(df_valid[phenotype_column], df_valid[outcome_column])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            return {
                "test": "chi2",
                "statistic": chi2,
                "p_value": p_value,
                "dof": dof,
                "contingency_table": contingency,
            }

        elif test_type == "anova":
            # One-way ANOVA for continuous outcomes
            f_stat, p_value = stats.f_oneway(*groups)
            return {
                "test": "anova",
                "statistic": f_stat,
                "p_value": p_value,
            }

        elif test_type == "kruskal":
            # Kruskal-Wallis for non-parametric comparison
            h_stat, p_value = stats.kruskal(*groups)
            return {
                "test": "kruskal",
                "statistic": h_stat,
                "p_value": p_value,
            }

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def odds_ratios_by_phenotype(
        self,
        df: pd.DataFrame,
        phenotype_column: str,
        outcome_column: str,
        reference_phenotype: Optional[int] = None,
        covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate odds ratios for each phenotype vs reference.

        Args:
            df: DataFrame with phenotypes and binary outcome.
            phenotype_column: Phenotype column name.
            outcome_column: Binary outcome column name.
            reference_phenotype: Reference phenotype for comparison.
            covariates: Optional covariates for adjusted OR.

        Returns:
            DataFrame with odds ratios and confidence intervals.
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")

        df = df.copy()

        if reference_phenotype is None:
            reference_phenotype = df[phenotype_column].min()

        # Create dummy variables
        phenotype_dummies = pd.get_dummies(df[phenotype_column], prefix="pheno", drop_first=False)
        ref_col = f"pheno_{reference_phenotype}"

        if ref_col in phenotype_dummies.columns:
            phenotype_dummies = phenotype_dummies.drop(columns=[ref_col])

        # Prepare model data
        X_cols = phenotype_dummies.columns.tolist()
        if covariates:
            X_cols += [c for c in covariates if c in df.columns]

        df_model = pd.concat([df[[outcome_column]], phenotype_dummies], axis=1)
        if covariates:
            for c in covariates:
                if c in df.columns:
                    df_model[c] = df[c]

        df_model = df_model.dropna()

        X = df_model[X_cols]
        X = sm.add_constant(X)
        y = df_model[outcome_column]

        # Fit logistic regression
        model = sm.Logit(y, X).fit(disp=0)

        # Extract results
        results = []
        for col in phenotype_dummies.columns:
            if col in model.params.index:
                or_val = np.exp(model.params[col])
                ci_low = np.exp(model.conf_int().loc[col, 0])
                ci_high = np.exp(model.conf_int().loc[col, 1])
                p_val = model.pvalues[col]

                phenotype_num = col.replace("pheno_", "")
                results.append({
                    "phenotype": phenotype_num,
                    "odds_ratio": or_val,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "p_value": p_val,
                })

        # Add reference
        results.append({
            "phenotype": str(reference_phenotype),
            "odds_ratio": 1.0,
            "ci_lower": 1.0,
            "ci_upper": 1.0,
            "p_value": np.nan,
        })

        return pd.DataFrame(results).sort_values("phenotype")
