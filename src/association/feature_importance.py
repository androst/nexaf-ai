"""
Feature importance analysis for interpretability.

Provides SHAP values, permutation importance, and feature ranking.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureImportanceAnalyzer:
    """Analyze feature importance for model interpretability."""

    def __init__(self, random_state: int = 42):
        """
        Initialize feature importance analyzer.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state

    def shap_analysis(
        self,
        model,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for feature importance.

        Args:
            model: Trained model (sklearn compatible).
            X: Feature DataFrame or array.
            feature_names: Optional list of feature names.
            max_samples: Maximum samples for SHAP (for speed).

        Returns:
            Dictionary with SHAP results.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap required. Install with: pip install shap")

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

        # Sample if needed
        if len(X_array) > max_samples:
            idx = np.random.choice(len(X_array), max_samples, replace=False)
            X_sample = X_array[idx]
        else:
            X_sample = X_array

        # Create explainer
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

            # Mean absolute SHAP value per feature
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            if len(mean_shap.shape) > 1:
                mean_shap = mean_shap.mean(axis=1)  # Average over classes if multi-class

            importance = dict(zip(feature_names, mean_shap))
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            return {
                "shap_values": shap_values,
                "mean_abs_shap": sorted_importance,
                "explainer": explainer,
                "feature_names": feature_names,
            }

        except Exception as e:
            # Fallback to TreeExplainer for tree-based models
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class for binary

                mean_shap = np.abs(shap_values).mean(axis=0)
                importance = dict(zip(feature_names, mean_shap))
                sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

                return {
                    "shap_values": shap_values,
                    "mean_abs_shap": sorted_importance,
                    "explainer": explainer,
                    "feature_names": feature_names,
                }
            except Exception as e2:
                return {"error": str(e2)}

    def permutation_importance_analysis(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        scoring: str = "roc_auc",
    ) -> Dict[str, Any]:
        """
        Compute permutation-based feature importance.

        Args:
            model: Trained model.
            X: Feature matrix.
            y: Target variable.
            feature_names: Optional feature names.
            n_repeats: Number of permutation repeats.
            scoring: Scoring metric.

        Returns:
            Dictionary with permutation importance results.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring,
            n_jobs=-1,
        )

        importance = dict(zip(feature_names, result.importances_mean))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        importance_std = dict(zip(feature_names, result.importances_std))

        return {
            "importance_mean": sorted_importance,
            "importance_std": importance_std,
            "raw_importances": result.importances,
            "feature_names": feature_names,
        }

    def random_forest_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        task: str = "classification",
        n_estimators: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute feature importance using Random Forest.

        Args:
            X: Feature matrix.
            y: Target variable.
            feature_names: Optional feature names.
            task: 'classification' or 'regression'.
            n_estimators: Number of trees.

        Returns:
            Dictionary with RF importance results.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )

        model.fit(X, y)

        importance = dict(zip(feature_names, model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "importance": sorted_importance,
            "model": model,
            "feature_names": feature_names,
        }

    def phenotype_discriminating_features(
        self,
        X: pd.DataFrame,
        phenotype_labels: np.ndarray,
        n_top: int = 20,
    ) -> pd.DataFrame:
        """
        Find features that best discriminate between phenotypes.

        Uses ANOVA F-test and RF importance.

        Args:
            X: Feature DataFrame.
            phenotype_labels: Phenotype cluster labels.
            n_top: Number of top features to return.

        Returns:
            DataFrame with ranked discriminating features.
        """
        from scipy import stats

        feature_names = X.columns.tolist()
        results = []

        for col in feature_names:
            values = X[col].values

            # Skip if too many missing
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() < len(values) * 0.5:
                continue

            # ANOVA F-test
            groups = [values[valid_mask & (phenotype_labels == p)]
                     for p in np.unique(phenotype_labels)]
            groups = [g for g in groups if len(g) > 1]

            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    results.append({
                        "feature": col,
                        "f_statistic": f_stat,
                        "p_value": p_value,
                    })
                except:
                    pass

        df_results = pd.DataFrame(results)
        if len(df_results) == 0:
            return df_results

        # Add RF importance
        X_clean = X.fillna(X.median())
        rf_importance = self.random_forest_importance(
            X_clean.values, phenotype_labels,
            feature_names=feature_names,
            task="classification",
        )

        df_results["rf_importance"] = df_results["feature"].map(rf_importance["importance"])

        # Rank features
        df_results = df_results.sort_values("f_statistic", ascending=False)
        df_results["rank_anova"] = range(1, len(df_results) + 1)

        df_results = df_results.sort_values("rf_importance", ascending=False)
        df_results["rank_rf"] = range(1, len(df_results) + 1)

        df_results["mean_rank"] = (df_results["rank_anova"] + df_results["rank_rf"]) / 2
        df_results = df_results.sort_values("mean_rank")

        return df_results.head(n_top)

    def feature_outcome_correlations(
        self,
        features: pd.DataFrame,
        outcomes: pd.DataFrame,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute correlations between features and outcomes.

        Args:
            features: Feature DataFrame.
            outcomes: Outcome DataFrame.
            method: Correlation method ('pearson', 'spearman').

        Returns:
            DataFrame with correlation matrix.
        """
        from scipy import stats

        feature_cols = features.columns.tolist()
        outcome_cols = outcomes.columns.tolist()

        results = []

        for feat in feature_cols:
            for outcome in outcome_cols:
                valid = features[feat].notna() & outcomes[outcome].notna()

                if valid.sum() < 10:
                    continue

                x = features.loc[valid, feat].values
                y = outcomes.loc[valid, outcome].values

                if method == "spearman":
                    corr, p_value = stats.spearmanr(x, y)
                else:
                    corr, p_value = stats.pearsonr(x, y)

                results.append({
                    "feature": feat,
                    "outcome": outcome,
                    "correlation": corr,
                    "p_value": p_value,
                    "n": valid.sum(),
                })

        df = pd.DataFrame(results)

        # Add significance flag
        if len(df) > 0:
            df["significant"] = df["p_value"] < 0.05

        return df.sort_values("correlation", key=abs, ascending=False)

    def top_features_summary(
        self,
        importance_results: Dict[str, Any],
        n_top: int = 15,
    ) -> pd.DataFrame:
        """
        Create summary table of top features.

        Args:
            importance_results: Results from any importance method.
            n_top: Number of top features.

        Returns:
            DataFrame with top features.
        """
        if "mean_abs_shap" in importance_results:
            importance = importance_results["mean_abs_shap"]
            method = "SHAP"
        elif "importance_mean" in importance_results:
            importance = importance_results["importance_mean"]
            method = "Permutation"
        elif "importance" in importance_results:
            importance = importance_results["importance"]
            method = "RF"
        else:
            raise ValueError("Unknown importance format")

        top_features = list(importance.items())[:n_top]

        return pd.DataFrame(top_features, columns=["feature", f"{method.lower()}_importance"])


def create_importance_report(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_top: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive feature importance report.

    Args:
        model: Trained model.
        X: Feature matrix.
        y: Target variable.
        feature_names: Feature names.
        n_top: Number of top features to include.

    Returns:
        Dictionary with multiple importance rankings.
    """
    analyzer = FeatureImportanceAnalyzer()

    reports = {}

    # Permutation importance
    try:
        perm_results = analyzer.permutation_importance_analysis(
            model, X, y, feature_names, scoring="roc_auc"
        )
        reports["permutation"] = analyzer.top_features_summary(perm_results, n_top)
    except Exception as e:
        print(f"Permutation importance failed: {e}")

    # SHAP (if available)
    try:
        shap_results = analyzer.shap_analysis(model, X, feature_names)
        if "error" not in shap_results:
            reports["shap"] = analyzer.top_features_summary(shap_results, n_top)
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    # RF importance
    try:
        rf_results = analyzer.random_forest_importance(X, y, feature_names)
        reports["random_forest"] = analyzer.top_features_summary(rf_results, n_top)
    except Exception as e:
        print(f"RF importance failed: {e}")

    return reports
