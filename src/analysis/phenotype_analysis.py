"""
Phenotype analysis functions for comparing characteristics across AF phenotypes.

Contains functions for creating statistical comparison tables and generating
descriptive phenotype labels.
"""

from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import stats


def generate_phenotype_names(features_df: pd.DataFrame, n_phenotypes: int) -> Dict[int, str]:
    """Generate descriptive phenotype names based on clustering characteristics.

    Creates human-readable labels for each phenotype based on AF burden,
    episode count, and mean episode duration.

    Args:
        features_df: DataFrame with phenotype assignments and feature columns
        n_phenotypes: Number of phenotypes

    Returns:
        Dictionary mapping phenotype index to descriptive name string
    """
    names = {}
    for p in range(n_phenotypes):
        subset = features_df[features_df['phenotype'] == p]
        burden = subset['af_burden_percent'].mean()
        n_eps = subset['n_episodes'].mean()
        duration = subset['mean_episode_duration'].mean()
        names[p] = f'P{p}: {burden:.1f}% burden - {n_eps:.0f} episodes - {duration:.0f} min'
    return names


def get_phenotype_display_cols(n_phenotypes: int) -> List[str]:
    """Generate display column list based on number of phenotypes.

    Args:
        n_phenotypes: Number of phenotypes

    Returns:
        List of column names for display tables
    """
    return ['Variable', 'Overall'] + [f'P{p}' for p in range(n_phenotypes)] + ['p-value']


def create_analysis_table(
    df: pd.DataFrame,
    variables_dict: Dict[str, str],
    phenotype_col: str = 'phenotype',
    n_phenotypes: int = 4,
    force_continuous: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create a standardized analysis table with statistics and p-values.

    Automatically determines if variables are continuous or categorical and
    applies appropriate statistical tests:
    - Continuous variables: Kruskal-Wallis test with mean (SD)
    - Binary variables: Chi-square test with percentages
    - Multi-category variables: Chi-square test with mode

    Args:
        df: DataFrame with data
        variables_dict: Dict mapping column names to display labels
        phenotype_col: Name of phenotype column
        n_phenotypes: Number of phenotypes
        force_continuous: List of column names to always treat as continuous
            (e.g., count variables that have few unique values)

    Returns:
        DataFrame with formatted results including Variable, N, Overall,
        P0-Pn columns, and p-value
    """
    if force_continuous is None:
        force_continuous = []

    table_data = []

    for col, label in variables_dict.items():
        if col not in df.columns:
            continue

        # Determine if continuous or categorical based on dtype and unique values
        dtype = df[col].dtype
        n_unique = df[col].nunique()

        # Force continuous if specified, otherwise use auto-detection
        if col in force_continuous:
            is_categorical = False
        else:
            # String/object columns are always categorical
            # Numeric columns with <= 5 unique values are treated as categorical
            is_categorical = (
                dtype == 'object' or
                dtype.name == 'category' or
                (n_unique <= 5 and np.issubdtype(dtype, np.number))
            )

        row = {'Variable': label}
        vals = df[col].dropna()
        row['N'] = len(vals)

        if not is_categorical:
            # Continuous variable
            try:
                row['Overall'] = f"{vals.mean():.1f} ({vals.std():.1f})" if len(vals) > 0 else 'N/A'

                groups = []
                for p in range(n_phenotypes):
                    pvals = df[df[phenotype_col] == p][col].dropna()
                    row[f'P{p}'] = f"{pvals.mean():.1f} ({pvals.std():.1f})" if len(pvals) > 0 else 'N/A'
                    groups.append(pvals)

                groups = [g for g in groups if len(g) >= 3]
                if len(groups) > 1:
                    h_stat, p_val = stats.kruskal(*groups)
                    row['p-value'] = p_val
                else:
                    row['p-value'] = np.nan
            except (TypeError, ValueError):
                # If mean/std fails, treat as categorical
                is_categorical = True

        if is_categorical:
            # Categorical variable - check if binary (2 unique values) or multi-category
            unique_vals = sorted(vals.unique())
            is_binary = len(unique_vals) == 2

            if is_binary:
                # Binary variable - show % of the higher value
                # Works for 0/1, 1/2, or any two-value coding
                positive_val = unique_vals[1]  # The higher value is "positive"
                pct_pos = 100 * (vals == positive_val).sum() / len(vals) if len(vals) > 0 else 0
                row['Overall'] = f"{pct_pos:.1f}%"

                for p in range(n_phenotypes):
                    pvals = df[df[phenotype_col] == p][col].dropna()
                    if len(pvals) > 0:
                        pct = 100 * (pvals == positive_val).sum() / len(pvals)
                        row[f'P{p}'] = f"{pct:.1f}%"
                    else:
                        row[f'P{p}'] = 'N/A'
            else:
                # Multi-category - show most common value
                most_common = vals.mode().iloc[0] if len(vals) > 0 else 'N/A'
                row['Overall'] = f"{most_common}"

                for p in range(n_phenotypes):
                    pvals = df[df[phenotype_col] == p][col].dropna()
                    if len(pvals) > 0:
                        mode_val = pvals.mode().iloc[0] if len(pvals.mode()) > 0 else 'N/A'
                        row[f'P{p}'] = f"{mode_val}"
                    else:
                        row[f'P{p}'] = 'N/A'

            # Chi-square test for categorical
            try:
                contingency = pd.crosstab(df[phenotype_col], df[col])
                if contingency.shape[1] > 1:
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                    row['p-value'] = p_val
                else:
                    row['p-value'] = np.nan
            except Exception:
                row['p-value'] = np.nan

        table_data.append(row)

    return pd.DataFrame(table_data)
