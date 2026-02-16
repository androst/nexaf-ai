"""
Clinical utility functions for phenotype analysis.

Contains general-purpose clinical calculations and formatting functions.
"""

import pandas as pd
import numpy as np


def calculate_bsa(height_cm: float, weight_kg: float) -> float:
    """Calculate Body Surface Area using the Dubois formula.

    BSA (m²) = 0.007184 × height^0.725 × weight^0.425

    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms

    Returns:
        Body Surface Area in square meters
    """
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)


def format_pvalue(p: float) -> str:
    """Format p-value for display in statistical tables.

    Args:
        p: P-value to format

    Returns:
        Formatted string: '<0.001' for p < 0.001, 'N/A' for missing,
        or value with 3 decimal places
    """
    if pd.isna(p):
        return 'N/A'
    elif p < 0.001:
        return '<0.001'
    else:
        return f'{p:.3f}'
