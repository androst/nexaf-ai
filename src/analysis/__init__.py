"""
Analysis module for phenotype comparison and reporting.

This module provides functions for:
- Creating statistical comparison tables across phenotypes
- Generating descriptive phenotype labels
- Clinical utility calculations (BSA, p-value formatting)
- HTML report generation

Example usage:
    from src.analysis import (
        create_analysis_table,
        generate_phenotype_names,
        calculate_bsa,
        format_pvalue,
    )
    from src.analysis.report_generator import PhenotypeReportGenerator, ReportData
"""

from .phenotype_analysis import (
    generate_phenotype_names,
    get_phenotype_display_cols,
    create_analysis_table,
)
from .clinical_utils import calculate_bsa, format_pvalue
from .report_generator import PhenotypeReportGenerator, ReportData, DEFAULT_PHENOTYPE_COLORS

__all__ = [
    # Phenotype analysis
    'generate_phenotype_names',
    'get_phenotype_display_cols',
    'create_analysis_table',
    # Clinical utilities
    'calculate_bsa',
    'format_pvalue',
    # Report generation
    'PhenotypeReportGenerator',
    'ReportData',
    'DEFAULT_PHENOTYPE_COLORS',
]
