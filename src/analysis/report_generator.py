"""
HTML report generator for phenotype analysis results.

Generates styled HTML reports with color-coded phenotype columns
and comprehensive analysis tables.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# Default phenotype colors (red, blue, green, purple, orange, brown)
DEFAULT_PHENOTYPE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628'
]


@dataclass
class ReportData:
    """Container for all data needed to generate a phenotype report."""

    # Required data
    summary_df: pd.DataFrame
    n_phenotypes: int
    n_patients: int

    # Optional analysis tables
    ilr_features_df: Optional[pd.DataFrame] = None
    baseline_df: Optional[pd.DataFrame] = None
    echo_df: Optional[pd.DataFrame] = None
    cpet_df: Optional[pd.DataFrame] = None
    qol_df: Optional[pd.DataFrame] = None
    exercise_df: Optional[pd.DataFrame] = None
    treatment_df: Optional[pd.DataFrame] = None
    hosp_df: Optional[pd.DataFrame] = None
    hosp_p_val: Optional[float] = None

    # Styling
    phenotype_colors: List[str] = field(default_factory=lambda: DEFAULT_PHENOTYPE_COLORS.copy())


class PhenotypeReportGenerator:
    """Generates styled HTML reports for phenotype analysis results."""

    CSS_TEMPLATE = """
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 40px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 12px; }
        th { background-color: #3498db; color: white; padding: 10px; text-align: left; }
        td { padding: 8px; border-bottom: 1px solid #ddd; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f5f5f5; }
        .significant { background-color: #d5f4e6; }
        .header-info { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .note { color: #7f8c8d; font-style: italic; font-size: 11px; }
    </style>
    """

    def __init__(self, data: ReportData):
        """Initialize the report generator.

        Args:
            data: ReportData container with all analysis results
        """
        self.data = data

    def colorize_phenotype_headers(self, html_str: str) -> str:
        """Add phenotype colors to table headers P0, P1, P2, etc.

        Args:
            html_str: HTML string to process

        Returns:
            HTML string with colored phenotype headers
        """
        for p in range(self.data.n_phenotypes):
            color = self.data.phenotype_colors[p]
            html_str = html_str.replace(
                f'<th>P{p}</th>',
                f'<th style="background-color: {color};">P{p}</th>'
            )
        return html_str

    def generate(self) -> str:
        """Generate the complete HTML report.

        Returns:
            Complete HTML string for the report
        """
        sections = []

        # Header
        sections.append(self._generate_header())

        # Section 1: Phenotype Summary
        sections.append(self._generate_section(
            "1. Phenotype Summary",
            f"Key characteristics of each phenotype (P0-P{self.data.n_phenotypes - 1})",
            self.data.summary_df.to_html(index=False, classes='summary-table')
        ))

        # Section 2: ILR Features
        if self.data.ilr_features_df is not None:
            ilr_html = self._prepare_ilr_table()
            sections.append(self._generate_section(
                "2. ILR Monitoring Features",
                "14 curated features from implantable loop recorder data used for phenotype clustering. "
                "Mean values per phenotype with Kruskal-Wallis p-values and normalized feature importance (ANOVA F-score).",
                ilr_html
            ))

        # Section 3: Baseline Characteristics
        if self.data.baseline_df is not None:
            sections.append(self._generate_section(
                "3. Baseline Characteristics",
                "Demographics, comorbidities, medications, laboratory values by phenotype. "
                "P-values from Kruskal-Wallis (continuous) or Chi-square (categorical) tests.",
                self.data.baseline_df.sort_values('p-value').to_html(index=False, classes='baseline-table', na_rep='N/A')
            ))

        # Section 4: Echocardiography
        if self.data.echo_df is not None:
            sections.append(self._generate_section(
                "4. Echocardiography",
                "Raw values, BSA-indexed, and VO₂peak-indexed parameters (per Letnes et al. 2023)",
                self.data.echo_df.sort_values('p-value').to_html(index=False, classes='echo-table', na_rep='N/A')
            ))

        # Section 5: CPET
        if self.data.cpet_df is not None:
            sections.append(self._generate_section(
                "5. Cardiopulmonary Exercise Test (CPET)",
                "Peak exercise capacity and HR recovery parameters",
                self.data.cpet_df.sort_values('p-value').to_html(index=False, classes='cpet-table', na_rep='N/A')
            ))

        # Section 6: Quality of Life
        if self.data.qol_df is not None:
            sections.append(self._generate_section(
                "6. Quality of Life",
                "AFEQT scores at baseline, 6 months, and post-intervention",
                self.data.qol_df.sort_values('p-value').to_html(index=False, classes='qol-table', na_rep='N/A')
            ))

        # Section 7: Exercise Time
        if self.data.exercise_df is not None:
            sections.append(self._generate_section(
                "7. Exercise Time",
                "Baseline and post-intervention exercise minutes per week",
                self.data.exercise_df.to_html(index=False, classes='exercise-table', na_rep='N/A')
            ))

        # Section 8: Treatments
        if self.data.treatment_df is not None:
            sections.append(self._generate_section(
                "8. Treatments & Procedures",
                "Post-intervention ablation, cardioversion, and medication use",
                self.data.treatment_df.sort_values('p-value').to_html(index=False, classes='treatment-table', na_rep='N/A')
            ))

        # Section 9: Hospitalization
        if self.data.hosp_df is not None:
            hosp_html = self.data.hosp_df.to_html(index=False, classes='hosp-table')
            if self.data.hosp_p_val is not None:
                hosp_html += f'\n    <p><strong>Chi-square test:</strong> p = {self.data.hosp_p_val:.4f}</p>'
            sections.append(self._generate_section(
                "9. Hospitalization",
                "AF-related hospitalization rates by phenotype",
                hosp_html
            ))

        # Footer
        sections.append(self._generate_footer())

        # Combine and colorize
        html = self._wrap_html('\n'.join(sections))
        html = self.colorize_phenotype_headers(html)

        return html

    def save(self, output_path: str) -> None:
        """Generate and save the report to a file.

        Args:
            output_path: Path to save the HTML file
        """
        html = self.generate()
        Path(output_path).write_text(html, encoding='utf-8')

    def _generate_header(self) -> str:
        """Generate the report header section."""
        return f"""
    <h1>NEXAF AF Phenotype Analysis Results</h1>

    <div class="header-info">
        <strong>Study:</strong> NEXAF - Exercise Training in Atrial Fibrillation<br>
        <strong>Patients:</strong> {self.data.n_patients} with phenotype assignments<br>
        <strong>Phenotypes:</strong> {self.data.n_phenotypes} clusters identified<br>
        <strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d')}
    </div>
"""

    def _generate_section(self, title: str, note: str, content: str) -> str:
        """Generate a report section."""
        return f"""
    <h2>{title}</h2>
    <p class="note">{note}</p>
    {content}
"""

    def _generate_footer(self) -> str:
        """Generate the report footer."""
        return """
    <hr>
    <p class="note">Generated by NEXAF-AI Analysis Pipeline</p>
"""

    def _wrap_html(self, body: str) -> str:
        """Wrap content in HTML document structure."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NEXAF Phenotype Analysis Results</title>
    {self.CSS_TEMPLATE}
</head>
<body>
{body}
</body>
</html>
"""

    def _prepare_ilr_table(self) -> str:
        """Prepare ILR features table with proper formatting."""
        ilr_html_df = self.data.ilr_features_df.copy()
        ilr_html_df['p-value'] = ilr_html_df['p-value'].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
        ilr_html_df['Importance'] = ilr_html_df['Importance'].apply(
            lambda x: f"{x:.2f}"
        )
        display_cols = ['Feature', 'Description', 'Category'] + \
                       [f'P{p}' for p in range(self.data.n_phenotypes)] + \
                       ['p-value', 'Importance']
        ilr_html_df = ilr_html_df[display_cols]
        return ilr_html_df.to_html(index=False, classes='ilr-table', na_rep='N/A')
