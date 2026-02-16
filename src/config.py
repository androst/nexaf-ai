"""
Configuration module for AF Trajectory Analysis.

Contains paths, constants, and settings used across the project.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATABASE_DIR = PROJECT_ROOT / "database"
EPISODE_FILE = DATABASE_DIR / "df_ep5_fil_sep25_EFL.sav"
BURDEN_FILE = DATABASE_DIR / "Hovedfil_analyser_hovedartikkel021225_oppdatertAFburden.sav"
BASELINE_FILE = DATABASE_DIR / "NEXAF_baselinefil_171125_EFL.sav"
VALIDATION_FILE = DATABASE_DIR / "valideringsdatasett300126.sav"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""

    # Time alignment
    max_follow_up_days: int = 365
    exclude_pre_implant: bool = True

    # Time windows for analysis (days)
    time_windows: List[Tuple[int, int]] = None

    # Episode cleaning thresholds
    min_episode_duration_minutes: float = 0.0
    max_episode_duration_minutes: float = 10080  # 7 days
    min_rr_interval_msec: float = 200
    max_rr_interval_msec: float = 2000

    # Feature extraction
    trajectory_n_timepoints: int = 100
    trajectory_smooth_window: int = 7

    # Clustering
    random_state: int = 42
    n_clusters_range: Tuple[int, int] = (2, 8)

    # Outcome columns
    hospitalization_col: str = "Post_AF_hosp"
    qol_col: str = "Post_Afeqt_symptoms_score"
    af_type_col: str = "BL_AF_type"

    def __post_init__(self):
        if self.time_windows is None:
            self.time_windows = [
                (0, 30),    # First month
                (30, 90),   # Months 2-3
                (90, 180),  # Months 4-6
                (180, 365), # Months 7-12
            ]


# Default configuration instance
DEFAULT_CONFIG = AnalysisConfig()


# Column name mappings (Norwegian to English)
COLUMN_TRANSLATIONS = {
    "Kjønn": "gender",
    "Alder": "age",
    "F_år": "birth_year",
    "År_AF_diagnose": "af_diagnosis_year",
}

# Key columns in episode data
EPISODE_COLUMNS = {
    "id": "ID",
    "implant_date": "date_ilr_implant",
    "episode_start": "time_start_ep",
    "episode_stop": "time_stop_ep",
    "duration_minutes": "af_episode_minutes",
    "episode_number": "ep_number",
    "rr_interval": "AF_MEAN_RR_INTERVAL_msec",
    "rr_rate": "AF_MEAN_RR_RATE_bpm",
    "daytime": "episode_start_during_day",
}

# Key columns in burden/outcome data
OUTCOME_COLUMNS = {
    "id": "ID",
    "af_type_baseline": "BL_AF_type",
    "afeqt_baseline": "BL_Afeqt_symptoms_score",
    "afeqt_6month": "Six_months_Afeqt_symptoms_score",
    "afeqt_post": "Post_Afeqt_symptoms_score",
    "hosp_cvd": "Post_CVD_hospi",
    "hosp_general": "Post_hospi",
    "hosp_af": "Post_AF_hosp",
    "hosp_non_af": "Post_non_AF_hospi",
}
