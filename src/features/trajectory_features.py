"""
Trajectory shape feature extraction.

Extracts features describing the shape of cumulative AF burden trajectories.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

from ..config import DEFAULT_CONFIG


class TrajectoryShapeExtractor:
    """Extract shape features from cumulative burden trajectories per patient."""

    def __init__(
        self,
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        time_column: str = "days_since_implant",
        n_timepoints: int = DEFAULT_CONFIG.trajectory_n_timepoints,
        smooth_window: int = DEFAULT_CONFIG.trajectory_smooth_window,
        max_follow_up_days: int = DEFAULT_CONFIG.max_follow_up_days,
    ):
        """
        Initialize the trajectory shape feature extractor.

        Args:
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            time_column: Time since implant column name.
            n_timepoints: Number of points for trajectory normalization.
            smooth_window: Days for rolling average smoothing.
            max_follow_up_days: Maximum follow-up period.
        """
        self.id_column = id_column
        self.duration_column = duration_column
        self.time_column = time_column
        self.n_timepoints = n_timepoints
        self.smooth_window = smooth_window
        self.max_follow_up_days = max_follow_up_days

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all trajectory shape features for each patient.

        Args:
            df: Episode DataFrame with time alignment.

        Returns:
            Patient-level DataFrame with trajectory features.
        """
        features = []

        for pid, group in df.groupby(self.id_column):
            patient_features = self._extract_patient_features(pid, group)
            features.append(patient_features)

        return pd.DataFrame(features)

    def _extract_patient_features(self, patient_id: int, df: pd.DataFrame) -> dict:
        """Extract features for a single patient."""
        features = {self.id_column: patient_id}

        # Sort by time
        df = df.sort_values(self.time_column)
        times = df[self.time_column].values
        durations = df[self.duration_column].values
        n_episodes = len(times)

        if n_episodes == 0:
            return self._empty_features(patient_id)

        # Calculate cumulative burden
        cumsum = np.cumsum(durations)
        total_burden = cumsum[-1]
        follow_up_days = min(times[-1], self.max_follow_up_days)

        # Overall slope (linear regression on cumulative burden)
        if n_episodes >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(times, cumsum)
            features["trajectory_slope"] = slope
            features["trajectory_intercept"] = intercept
            features["trajectory_r_squared"] = r_value ** 2
            features["trajectory_slope_pvalue"] = p_value
            features["trajectory_slope_stderr"] = std_err
        else:
            features["trajectory_slope"] = total_burden / follow_up_days if follow_up_days > 0 else 0
            features["trajectory_intercept"] = 0
            features["trajectory_r_squared"] = 1.0
            features["trajectory_slope_pvalue"] = np.nan
            features["trajectory_slope_stderr"] = np.nan

        # Curvature analysis (fit quadratic)
        if n_episodes >= 3:
            try:
                coeffs = np.polyfit(times, cumsum, 2)
                features["trajectory_quadratic_coef"] = coeffs[0]  # Curvature
                features["trajectory_linear_coef"] = coeffs[1]

                # Positive quadratic = accelerating burden
                # Negative quadratic = decelerating burden
                features["trajectory_convexity"] = np.sign(coeffs[0])
            except:
                features["trajectory_quadratic_coef"] = 0
                features["trajectory_linear_coef"] = features["trajectory_slope"]
                features["trajectory_convexity"] = 0
        else:
            features["trajectory_quadratic_coef"] = 0
            features["trajectory_linear_coef"] = features["trajectory_slope"]
            features["trajectory_convexity"] = 0

        # Segmented slope analysis (quarters)
        if follow_up_days > 0 and n_episodes >= 4:
            quarter = follow_up_days / 4
            slopes = []
            for i, (start, end) in enumerate([(0, quarter), (quarter, 2*quarter),
                                               (2*quarter, 3*quarter), (3*quarter, follow_up_days + 1)]):
                mask = (times >= start) & (times < end)
                if mask.sum() >= 2:
                    segment_times = times[mask]
                    # Get cumulative sum at start and end of segment
                    segment_cumsum = cumsum[mask]
                    segment_slope, _, _, _, _ = stats.linregress(segment_times, segment_cumsum)
                    slopes.append(segment_slope)
                else:
                    slopes.append(np.nan)

            features["slope_q1"] = slopes[0]
            features["slope_q2"] = slopes[1]
            features["slope_q3"] = slopes[2]
            features["slope_q4"] = slopes[3]

            # Slope change ratio (late vs early)
            if slopes[0] > 0 and not np.isnan(slopes[3]):
                features["slope_change_ratio"] = slopes[3] / slopes[0]
            else:
                features["slope_change_ratio"] = np.nan
        else:
            features["slope_q1"] = np.nan
            features["slope_q2"] = np.nan
            features["slope_q3"] = np.nan
            features["slope_q4"] = np.nan
            features["slope_change_ratio"] = np.nan

        # Plateau detection
        if n_episodes >= 5:
            # Look for periods where cumulative burden doesn't increase much
            # Calculate derivative (burden increase rate)
            time_diffs = np.diff(times)
            burden_diffs = np.diff(cumsum)

            # Avoid division by zero
            time_diffs = np.where(time_diffs == 0, 1, time_diffs)
            rates = burden_diffs / time_diffs

            # Plateau = rate near zero for extended period
            plateau_threshold = np.percentile(rates, 25)  # Bottom quartile
            is_plateau = rates < plateau_threshold

            # Find longest plateau
            plateau_lengths = []
            current_length = 0
            for p in is_plateau:
                if p:
                    current_length += 1
                else:
                    if current_length > 0:
                        plateau_lengths.append(current_length)
                    current_length = 0
            if current_length > 0:
                plateau_lengths.append(current_length)

            features["has_plateau"] = len(plateau_lengths) > 0 and max(plateau_lengths, default=0) >= 3
            features["longest_plateau_episodes"] = max(plateau_lengths, default=0)

            # Time to plateau (if exists)
            if features["has_plateau"]:
                plateau_start_idx = np.where(is_plateau)[0]
                if len(plateau_start_idx) > 0:
                    features["plateau_onset_days"] = times[plateau_start_idx[0]]
                else:
                    features["plateau_onset_days"] = np.nan
            else:
                features["plateau_onset_days"] = np.nan
        else:
            features["has_plateau"] = False
            features["longest_plateau_episodes"] = 0
            features["plateau_onset_days"] = np.nan

        # Time to burden milestones
        if total_burden > 0:
            for pct in [25, 50, 75]:
                threshold = total_burden * pct / 100
                idx = np.searchsorted(cumsum, threshold)
                if idx < len(times):
                    features[f"time_to_{pct}pct_burden"] = times[idx]
                else:
                    features[f"time_to_{pct}pct_burden"] = np.nan
        else:
            features["time_to_25pct_burden"] = np.nan
            features["time_to_50pct_burden"] = np.nan
            features["time_to_75pct_burden"] = np.nan

        # Step pattern analysis (sudden jumps)
        if n_episodes >= 3:
            # Define a "step" as a single episode contributing > 10% of total burden
            step_threshold = total_burden * 0.1
            n_major_steps = np.sum(durations > step_threshold)
            features["n_major_steps"] = n_major_steps
            features["max_step_size_minutes"] = np.max(durations)
            features["max_step_size_pct"] = 100 * np.max(durations) / total_burden if total_burden > 0 else 0

            # Step concentration (what % of burden from top 3 episodes)
            top_3_burden = np.sum(np.sort(durations)[-3:])
            features["top3_episode_contribution_pct"] = 100 * top_3_burden / total_burden if total_burden > 0 else 0
        else:
            features["n_major_steps"] = n_episodes
            features["max_step_size_minutes"] = np.max(durations) if n_episodes > 0 else 0
            features["max_step_size_pct"] = 100
            features["top3_episode_contribution_pct"] = 100

        # Normalized trajectory area (AUC-like metric)
        # Higher = earlier burden accumulation
        if follow_up_days > 0 and n_episodes >= 2:
            # Add endpoints
            times_extended = np.concatenate([[0], times, [follow_up_days]])
            cumsum_extended = np.concatenate([[0], cumsum, [total_burden]])

            # Trapezoidal integration
            # Use trapezoid (numpy 2.x) or fallback to trapz (numpy 1.x)
            trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            auc = trapz_func(cumsum_extended, times_extended)
            # Maximum possible AUC (all burden at day 0)
            max_auc = total_burden * follow_up_days
            features["trajectory_auc_normalized"] = auc / max_auc if max_auc > 0 else 0.5
        else:
            features["trajectory_auc_normalized"] = 0.5

        return features

    def _empty_features(self, patient_id: int) -> dict:
        """Return empty features for patient with no episodes."""
        features = {self.id_column: patient_id}
        for name in self.get_feature_names():
            if name != self.id_column:
                features[name] = np.nan if name not in ["has_plateau", "n_major_steps"] else 0
        return features

    def create_normalized_trajectory(
        self,
        df: pd.DataFrame,
        patient_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a normalized trajectory for a single patient.

        Useful for trajectory clustering and visualization.

        Args:
            df: Episode DataFrame (already filtered to patient).
            patient_id: Patient ID (for reference).

        Returns:
            Tuple of (time_points, cumulative_burden) arrays.
        """
        df = df.sort_values(self.time_column)
        times = df[self.time_column].values
        durations = df[self.duration_column].values

        if len(times) == 0:
            return np.linspace(0, 1, self.n_timepoints), np.zeros(self.n_timepoints)

        cumsum = np.cumsum(durations)
        follow_up_days = min(times[-1], self.max_follow_up_days)

        # Add endpoints
        times_extended = np.concatenate([[0], times])
        cumsum_extended = np.concatenate([[0], cumsum])

        # Interpolate to fixed grid
        time_grid = np.linspace(0, follow_up_days, self.n_timepoints)

        try:
            interp_func = interp1d(times_extended, cumsum_extended,
                                   kind='linear', fill_value='extrapolate')
            burden_grid = interp_func(time_grid)
            burden_grid = np.maximum(burden_grid, 0)  # No negative values
        except:
            burden_grid = np.zeros(self.n_timepoints)

        return time_grid, burden_grid

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return [
            "trajectory_slope",
            "trajectory_intercept",
            "trajectory_r_squared",
            "trajectory_slope_pvalue",
            "trajectory_slope_stderr",
            "trajectory_quadratic_coef",
            "trajectory_linear_coef",
            "trajectory_convexity",
            "slope_q1",
            "slope_q2",
            "slope_q3",
            "slope_q4",
            "slope_change_ratio",
            "has_plateau",
            "longest_plateau_episodes",
            "plateau_onset_days",
            "time_to_25pct_burden",
            "time_to_50pct_burden",
            "time_to_75pct_burden",
            "n_major_steps",
            "max_step_size_minutes",
            "max_step_size_pct",
            "top3_episode_contribution_pct",
            "trajectory_auc_normalized",
        ]
