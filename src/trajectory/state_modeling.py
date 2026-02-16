"""
Hidden state modeling for AF trajectories.

Models AF burden as transitions between discrete states.
"""

from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class AFStateModeler:
    """Model AF burden as hidden state transitions."""

    def __init__(
        self,
        n_states: int = 3,
        state_definition: str = "burden_level",
        time_resolution: str = "weekly",
        id_column: str = "ID",
        duration_column: str = "af_episode_minutes",
        time_column: str = "days_since_implant",
        max_days: int = 365,
    ):
        """
        Initialize state modeler.

        Args:
            n_states: Number of burden states (e.g., 3 = low/medium/high).
            state_definition: How to define states:
                - 'burden_level': Based on burden tertiles
                - 'clinical': Based on clinical thresholds
                - 'quantile': Based on quantile thresholds
            time_resolution: Time unit for state sequences ('daily', 'weekly').
            id_column: Patient ID column name.
            duration_column: Episode duration column name.
            time_column: Time since implant column name.
            max_days: Maximum follow-up period.
        """
        self.n_states = n_states
        self.state_definition = state_definition
        self.time_resolution = time_resolution
        self.id_column = id_column
        self.duration_column = duration_column
        self.time_column = time_column
        self.max_days = max_days

        # State thresholds (set during fitting)
        self.thresholds: Optional[np.ndarray] = None
        self.state_labels: List[str] = []

    def define_states(
        self,
        df: pd.DataFrame,
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Define burden states based on data or provided thresholds.

        Args:
            df: Episode DataFrame.
            thresholds: Optional manual thresholds (in minutes per time unit).

        Returns:
            Tuple of (threshold array, state labels).
        """
        if thresholds is not None:
            self.thresholds = np.array(thresholds)
        else:
            # Calculate burden per time unit across all patients
            burden_per_unit = self._calculate_burden_per_unit(df)

            if self.state_definition == "burden_level":
                # Use tertiles
                self.thresholds = np.percentile(
                    burden_per_unit[burden_per_unit > 0],
                    [100 / self.n_states * i for i in range(1, self.n_states)]
                )
            elif self.state_definition == "clinical":
                # Clinical thresholds (example: <5%, 5-20%, >20% burden)
                if self.time_resolution == "weekly":
                    # Minutes per week thresholds
                    self.thresholds = np.array([50.4, 201.6])  # 0.5%, 2% of week
                else:
                    # Minutes per day thresholds
                    self.thresholds = np.array([7.2, 28.8])  # 0.5%, 2% of day
            else:
                # Default to tertiles
                self.thresholds = np.percentile(
                    burden_per_unit[burden_per_unit > 0],
                    [33, 67]
                )

        # Create state labels
        self.state_labels = [f"State_{i}" for i in range(self.n_states)]
        if self.n_states == 3:
            self.state_labels = ["Low", "Medium", "High"]
        elif self.n_states == 2:
            self.state_labels = ["Low", "High"]

        return self.thresholds, self.state_labels

    def _calculate_burden_per_unit(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate burden per time unit for threshold estimation."""
        # Aggregate burden per time unit per patient
        if self.time_resolution == "weekly":
            df = df.copy()
            df["time_unit"] = (df[self.time_column] / 7).astype(int)
            unit_burden = df.groupby([self.id_column, "time_unit"])[self.duration_column].sum()
        else:
            df = df.copy()
            df["time_unit"] = df[self.time_column].astype(int)
            unit_burden = df.groupby([self.id_column, "time_unit"])[self.duration_column].sum()

        return unit_burden.values

    def create_state_sequences(
        self,
        df: pd.DataFrame,
    ) -> Dict[int, np.ndarray]:
        """
        Create state sequence for each patient.

        Args:
            df: Episode DataFrame with time alignment.

        Returns:
            Dictionary mapping patient ID to state sequence array.
        """
        if self.thresholds is None:
            self.define_states(df)

        patient_ids = df[self.id_column].unique()
        sequences = {}

        for pid in patient_ids:
            patient_df = df[df[self.id_column] == pid]
            seq = self._create_patient_sequence(patient_df)
            sequences[pid] = seq

        return sequences

    def _create_patient_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Create state sequence for a single patient."""
        # Determine number of time units
        if self.time_resolution == "weekly":
            n_units = self.max_days // 7
            unit_size = 7
        else:
            n_units = self.max_days
            unit_size = 1

        # Aggregate burden per time unit
        df = df.copy()
        df["time_unit"] = (df[self.time_column] / unit_size).astype(int)
        unit_burden = df.groupby("time_unit")[self.duration_column].sum()

        # Create full sequence with zeros for missing units
        burden_seq = np.zeros(n_units)
        for unit_idx, burden in unit_burden.items():
            if 0 <= unit_idx < n_units:
                burden_seq[unit_idx] = burden

        # Convert burden to states
        state_seq = np.digitize(burden_seq, self.thresholds)

        return state_seq

    def extract_state_features(
        self,
        sequences: Dict[int, np.ndarray],
    ) -> pd.DataFrame:
        """
        Extract features from state sequences.

        Args:
            sequences: Dictionary of patient state sequences.

        Returns:
            DataFrame with state-based features per patient.
        """
        features = []

        for pid, seq in sequences.items():
            feat = self._extract_patient_state_features(pid, seq)
            features.append(feat)

        return pd.DataFrame(features)

    def _extract_patient_state_features(
        self,
        patient_id: int,
        seq: np.ndarray,
    ) -> Dict:
        """Extract features from a single patient's state sequence."""
        features = {self.id_column: patient_id}
        n_units = len(seq)

        # Time in each state
        for i in range(self.n_states):
            state_name = self.state_labels[i] if i < len(self.state_labels) else f"State_{i}"
            features[f"pct_time_{state_name.lower()}"] = 100 * np.mean(seq == i)
            features[f"n_units_{state_name.lower()}"] = np.sum(seq == i)

        # Dominant state
        state_counts = np.bincount(seq, minlength=self.n_states)
        features["dominant_state"] = np.argmax(state_counts)
        features["dominant_state_pct"] = 100 * np.max(state_counts) / n_units

        # Final state
        features["final_state"] = seq[-1]

        # State transitions
        transitions = np.diff(seq)
        n_transitions = np.sum(transitions != 0)
        features["n_state_transitions"] = n_transitions
        features["transition_rate"] = n_transitions / (n_units - 1) if n_units > 1 else 0

        # Transition types
        features["n_increases"] = np.sum(transitions > 0)
        features["n_decreases"] = np.sum(transitions < 0)

        # First high state (if any)
        high_state = self.n_states - 1
        high_indices = np.where(seq == high_state)[0]
        if len(high_indices) > 0:
            features["first_high_state_unit"] = high_indices[0]
            features["reached_high_state"] = True
        else:
            features["first_high_state_unit"] = np.nan
            features["reached_high_state"] = False

        # State entropy (disorder measure)
        state_probs = state_counts / n_units
        state_probs = state_probs[state_probs > 0]
        features["state_entropy"] = -np.sum(state_probs * np.log2(state_probs))
        features["state_entropy_normalized"] = (
            features["state_entropy"] / np.log2(self.n_states)
            if self.n_states > 1 else 0
        )

        # Consecutive runs
        runs = self._get_runs(seq)
        if runs:
            run_lengths = [r[1] for r in runs]
            features["mean_run_length"] = np.mean(run_lengths)
            features["max_run_length"] = np.max(run_lengths)
            features["n_runs"] = len(runs)
        else:
            features["mean_run_length"] = n_units
            features["max_run_length"] = n_units
            features["n_runs"] = 1

        return features

    def _get_runs(self, seq: np.ndarray) -> List[Tuple[int, int]]:
        """Get runs (consecutive same-state periods) from sequence."""
        if len(seq) == 0:
            return []

        runs = []
        current_state = seq[0]
        current_length = 1

        for i in range(1, len(seq)):
            if seq[i] == current_state:
                current_length += 1
            else:
                runs.append((current_state, current_length))
                current_state = seq[i]
                current_length = 1

        runs.append((current_state, current_length))
        return runs

    def compute_transition_matrix(
        self,
        sequences: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Compute empirical transition matrix from all sequences.

        Args:
            sequences: Dictionary of patient state sequences.

        Returns:
            Transition probability matrix of shape (n_states, n_states).
        """
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))

        for seq in sequences.values():
            for i in range(len(seq) - 1):
                from_state = seq[i]
                to_state = seq[i + 1]
                transition_counts[from_state, to_state] += 1

        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_counts,
            row_sums,
            where=row_sums > 0,
            out=np.zeros_like(transition_counts)
        )

        return transition_matrix

    def get_state_labels(self) -> List[str]:
        """Return state labels."""
        return self.state_labels


def create_state_features(
    df: pd.DataFrame,
    n_states: int = 3,
    time_resolution: str = "weekly",
) -> pd.DataFrame:
    """
    Convenience function to create state-based features.

    Args:
        df: Episode DataFrame with time alignment.
        n_states: Number of burden states.
        time_resolution: Time unit for state sequences.

    Returns:
        DataFrame with state-based features per patient.
    """
    modeler = AFStateModeler(n_states=n_states, time_resolution=time_resolution)
    modeler.define_states(df)
    sequences = modeler.create_state_sequences(df)
    features = modeler.extract_state_features(sequences)
    return features
