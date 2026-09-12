"""
Repository pattern for roulette data operations.

Provides high-level operations for training data management.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from src.database.models import Database, Session, Spin, Prediction
from src.datasets import EvaluationSession
from src.settlement import validate_number


class RouletteRepository:
    """
    High-level repository for roulette data operations.
    
    Provides methods optimized for ML training workflows.
    """
    
    def __init__(self, db_path: str = None):
        """Initialize repository with database connection."""
        self.db = Database(db_path)
    
    # ==================== Session Management ====================
    
    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions with spin counts."""
        return self.db.get_all_sessions()
    
    def get_numbers_by_session(self, session_id: int) -> List[int]:
        """Get all numbers for a specific session."""
        return self.db.get_all_numbers(session_id)
    
    def get_all_numbers(self) -> List[int]:
        """Get all numbers from all sessions."""
        return self.db.get_all_numbers()

    def get_evaluation_sessions(self, session_id=None):
        with self.db.transaction() as connection:
            query = '''SELECT s.id AS session_id, s.source, sp.id AS spin_id, sp.number
                       FROM sessions s LEFT JOIN spins sp ON sp.session_id=s.id'''
            parameters = ()
            if session_id is not None:
                query += ' WHERE s.id=?'
                parameters = (session_id,)
            query += ' ORDER BY s.id, sp.timestamp, sp.id'
            records = connection.execute(query, parameters).fetchall()
        grouped = {}
        for record in records:
            item = grouped.setdefault(record['session_id'], {'numbers': [], 'ids': [], 'source': record['source']})
            if record['spin_id'] is not None:
                item['numbers'].append(record['number'])
                item['ids'].append(str(record['spin_id']))
        return [EvaluationSession(str(key), tuple(item['numbers']), tuple(item['ids']), item['source'])
                for key, item in grouped.items()]
    
    # ==================== Data Ingestion ====================
    
    def create_session_with_numbers(
        self, 
        numbers: List[int],
        name: str = None,
        source: str = "manual",
        casino: str = "",
        notes: str = ""
    ) -> Session:
        """
        Create a new session and populate it with numbers.
        
        Args:
            numbers: List of roulette numbers (0-36)
            name: Session name (auto-generated if not provided)
            source: Source of data ('manual', 'online', 'simulated')
            casino: Casino name (optional)
            notes: Additional notes
            
        Returns:
            Created Session object
        """
        if name is None:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        numbers = [validate_number(number) for number in numbers]
        with self.db.transaction() as connection:
            cursor = connection.execute('INSERT INTO sessions(name, source, casino, notes) VALUES (?, ?, ?, ?)',
                                        (name, source, casino, notes))
            session = Session(id=cursor.lastrowid, name=name, source=source, casino=casino, notes=notes)
            for number in numbers:
                self.db._insert_spin(connection, session.id, number)
        
        return session
    
    def import_from_csv(self, filepath: str, session_name: str = None) -> Session:
        """
        Import numbers from a CSV file.
        
        Expected format: One number per line, or comma-separated.
        """
        import csv
        
        numbers = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for val in row:
                    val = val.strip()
                    if val.isdigit():
                        n = int(val)
                        if 0 <= n <= 36:
                            numbers.append(n)
        
        if not numbers:
            raise ValueError(f"No valid numbers found in {filepath}")
        
        name = session_name or f"Import from {filepath}"
        return self.create_session_with_numbers(numbers, name, source="csv_import")
    
    def import_from_string(self, text: str, session_name: str = None) -> Session:
        """
        Import numbers from a string.
        
        Accepts formats: "1,2,3,4" or "1 2 3 4" or "1\n2\n3\n4"
        """
        import re
        
        # Extract all numbers from the string
        numbers = []
        for match in re.finditer(r'\d+', text):
            n = int(match.group())
            if 0 <= n <= 36:
                numbers.append(n)
        
        if not numbers:
            raise ValueError("No valid roulette numbers found in input")
        
        name = session_name or f"Manual input {datetime.now().strftime('%H:%M')}"
        return self.create_session_with_numbers(numbers, name, source="manual")
    
    # ==================== Training Data ====================
    
    def get_training_sequences(
        self, 
        session_id: int = None,
        sequence_length: int = 20,
        include_target: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get sequences suitable for training.
        
        Args:
            session_id: Session to use (None for all sessions)
            sequence_length: Length of each sequence
            include_target: Whether to include target (next number)
            
        Returns:
            (sequences, targets) arrays or just sequences if include_target=False
        """
        if sequence_length < 1:
            raise ValueError('Sequence length must be positive')
        sequences, targets = [], []
        for session in self.get_evaluation_sessions(session_id):
            for i in range(len(session.numbers) - sequence_length):
                sequences.append(session.numbers[i:i + sequence_length])
                if include_target:
                    targets.append(session.numbers[i + sequence_length])
        if not sequences:
            raise ValueError('No session has enough observations for a training sequence')

        X = np.array(sequences, dtype=np.int32)
        
        if include_target:
            y = np.array(targets, dtype=np.int32)
            return X, y
        
        return X, None
    
    def get_training_data_with_context(
        self,
        session_id: int = None,
        sequence_length: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Get comprehensive training data including context features.
        
        Returns dict with:
            - sequences: (N, seq_len) array of numbers
            - targets: (N,) array of next numbers
            - prev_colors: (N, seq_len) array of color encodings
            - prev_dozens: (N, seq_len) array of dozen encodings
        """
        sequences, targets = self.get_training_sequences(session_id, sequence_length)
        return {
            'sequences': sequences, 'targets': targets,
            'colors': np.asarray([[0 if n == 0 else 1 if n in Spin.RED_NUMBERS else 2 for n in seq] for seq in sequences], dtype=np.int32),
            'dozens': np.asarray([[0 if n == 0 else (n - 1) // 12 + 1 for n in seq] for seq in sequences], dtype=np.int32),
        }

    # ==================== Statistics & Analysis ====================
    
    def compute_transition_matrix(self, session_id: int = None) -> np.ndarray:
        """
        Compute transition probability matrix P[i,j] = P(next=j | current=i).
        
        Returns:
            37x37 matrix of transition probabilities
        """
        counts = np.zeros((37, 37), dtype=np.float32)
        for session in self.get_evaluation_sessions(session_id):
            for current, next_number in zip(session.numbers, session.numbers[1:]):
                counts[current, next_number] += 1
        if not counts.sum():
            raise ValueError('Need at least two observations within one session')

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        probabilities = counts / row_sums
        
        return probabilities
    
    def compute_statistics(self, session_id: int = None) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for the data.
        
        Returns dict with various statistics.
        """
        numbers = self.db.get_all_numbers(session_id)
        
        if not numbers:
            return {"error": "No data available"}
        
        red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        
        # Basic counts
        total = len(numbers)
        freqs = self.db.get_number_frequencies(session_id)
        
        # Color distribution
        colors = {"red": 0, "black": 0, "green": 0}
        for n in numbers:
            if n == 0:
                colors["green"] += 1
            elif n in red_numbers:
                colors["red"] += 1
            else:
                colors["black"] += 1
        
        # Odd/Even
        odd_count = sum(1 for n in numbers if n != 0 and n % 2 == 1)
        even_count = sum(1 for n in numbers if n != 0 and n % 2 == 0)
        
        # High/Low
        high_count = sum(1 for n in numbers if n >= 19)
        low_count = sum(1 for n in numbers if 1 <= n <= 18)
        
        # Dozen distribution
        dozens = {1: 0, 2: 0, 3: 0}
        for n in numbers:
            if n > 0:
                d = (n - 1) // 12 + 1
                dozens[d] += 1
        
        # Expected vs actual (chi-square test hint)
        expected_per_number = total / 37
        chi_square = sum(
            (freqs[i] - expected_per_number) ** 2 / expected_per_number
            for i in range(37)
        )
        
        # Streaks analysis
        max_streak = 1
        current_streak = 1
        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        # Most/least common
        sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_freqs[:5]
        least_common = sorted_freqs[-5:]
        
        return {
            "total_spins": total,
            "unique_numbers": len([f for f in freqs.values() if f > 0]),
            "colors": colors,
            "color_percentages": {k: v/total*100 for k, v in colors.items()},
            "odd_even": {"odd": odd_count, "even": even_count},
            "high_low": {"high": high_count, "low": low_count},
            "dozens": dozens,
            "chi_square": chi_square,
            "chi_square_critical_95": 50.998,  # df=36, alpha=0.05
            "is_random_at_95": chi_square < 50.998,
            "max_repeat_streak": max_streak,
            "most_common": most_common,
            "least_common": least_common,
            "expected_per_number": expected_per_number
        }
    
    def detect_patterns(self, session_id: int = None) -> Dict[str, Any]:
        """
        Detect potential patterns in the data.
        
        Note: These are for analysis only - roulette is random!
        """
        numbers = self.db.get_all_numbers(session_id)
        
        if len(numbers) < 50:
            return {"warning": "Need at least 50 spins for pattern analysis"}
        
        # Autocorrelation at different lags
        numbers_arr = np.array(numbers)
        mean = numbers_arr.mean()
        var = numbers_arr.var()
        
        autocorr = {}
        for lag in [1, 2, 3, 5, 10]:
            if lag < len(numbers):
                corr = np.corrcoef(numbers_arr[:-lag], numbers_arr[lag:])[0, 1]
                autocorr[f"lag_{lag}"] = float(corr)
        
        # Hot/Cold numbers (recent vs overall)
        recent_50 = numbers[-50:]
        recent_freqs = {}
        for n in recent_50:
            recent_freqs[n] = recent_freqs.get(n, 0) + 1
        
        overall_freqs = self.db.get_number_frequencies(session_id)
        total = len(numbers)
        
        hot_numbers = []
        cold_numbers = []
        
        for n in range(37):
            expected = total / 37
            actual = overall_freqs.get(n, 0)
            recent = recent_freqs.get(n, 0)
            
            # Hot if significantly above expected in recent spins
            if recent > 50/37 * 1.5:
                hot_numbers.append((n, recent))
            # Cold if not appeared in recent 50
            elif recent == 0:
                cold_numbers.append((n, actual))
        
        return {
            "autocorrelation": autocorr,
            "hot_numbers": sorted(hot_numbers, key=lambda x: x[1], reverse=True),
            "cold_numbers": cold_numbers,
            "analysis_note": "These patterns are statistical artifacts and don't predict future outcomes"
        }
    
    # ==================== Utility ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all data in the database."""
        sessions = self.db.get_all_sessions()
        total_spins = self.db.get_total_spins()
        
        return {
            "total_sessions": len(sessions),
            "total_spins": total_spins,
            "sessions": [
                {
                    "id": s.id,
                    "name": s.name,
                    "source": s.source,
                    "created_at": s.created_at.isoformat()
                }
                for s in sessions
            ],
            "database_path": str(self.db.db_path)
        }
    
    def clear_all_data(self, confirm: bool = False) -> bool:
        """Clear all data from the database. Requires confirmation."""
        if not confirm:
            raise ValueError("Must set confirm=True to clear all data")
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM spins')
        cursor.execute('DELETE FROM sessions')
        cursor.execute('DELETE FROM statistics_cache')
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        return True
    
    def save_prediction(self, prediction: Prediction) -> Prediction:
        return self.db.add_prediction(prediction)
    
    def get_prediction_accuracy(self, session_id: int = None) -> Dict[str, Any]:
        return self.db.get_prediction_accuracy(session_id)
    
    def get_spin_components(self, session_id: int = None, limit: int = None) -> List[Dict[str, Any]]:
        return self.db.get_spin_components(session_id, limit)
    
    def get_last_n_numbers(self, session_id: int, n: int = 20) -> List[int]:
        numbers = self.db.get_all_numbers(session_id)
        return numbers[-n:] if len(numbers) >= n else numbers
    
    def create_empty_session(self, name: str, source: str = "manual", casino: str = "", notes: str = "") -> Session:
        return self.db.create_session(name, source, casino, notes)
    
    def add_number_to_session(self, session_id: int, number: int) -> Spin:
        return self.db.add_spin(session_id, number)


if __name__ == "__main__":
    # Demo
    repo = RouletteRepository()
    
    # Create a session with some random numbers
    import random
    numbers = [random.randint(0, 36) for _ in range(100)]
    session = repo.create_session_with_numbers(
        numbers, 
        name="Demo Session",
        source="simulated"
    )
    
    print(f"Created session: {session.name} (ID: {session.id})")
    print(f"\nStatistics:")
    stats = repo.compute_statistics(session.id)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nTransition matrix shape: {repo.compute_transition_matrix(session.id).shape}")
    print(f"\nPatterns:")
    patterns = repo.detect_patterns(session.id)
    for key, value in patterns.items():
        print(f"  {key}: {value}")
