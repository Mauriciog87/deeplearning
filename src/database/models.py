"""
SQLite Database Models for RL Roulette.

Stores roulette spins from real sessions for training and analysis.
"""

import sqlite3
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "roulette.db"


@dataclass
class Spin:
    """Represents a single roulette spin."""
    id: Optional[int] = None
    session_id: int = 0
    number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    
    @property
    def color(self) -> str:
        if self.number == 0:
            return "green"
        return "red" if self.number in self.RED_NUMBERS else "black"
    
    @property
    def parity(self) -> str:
        if self.number == 0:
            return "zero"
        return "odd" if self.number % 2 == 1 else "even"
    
    @property
    def high_low(self) -> str:
        if self.number == 0:
            return "zero"
        return "high" if self.number >= 19 else "low"
    
    @property
    def is_even(self) -> bool:
        return self.number != 0 and self.number % 2 == 0
    
    @property
    def is_high(self) -> bool:
        return self.number >= 19
    
    @property
    def dozen(self) -> int:
        if self.number == 0:
            return 0
        return (self.number - 1) // 12 + 1
    
    @property
    def column(self) -> int:
        if self.number == 0:
            return 0
        return ((self.number - 1) % 3) + 1
    
    @property
    def row(self) -> int:
        if self.number == 0:
            return 0
        return ((self.number - 1) // 3) + 1
    
    def get_components_display(self) -> str:
        if self.number == 0:
            return "🟢 Verde"
        color_emoji = "🔴" if self.color == "red" else "⚫"
        return f"{color_emoji} {self.color.capitalize()}, {self.parity.capitalize()}, {self.high_low.capitalize()}, D{self.dozen}, C{self.column}"
    
    @staticmethod
    def get_components_from_number(number: int) -> dict:
        spin = Spin(number=number)
        return {
            "number": number,
            "color": spin.color,
            "parity": spin.parity,
            "high_low": spin.high_low,
            "dozen": spin.dozen,
            "column": spin.column,
            "row": spin.row
        }


@dataclass
class Session:
    """Represents a roulette session (a sequence of spins)."""
    id: Optional[int] = None
    name: str = ""
    source: str = "manual"
    casino: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    spins: List[Spin] = field(default_factory=list)


@dataclass
class Prediction:
    """Represents a prediction made by the model."""
    id: Optional[int] = None
    session_id: int = 0
    spin_id: Optional[int] = None
    predicted_number: Optional[int] = None
    actual_number: Optional[int] = None
    predicted_color: Optional[str] = None
    actual_color: Optional[str] = None
    color_correct: Optional[bool] = None
    predicted_parity: Optional[str] = None
    actual_parity: Optional[str] = None
    parity_correct: Optional[bool] = None
    predicted_high_low: Optional[str] = None
    actual_high_low: Optional[str] = None
    high_low_correct: Optional[bool] = None
    predicted_dozen: Optional[int] = None
    actual_dozen: Optional[int] = None
    dozen_correct: Optional[bool] = None
    predicted_column: Optional[int] = None
    actual_column: Optional[int] = None
    column_correct: Optional[bool] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def compute_correctness(self):
        if self.actual_number is not None:
            actual = Spin.get_components_from_number(self.actual_number)
            self.actual_color = actual["color"]
            self.actual_parity = actual["parity"]
            self.actual_high_low = actual["high_low"]
            self.actual_dozen = actual["dozen"]
            self.actual_column = actual["column"]
            
            self.color_correct = self.predicted_color == self.actual_color if self.predicted_color else None
            self.parity_correct = self.predicted_parity == self.actual_parity if self.predicted_parity else None
            self.high_low_correct = self.predicted_high_low == self.actual_high_low if self.predicted_high_low else None
            self.dozen_correct = self.predicted_dozen == self.actual_dozen if self.predicted_dozen is not None else None
            self.column_correct = self.predicted_column == self.actual_column if self.predicted_column is not None else None


class Database:
    """SQLite database manager for roulette data."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                casino TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT DEFAULT ''
            )
        ''')
        
        # Spins table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                number INTEGER NOT NULL CHECK(number >= 0 AND number <= 36),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        
        # Index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_spins_session 
            ON spins(session_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_spins_timestamp 
            ON spins(timestamp)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_type TEXT NOT NULL,
                session_id INTEGER,
                data BLOB,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                spin_id INTEGER,
                predicted_number INTEGER,
                actual_number INTEGER,
                predicted_color TEXT,
                actual_color TEXT,
                color_correct INTEGER,
                predicted_parity TEXT,
                actual_parity TEXT,
                parity_correct INTEGER,
                predicted_high_low TEXT,
                actual_high_low TEXT,
                high_low_correct INTEGER,
                predicted_dozen INTEGER,
                actual_dozen INTEGER,
                dozen_correct INTEGER,
                predicted_column INTEGER,
                actual_column INTEGER,
                column_correct INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (spin_id) REFERENCES spins(id) ON DELETE SET NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_session 
            ON predictions(session_id)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictor_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                predictor_type TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                number_correct INTEGER DEFAULT 0,
                color_correct INTEGER DEFAULT 0,
                parity_correct INTEGER DEFAULT 0,
                high_low_correct INTEGER DEFAULT 0,
                dozen_correct INTEGER DEFAULT 0,
                column_correct INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, predictor_type)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictor_stats_session
            ON predictor_stats(session_id)
        ''')
        
        try:
            cursor.execute('ALTER TABLE predictions ADD COLUMN predictor_type TEXT')
        except sqlite3.OperationalError:
            pass
        
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS spin_components AS
            SELECT 
                s.id,
                s.session_id,
                s.number,
                s.timestamp,
                CASE 
                    WHEN s.number = 0 THEN 'green'
                    WHEN s.number IN (1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36) THEN 'red'
                    ELSE 'black'
                END as color,
                CASE 
                    WHEN s.number = 0 THEN 'zero'
                    WHEN s.number % 2 = 1 THEN 'odd'
                    ELSE 'even'
                END as parity,
                CASE 
                    WHEN s.number = 0 THEN 'zero'
                    WHEN s.number >= 19 THEN 'high'
                    ELSE 'low'
                END as high_low,
                CASE 
                    WHEN s.number = 0 THEN 0
                    WHEN s.number <= 12 THEN 1
                    WHEN s.number <= 24 THEN 2
                    ELSE 3
                END as dozen,
                CASE 
                    WHEN s.number = 0 THEN 0
                    ELSE ((s.number - 1) % 3) + 1
                END as col,
                CASE 
                    WHEN s.number = 0 THEN 0
                    ELSE ((s.number - 1) / 3) + 1
                END as row
            FROM spins s
        ''')
        
        conn.commit()
        conn.close()
    
    # ==================== Session Operations ====================
    
    def create_session(self, name: str, source: str = "manual", 
                       casino: str = "", notes: str = "") -> Session:
        """Create a new session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (name, source, casino, notes)
            VALUES (?, ?, ?, ?)
        ''', (name, source, casino, notes))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return Session(
            id=session_id,
            name=name,
            source=source,
            casino=casino,
            notes=notes
        )
    
    def get_session(self, session_id: int) -> Optional[Session]:
        """Get a session by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        session = Session(
            id=row['id'],
            name=row['name'],
            source=row['source'],
            casino=row['casino'],
            created_at=datetime.fromisoformat(row['created_at']),
            notes=row['notes']
        )
        
        # Load spins
        cursor.execute('''
            SELECT * FROM spins WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        
        for spin_row in cursor.fetchall():
            session.spins.append(Spin(
                id=spin_row['id'],
                session_id=spin_row['session_id'],
                number=spin_row['number'],
                timestamp=datetime.fromisoformat(spin_row['timestamp'])
            ))
        
        conn.close()
        return session
    
    def get_all_sessions(self) -> List[dict]:
        """Get all sessions with spin counts (as dicts for convenience)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.*, COUNT(sp.id) as spin_count 
            FROM sessions s 
            LEFT JOIN spins sp ON s.id = sp.session_id 
            GROUP BY s.id 
            ORDER BY s.created_at DESC
        ''')
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row['id'],
                'name': row['name'],
                'source': row['source'],
                'casino': row['casino'],
                'created_at': row['created_at'],
                'notes': row['notes'],
                'spin_count': row['spin_count']
            })
        
        conn.close()
        return sessions
    
    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its spins."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    # ==================== Spin Operations ====================
    
    def add_spin(self, session_id: int, number: int) -> Spin:
        """Add a single spin to a session."""
        if not 0 <= number <= 36:
            raise ValueError(f"Invalid roulette number: {number}")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO spins (session_id, number)
            VALUES (?, ?)
        ''', (session_id, number))
        
        spin_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return Spin(id=spin_id, session_id=session_id, number=number)
    
    def add_spins_bulk(self, session_id: int, numbers: List[int]) -> int:
        """Add multiple spins to a session efficiently."""
        for n in numbers:
            if not 0 <= n <= 36:
                raise ValueError(f"Invalid roulette number: {n}")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.executemany('''
            INSERT INTO spins (session_id, number)
            VALUES (?, ?)
        ''', [(session_id, n) for n in numbers])
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return count
    
    def get_spins(self, session_id: int = None, limit: int = None) -> List[Spin]:
        """
        Get spins, optionally filtered by session.
        
        Args:
            session_id: If provided, only get spins from this session
            limit: Maximum number of spins to return
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM spins'
        params = []
        
        if session_id is not None:
            query += ' WHERE session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        
        spins = []
        for row in cursor.fetchall():
            spins.append(Spin(
                id=row['id'],
                session_id=row['session_id'],
                number=row['number'],
                timestamp=datetime.fromisoformat(row['timestamp'])
            ))
        
        conn.close()
        return spins
    
    def get_all_numbers(self, session_id: int = None) -> List[int]:
        """Get all spin numbers as a simple list (for training)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT number FROM spins 
                WHERE session_id = ? 
                ORDER BY timestamp
            ''', (session_id,))
        else:
            cursor.execute('SELECT number FROM spins ORDER BY timestamp')
        
        numbers = [row['number'] for row in cursor.fetchall()]
        conn.close()
        return numbers
    
    def get_total_spins(self, session_id: int = None) -> int:
        """Get total number of spins."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('SELECT COUNT(*) FROM spins WHERE session_id = ?', (session_id,))
        else:
            cursor.execute('SELECT COUNT(*) FROM spins')
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    # ==================== Statistics ====================
    
    def get_number_frequencies(self, session_id: int = None) -> dict:
        """Get frequency of each number."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT number, COUNT(*) as count 
                FROM spins 
                WHERE session_id = ?
                GROUP BY number
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT number, COUNT(*) as count 
                FROM spins 
                GROUP BY number
            ''')
        
        frequencies = {i: 0 for i in range(37)}
        for row in cursor.fetchall():
            frequencies[row['number']] = row['count']
        
        conn.close()
        return frequencies
    
    def get_color_frequencies(self, session_id: int = None) -> dict:
        """Get frequency of colors."""
        numbers = self.get_all_numbers(session_id)
        red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        
        colors = {"red": 0, "black": 0, "green": 0}
        for n in numbers:
            if n == 0:
                colors["green"] += 1
            elif n in red_numbers:
                colors["red"] += 1
            else:
                colors["black"] += 1
        
        return colors
    
    def add_prediction(self, prediction: Prediction) -> Prediction:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                session_id, spin_id, predicted_number, actual_number,
                predicted_color, actual_color, color_correct,
                predicted_parity, actual_parity, parity_correct,
                predicted_high_low, actual_high_low, high_low_correct,
                predicted_dozen, actual_dozen, dozen_correct,
                predicted_column, actual_column, column_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.session_id, prediction.spin_id,
            prediction.predicted_number, prediction.actual_number,
            prediction.predicted_color, prediction.actual_color,
            1 if prediction.color_correct else 0 if prediction.color_correct is not None else None,
            prediction.predicted_parity, prediction.actual_parity,
            1 if prediction.parity_correct else 0 if prediction.parity_correct is not None else None,
            prediction.predicted_high_low, prediction.actual_high_low,
            1 if prediction.high_low_correct else 0 if prediction.high_low_correct is not None else None,
            prediction.predicted_dozen, prediction.actual_dozen,
            1 if prediction.dozen_correct else 0 if prediction.dozen_correct is not None else None,
            prediction.predicted_column, prediction.actual_column,
            1 if prediction.column_correct else 0 if prediction.column_correct is not None else None
        ))
        
        prediction.id = cursor.lastrowid
        conn.commit()
        conn.close()
        return prediction
    
    def get_prediction_accuracy(self, session_id: int = None) -> dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        base_query = '''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_number = actual_number THEN 1 ELSE 0 END) as number_correct,
                SUM(color_correct) as color_correct,
                SUM(parity_correct) as parity_correct,
                SUM(high_low_correct) as high_low_correct,
                SUM(dozen_correct) as dozen_correct,
                SUM(column_correct) as column_correct
            FROM predictions
            WHERE actual_number IS NOT NULL
        '''
        
        if session_id:
            cursor.execute(base_query + ' AND session_id = ?', (session_id,))
        else:
            cursor.execute(base_query)
        
        row = cursor.fetchone()
        conn.close()
        
        total = row['total'] or 0
        if total == 0:
            return {"total": 0, "message": "No predictions with results yet"}
        
        return {
            "total": total,
            "number": {"correct": row['number_correct'] or 0, "pct": (row['number_correct'] or 0) / total * 100},
            "color": {"correct": row['color_correct'] or 0, "pct": (row['color_correct'] or 0) / total * 100},
            "parity": {"correct": row['parity_correct'] or 0, "pct": (row['parity_correct'] or 0) / total * 100},
            "high_low": {"correct": row['high_low_correct'] or 0, "pct": (row['high_low_correct'] or 0) / total * 100},
            "dozen": {"correct": row['dozen_correct'] or 0, "pct": (row['dozen_correct'] or 0) / total * 100},
            "column": {"correct": row['column_correct'] or 0, "pct": (row['column_correct'] or 0) / total * 100}
        }
    
    def upsert_predictor_stats(
        self, 
        session_id: int, 
        predictor_type: str,
        total_predictions: int,
        number_correct: int,
        color_correct: int,
        parity_correct: int,
        high_low_correct: int,
        dozen_correct: int,
        column_correct: int
    ):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictor_stats (
                session_id, predictor_type, total_predictions,
                number_correct, color_correct, parity_correct,
                high_low_correct, dozen_correct, column_correct, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id, predictor_type) DO UPDATE SET
                total_predictions = excluded.total_predictions,
                number_correct = excluded.number_correct,
                color_correct = excluded.color_correct,
                parity_correct = excluded.parity_correct,
                high_low_correct = excluded.high_low_correct,
                dozen_correct = excluded.dozen_correct,
                column_correct = excluded.column_correct,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            session_id, predictor_type, total_predictions,
            number_correct, color_correct, parity_correct,
            high_low_correct, dozen_correct, column_correct
        ))
        
        conn.commit()
        conn.close()
    
    def get_predictor_stats(self, session_id: int) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictor_stats WHERE session_id = ?
        ''', (session_id,))
        
        results = []
        for row in cursor.fetchall():
            total = row['total_predictions'] or 0
            if total == 0:
                pcts = {cat: 0.0 for cat in ['number', 'color', 'parity', 'high_low', 'dozen', 'column']}
            else:
                pcts = {
                    'number': (row['number_correct'] or 0) / total * 100,
                    'color': (row['color_correct'] or 0) / total * 100,
                    'parity': (row['parity_correct'] or 0) / total * 100,
                    'high_low': (row['high_low_correct'] or 0) / total * 100,
                    'dozen': (row['dozen_correct'] or 0) / total * 100,
                    'column': (row['column_correct'] or 0) / total * 100
                }
            
            results.append({
                'predictor_type': row['predictor_type'],
                'total': total,
                'number_correct': row['number_correct'] or 0,
                'color_correct': row['color_correct'] or 0,
                'parity_correct': row['parity_correct'] or 0,
                'high_low_correct': row['high_low_correct'] or 0,
                'dozen_correct': row['dozen_correct'] or 0,
                'column_correct': row['column_correct'] or 0,
                'accuracies': pcts
            })
        
        conn.close()
        return results
    
    def get_best_predictor_per_category(self, session_id: int) -> dict:
        stats = self.get_predictor_stats(session_id)
        
        categories = ['number', 'color', 'parity', 'high_low', 'dozen', 'column']
        best = {}
        
        for cat in categories:
            best_pred = None
            best_acc = -1.0
            
            for s in stats:
                if s['total'] > 0:
                    acc = s['accuracies'][cat]
                    if acc > best_acc:
                        best_acc = acc
                        best_pred = s['predictor_type']
            
            best[cat] = {'predictor': best_pred, 'accuracy': best_acc if best_pred else 0.0}
        
        return best
    
    def get_spin_components(self, session_id: int = None, limit: int = None) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM spin_components'
        params = []
        
        if session_id:
            query += ' WHERE session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'session_id': row['session_id'],
                'number': row['number'],
                'timestamp': row['timestamp'],
                'color': row['color'],
                'parity': row['parity'],
                'high_low': row['high_low'],
                'dozen': row['dozen'],
                'column': row['col'],
                'row': row['row']
            })
        
        conn.close()
        return results


if __name__ == "__main__":
    # Quick test
    db = Database()
    print(f"Database initialized at: {db.db_path}")
    
    # Create a test session
    session = db.create_session("Test Session", source="simulated")
    print(f"Created session: {session.id}")
    
    # Add some spins
    import random
    numbers = [random.randint(0, 36) for _ in range(10)]
    db.add_spins_bulk(session.id, numbers)
    print(f"Added {len(numbers)} spins: {numbers}")
    
    # Get frequencies
    freqs = db.get_number_frequencies(session.id)
    print(f"Frequencies: {freqs}")
