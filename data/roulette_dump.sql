PRAGMA foreign_keys=ON;
PRAGMA user_version=2;
BEGIN TRANSACTION;
CREATE TABLE capture_observations (
                event_id TEXT PRIMARY KEY, session_id INTEGER NOT NULL, observed_at TEXT NOT NULL,
                numbers TEXT NOT NULL CHECK(json_valid(numbers)), confidence REAL NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                spin_id INTEGER,
                predictor_type TEXT NOT NULL DEFAULT 'statistical',
                context_count INTEGER NOT NULL DEFAULT 0,
                probabilities TEXT CHECK(probabilities IS NULL OR json_valid(probabilities)),
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'settled', 'superseded')),
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
                FOREIGN KEY (session_id, spin_id) REFERENCES spins(session_id, id) ON DELETE CASCADE
            );
CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                casino TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT DEFAULT ''
            );
CREATE TABLE spins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                number INTEGER NOT NULL CHECK(typeof(number) = 'integer' AND number >= 0 AND number <= 36),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_id TEXT,
                UNIQUE(session_id, event_id),
                UNIQUE(session_id, id),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
CREATE TABLE statistics_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_type TEXT NOT NULL,
                session_id INTEGER,
                data BLOB,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
CREATE INDEX idx_spins_session 
            ON spins(session_id)
        ;
CREATE INDEX idx_spins_timestamp 
            ON spins(timestamp)
        ;
CREATE INDEX idx_predictions_session 
            ON predictions(session_id)
        ;
CREATE VIEW predictor_stats AS
            SELECT session_id, predictor_type, COUNT(*) AS total_predictions,
                   SUM(predicted_number = actual_number) AS number_correct,
                   SUM(color_correct) AS color_correct, SUM(parity_correct) AS parity_correct,
                   SUM(high_low_correct) AS high_low_correct, SUM(dozen_correct) AS dozen_correct,
                   SUM(column_correct) AS column_correct
            FROM predictions WHERE status = 'settled' GROUP BY session_id, predictor_type;
CREATE VIEW spin_components AS
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
            FROM spins s;
DELETE FROM "sqlite_sequence";
COMMIT;
