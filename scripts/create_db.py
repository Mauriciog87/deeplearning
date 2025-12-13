#!/usr/bin/env python
"""
Script to create the SQLite database for RL Roulette.
This uses the Database class to create the schema if it doesn't exist.

Usage:
  python scripts/create_db.py
"""

from src.database.models import Database


def main():
    db = Database()
    print(f"Database initialized at: {db.db_path}")


if __name__ == '__main__':
    main()
