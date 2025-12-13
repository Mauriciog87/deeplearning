#!/usr/bin/env python
"""
RL Roulette GUI Launcher

Launch the prediction GUI with optional model path.

Usage:
    python roulette_gui.py [model_path]
    
Examples:
    python roulette_gui.py
    python roulette_gui.py models/roulette_agent.pt
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui.app import main

if __name__ == "__main__":
    main()
