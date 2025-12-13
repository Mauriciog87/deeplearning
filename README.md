# RL Roulette 🎰

Deep Reinforcement Learning agent for European roulette using DQN with BatchNorm architecture, LSTM neural networks, and ensemble prediction models.

## Overview

This project implements a sophisticated prediction system that combines multiple ML approaches for European roulette (0-36):

- **47 possible actions**: All straight bets (0-36), outside bets (Red/Black, Odd/Even, Low/High, Dozens), and PASS
- **Real data support**: Train on actual roulette numbers stored in SQLite database
- **Statistical analysis**: Chi-square tests, autocorrelation, pattern detection, O-U test
- **Modern architecture**: BatchNorm + Dense layers (inspired by FAIRS-Roulette research)
- **LSTM Predictor**: Sequence-based neural network with GPU acceleration (inspired by NeuralRoulette-AI)
- **Fuzzy Adaptive Exploration**: Intelligent epsilon control using fuzzy logic
- **Near-miss Analysis**: Wheel and table proximity features based on behavioral research
- **Behavioral Agents**: Simulate human betting patterns (Gambler's Fallacy, Hot Hand)
- **Bias Detection**: Formal Chi-square tests with Wilson confidence intervals
- **Hyper-Heuristic Agent**: RL-based meta-learner that selects betting strategies (based on Li et al., 2024)
- **Walk-Forward Backtesting**: Rigorous strategy validation with Kelly criterion
- **GUI Application**: Modern CustomTkinter interface with real-time predictions

## Features

- 🎯 **47-Action Space**: Bet on any number or combination
- 📊 **SQLite Database**: Store and analyze real casino data
- 🧠 **Multi-Model Predictions**: LSTM, DQN, ExtraTrees, and Bias-based predictors
- 🖥️ **Modern GUI**: CustomTkinter interface with probability visualizations
- 🚀 **GPU Acceleration**: CUDA support for RTX 3060 and similar GPUs
- 📈 **Capital Context**: Agent considers current bankroll in decisions
- 🔄 **Double DQN**: Stable value estimation
- 📉 **Advanced Statistics**: JSD, Wasserstein distance, anomaly detection
- 🎛️ **Fuzzy Epsilon Control**: Adaptive exploration based on performance
- 🎯 **Near-Miss Features**: Wheel and table proximity for behavioral insights
- 🤖 **Behavioral Agents**: Simulate real human betting patterns
- 🧬 **Hyper-Heuristic Agent**: Meta-level Q-Learning to select from 10 betting strategies

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0 (with CUDA support recommended)
- gymnasium>=0.29
- numpy
- matplotlib
- tqdm
- scipy
- scikit-learn
- customtkinter>=5.0

### GPU Support (Recommended)

For NVIDIA GPUs (RTX 3060, etc.):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Quick Start

### 1. Launch the GUI (Recommended)

```bash
# Start the prediction GUI
python roulette_gui.py

# With a pre-trained DQN model
python roulette_gui.py models/roulette_agent.pt
```

The GUI provides:
- Session management (create/load sessions)
- Real-time number input
- Multi-model predictions (LSTM, DQN, ExtraTrees, Bias)
- Probability bars for all categories
- Top-10 number fan visualization
- Predictor accuracy comparison table

### 2. Train an agent

```bash
# Train with simulated random data
python train.py --episodes 1000

# Train with Fuzzy Adaptive epsilon control
python train.py --episodes 1000 --fuzzy-adaptive

# Train with real data from database
python train.py --use-real-data --episodes 500

# Resume training from checkpoint
python train.py --resume models/checkpoint.pt --episodes 500
```

### 2. Test the agent

```bash
# Test with simulated data
python test.py --episodes 100

# Test with real data
python test.py --use-real-data --episodes 100

# Compare trained vs random agent
python test.py --compare-random --episodes 100
```

### 3. Manage real data

```bash
# Add numbers manually
python roulette_cli.py add 17 23 0 5 32

# Interactive mode (enter numbers one by one)
python roulette_cli.py interactive

# Import from CSV
python roulette_cli.py import casino_data.csv

# View statistics
python roulette_cli.py stats

# List sessions
python roulette_cli.py sessions
```

## Database & Data Dump

This project provides utilities to create and export the SQLite schema and data used for training and analysis:

- Initialize a fresh local database (creates schema):
```
python scripts/create_db.py
```

- Generate an SQL dump of your local database (saved to `data/roulette_dump.sql`):
```
python scripts/dump_db.py
```

- Recreate a database from the tracked SQL dump:
```
sqlite3 data/roulette.db < data/roulette_dump.sql
```

We track `data/roulette_dump.sql` so it can be shared between machines, but `data/roulette.db` (the runtime DB) remains ignored to avoid committing runtime data.

## Models & Artifacts

Model weights and large artifacts (e.g., `*.pt`, `*.pkl`, training images) are not tracked by default and should be stored outside the repository or in an object store. The `models/` folder is kept with a `.gitkeep` so you can place downloaded models there locally.

Suggested workflow for model files:
- Store models in a cloud bucket or a release asset
- Add a small download script (e.g., `scripts/get_models.py`) to fetch model files into `models/`
---

## Screen Capture & OCR

This project includes a screen capture component that uses EasyOCR and a configurable region selector to automatically capture roulette numbers shown on a casino screen. Important defaults:

- post_detection_pause: 20 seconds  — wait period after a number is detected before resuming capture (prevents duplicate detections)
- min_confidence: 0.4  — OCR confidence threshold
- inactivity_timeout: 60 seconds — if no new number is detected during this time, the system clicks the configured reconnect button
- reconcile_after_reconnect: true — after reconnect, the system reads a configured history region and reconciles any missing numbers
- reconcile_count: 5 — max number of history numbers to reconcile

Use `test_capture.py` to interactively configure capture, reconnect and history regions, and the monitor settings.


## Project Structure

```
rl-roulette/
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py       # DQN with BatchNorm architecture
│   │   ├── fuzzy_adaptive.py  # Fuzzy epsilon controller
│   │   ├── behavioral.py      # Human behavior simulation agents
│   │   └── hyper_heuristic.py # RL-based Hyper-Heuristic agent
│   ├── environment/
│   │   └── roulette_env.py    # 47-action Gymnasium environment + near-miss variants
│   ├── database/
│   │   ├── models.py          # SQLite database layer + predictor stats
│   │   └── repository.py      # High-level data operations
│   ├── engine/
│   │   └── prediction_engine.py # Multi-model prediction coordinator
│   ├── gui/
│   │   ├── app.py             # Main GUI application (CustomTkinter)
│   │   └── components.py      # Reusable GUI components
│   └── utils/
│       ├── visualization.py   # Plotting utilities
│       ├── analysis.py        # Statistical analysis
│       ├── near_miss.py       # Wheel/table proximity utilities
│       ├── bias_detection.py  # Wheel bias detection (Chi-square, Wilson CI)
│       ├── predictor.py       # LSTM, ExtraTrees, DQN predictors
│       ├── statistics.py      # Advanced stats (O-U test, JSD)
│       ├── backtesting.py     # Walk-forward validation, Kelly criterion
│       └── metrics.py         # Anomaly detection, distribution metrics
├── train.py                   # Training script (DQN/Fuzzy)
├── train_hh.py               # Training script (Hyper-Heuristic)
├── test.py                    # Evaluation script
├── roulette_cli.py           # CLI for data management
├── roulette_gui.py           # GUI launcher
├── models/                    # Saved models
└── docs/                      # Research papers
```

## Architecture

### Environment

- **Observation Space**: 
  - `history`: Last 20 roulette numbers (integers 0-36)
  - `gain`: Current bankroll / initial bankroll (float)
- **Action Space**: 47 discrete actions
  - 0-36: Straight bet on that number (35:1 payout)
  - 37: Red, 38: Black (1:1 payout)
  - 39: Odd, 40: Even (1:1 payout)
  - 41: Low (1-18), 42: High (19-36) (1:1 payout)
  - 43-45: Dozens (2:1 payout)
  - 46: PASS (no bet)

### Agent

```
Input: history[20] + gain[1]
  │
  ├─→ RouletteEmbedding(37 → 64) → Flatten(1280)
  │     │
  │     └─→ BatchNormDense(1280 → 128) → BatchNormDense(128 → 128)
  │
  └─→ GainNet(1 → 32)
        │
        └─→ Concatenate → Dense(160 → 64) → Dense(64 → 47)
```

Features:
- **Embedding layer**: Learns representations for each number
- **BatchNorm**: Stabilizes training without LSTM complexity
- **Dual input**: History + capital context
- **Double DQN**: Decouples action selection from evaluation

### Database

SQLite schema:
- **sessions**: ID, name, source, notes, created_at
- **spins**: ID, session_id, number, timestamp
- **predictions**: Predicted vs actual tracking with category breakdown
- **predictor_stats**: Accuracy per predictor per category per session

## Prediction Engine

The `PredictionEngine` coordinates multiple prediction models:

```python
from src.engine import PredictionEngine

engine = PredictionEngine(model_path="models/roulette_agent.pt")

# Add historical data
engine.load_history([17, 23, 0, 5, 32, 14, 9, 22, 18, 7])

# Train models (GPU accelerated)
engine.train_sync(epochs=50)

# Get predictions from all models
predictions = engine.predict_all()

# Get consensus prediction (weighted average)
consensus = engine.get_consensus_prediction()
print(f"Predicted: {consensus.number.value} ({consensus.number.probability:.1%})")
print(f"Top 3: {consensus.top_numbers[:3]}")
```

### Available Predictors

| Predictor | Description | Min History |
|-----------|-------------|-------------|
| **LSTM** | Sequence learning neural network (GPU) | 20 spins |
| **DQN** | Pre-trained reinforcement learning agent | 20 spins |
| **ExtraTrees** | Ensemble ML from Merchie (2018) thesis | 30 spins |
| **Bias** | Frequency-based statistical analysis | 50 spins |

## Training Tips

1. **Start with simulated data** to validate the pipeline
2. **Collect 500+ real spins** before training on real data
3. **Use lower epsilon decay** (0.998) for more exploration
4. **Gamma of 0.99** works well for delayed rewards

```bash
# Recommended training command
python train.py --episodes 1000 --lr 0.0003 --gamma 0.99 --epsilon-decay 0.998
```

## Analysis

The CLI provides comprehensive statistical analysis:

```bash
python roulette_cli.py stats
```

Outputs:
- Color distribution (Red/Black/Green percentages)
- Parity distribution (Odd/Even)
- Dozen distribution
- Hot/Cold numbers
- Chi-square uniformity test
- Runs test for randomness
- Streak analysis

## Research Background

Inspired by and implementing techniques from:

### Architecture & Models
- [FAIRS-Roulette-Player](https://github.com/CTCycle/FAIRS-roulette-player) - BatchNorm architecture
- [NeuralRoulette-AI](https://github.com/devddine/NeuralRoulette-AI) - LSTM sequence prediction
- [RLette](https://ucladatares.medium.com/rlette-casino-roulette-through-reinforcement-learning-67e865843f0d) - RL approach

### Academic Papers
- **Salirrosas (2016)** - "Optimización de la predicción de resultados en la ruleta": 3% probability threshold filter, chi-square sector analysis
- **Merchie (2018)** - Thesis on casino anomaly detection: Extra Trees classifier, walk-forward backtesting
- **Li et al. (2024)** - [RL-based Hyper-Heuristics Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11232579/): Hyper-Heuristic architecture
- ZCSAR Learning Classifier Systems with R-Learning
- Reversed Roulette Wheel Selection in evolutionary algorithms

## Hyper-Heuristic Agent 🧬

The Hyper-Heuristic Agent is a meta-level RL approach that learns **which betting strategy to use when**, rather than learning actions directly.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    High-Level Strategy (HLS)                     │
│                       Q-Learning Selector                        │
│                                                                  │
│  State: [bankroll_level, trend, volatility, llh_perf, bias]     │
│                            ↓                                     │
│                    Select Best LLH                               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Low-Level Heuristics (LLH)                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ Hot Numbers │ Cold Numbers│ Sector Bet  │ Martingale  │  Pass   │
│ Anti-Martin.│ Flat Betting│ Fibonacci   │ D'Alembert  │ Random  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
```

### Available Strategies (LLH)

| Strategy | Description | Risk Level |
|----------|-------------|------------|
| HOT_NUMBERS | Bet on frequently occurring numbers | Medium |
| COLD_NUMBERS | Bet on numbers that haven't appeared | Medium |
| SECTOR_BETTING | Bet on active wheel sectors | Medium |
| MARTINGALE | Double bet after loss | High |
| ANTI_MARTINGALE | Double bet after win | Medium |
| FLAT_BETTING | Consistent bet amount | Low |
| FIBONACCI | Fibonacci sequence betting | Medium-High |
| DALEMBERT | Increase/decrease by 1 unit | Low-Medium |
| PASS_ACTION | Skip round (risk management) | None |
| RANDOM_POLICY | Random exploration | Variable |

### Training

```bash
# Train Q-Learning based Hyper-Heuristic
python train_hh.py --episodes 1000 --lr 0.1 --gamma 0.95

# Train DQN-based Hyper-Heuristic (neural network)
python train_hh.py --agent-type dqn --episodes 1000

# With bias detection enabled
python train_hh.py --episodes 1000 --enable-bias-detection

# Full configuration
python train_hh.py \
    --episodes 2000 \
    --max-steps 500 \
    --lr 0.1 \
    --gamma 0.95 \
    --epsilon-start 1.0 \
    --epsilon-end 0.05 \
    --initial-bankroll 1000 \
    --base-bet 10 \
    --enable-bias-detection \
    --use-near-miss \
    --verbose 2
```

### Usage Example

```python
from src.agents import HyperHeuristicAgent, LLHType
from src.environment import RouletteEnv

# Create agent
agent = HyperHeuristicAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    initial_bankroll=1000.0,
    base_bet=10.0
)

# Training loop
env = RouletteEnv()
obs, info = env.reset()

for episode in range(1000):
    agent.reset_episode()
    
    for step in range(500):
        # Agent selects which strategy to use
        agent.select_llh(explore=True)
        
        # Strategy selects specific action
        action, bet_amount = agent.select_action()
        
        # Execute in environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Agent learns from outcome
        outcome = info.get('winning_number', 0)
        agent.update(outcome, action, reward)
        
        if done:
            break

# Check statistics
print(agent.get_statistics())
```

## Important Note

⚠️ **This is an educational project.** Roulette is a game with negative expected value - the house always wins in the long run. No AI can overcome the mathematical house edge. This project explores RL techniques, not gambling strategies.

## Screenshots

### GUI Application
The modern CustomTkinter interface shows:
- Real-time predictions with confidence percentages
- Category probability bars (Color, Parity, High/Low, Dozen, Column)
- Top-10 predicted numbers with roulette colors
- Predictor accuracy comparison table

## License

MIT
