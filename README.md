# RL Roulette

A research application for European roulette (0–36), with a simulator, betting agents, probability forecasts, SQLite session storage, a CLI, and a CustomTkinter GUI.

The simulator uses independent uniform outcomes by default. Every ordinary bet has expected net return −1/37 per unit staked under that model. PASS stakes nothing and earns zero. Historical patterns alone do not establish an advantage over an independent fair wheel.

## Install

The validated interpreter is **Python 3.12 on Windows x64**. Use a virtual environment; the system Python may be a different version.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```

Choose a profile:

| File | Includes |
| --- | --- |
| requirements-core.txt | Database, simulation, statistical analysis, baseline evaluation and plots; no Torch |
| requirements-ml.txt | Core plus PyTorch and scikit-learn |
| requirements-gui.txt | ML plus CustomTkinter |
| requirements-capture.txt | ML plus EasyOCR, torchvision, MSS, Pillow and PyAutoGUI |
| requirements.txt | All profiles |

Install the matching PyTorch/torchvision pair before the ML or full profile. These commands use the versions pinned in this repository.

CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CUDA 12.4, with a compatible NVIDIA driver:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `--device cpu`, `--device cuda`, or `--device auto`. Explicit CUDA requests fail if CUDA is unavailable. EasyOCR may download its recognition models on first use. Import checks do not perform OCR or validate a live capture region.

Direct dependencies are pinned. Transitive packages are resolved by pip; evaluation manifests record the installed scientific package versions.

## Run

After activating the environment, use:

```powershell
python roulette_gui.py --device cpu
python roulette_gui.py models/roulette_agent.pt --device cuda
python roulette_cli.py --help
python roulette_cli.py session create "Wheel A"
python roulette_cli.py add 7 0 12 7 --session "Wheel A"
python roulette_cli.py predict --session 1
python test_capture.py
```

Keep independent wheels and recording sessions separate. The database returns observations in timestamp/ID order. Evaluations, training sequences, transitions, and rolling diagnostics do not concatenate sessions.

The GUI shows LSTM, ExtraTrees and frequency-based forecasts separately from the DQN betting policy. Models report ready, untrained, insufficient data, unavailable or failed. When no forecast model is active, the displayed distribution is explicitly labeled **Fair baseline**. It is not an ensemble.

A prediction is recorded before the next result is entered. Adding that result settles the pending records in the same database transaction. Accuracy tables are derived from those records, scoped to the selected session. Repeated requests for the same prediction are idempotent. CLI baselines displayed during prediction are also logged.

GUI training and backtests use snapshots. Completion, failure and cancellation return through the UI event queue. Results for an obsolete session or history version are discarded. Closing the window cancels work; it does not publish late results.

## Temporal evaluation

```powershell
python roulette_cli.py evaluate --session 1 --device cpu --output reports/evaluation.json
python roulette_cli.py evaluate --models --runs 1 --no-intervals
python roulette_cli.py evaluate --model models/roulette_agent.pt --device cuda
```

An empty `--models` list runs baselines only. The default configuration is:

| Parameter | Default |
| --- | --- |
| Train / test / step | 500 / 100 / 100 observations, within each session |
| Training runs | 5, with seeds 42–46 |
| LSTM epochs | 30 per training fold and run |
| Betting projection | Top 5 numbers, 1 unit **per number** |
| Initial bankroll | 1,000 per model, run and session |
| Bootstrap | 2,000 resamples, 95% intervals |
| Calibration | At most 10 adaptive bins, target 20 observations per bin |
| Device | auto |

Overlapping test folds are rejected. LSTM and ExtraTrees are fitted only on each training window. Forecasts use the available prefix before each test outcome. Parameters remain fixed during a test fold; the observed prefix advances after each result. Insufficient sessions are reported and skipped.

Each forecast supplies a validated probability vector over all 37 outcomes and a complete ranking. Number and category displays come from the same vector, including zero. Q-values are not converted to outcome probabilities.

The harness reports Top-1/3/5/10, log loss, multiclass Brier score, confidence ECE, classwise ECE, top-k calibration, actual exposure, net profit, ROI and drawdown. Log loss uses a reported probability floor of 1e-12. Calibration errors and adaptive bins are computed separately within each training run, then errors are averaged across runs with equal weight. Repeating identical runs does not change these metrics. Tied probabilities stay together; classwise ECE computes each class error before averaging. Classwise calibration does not establish joint calibration, and top-k calibration concerns the probability of the selected set. Exported reliability bins include their run ID.

Two full-information policies compare all 47 actions through the shared settlement rules. `expected_value` maximizes expected net profit from smoothed history frequencies; `expected_value_cs` maximizes a lower expected-profit bound from the session's sequential confidence region. Both choose PASS unless the selected score is positive and the stake is affordable. Policy rows report profit and exposure, without treating actions as probability forecasts. Counterfactual replay assumes actions cannot affect outcomes or which outcomes are observed.

LSTM inputs use a 37-category one-hot representation by default. `evaluate --lstm-representation ordinal` retains the previous scalar `number/36` representation as an explicit ablation. Checkpoints record the representation; existing version-2 scalar checkpoints load as ordinal models. CPU loading preserves an explicitly selected CPU device. Exact train/save/resume continuation is tested for both representations; representation choice alone does not establish better predictions.

Optional chronological recalibration is available through `evaluate --recalibrators temperature mcllo normalized_isotonic --calibration-window 100`. The calibration window is reserved from the end of `--train-window`: the predictor fits the earlier portion, forecasts each calibration outcome before seeing it, and stays frozen while the calibrator fits. Testing begins afterward. The uncalibrated and calibrated variants use the same predictor and test outcomes. The manifest exports disjoint training/calibration/test IDs, fitted calibrator state, optimization results and error allocations. No calibrator is selected or tuned using test labels.

Temperature scaling fits one positive inverse temperature. MCLLO uses reference-category linear log odds with bounded parameters and a ridge penalty toward identity. Normalized isotonic calibration uses the paper's multiclass normalized likelihood on fixed PAVA blocks, optimized in monotone log values. The latter is a convex finite problem and reports a first-order bound on objective suboptimality. It is an adaptation of the paper's objective, not its MCMC algorithm or an unrestricted functional optimum. All three remain experimental; input probability floors, regularization, bounds and optimization failures are reported.

With five straight bets, a hit earns +31 units net and a miss loses 5. An unaffordable betting projection becomes PASS. Capital carries across folds within each model/run/session. ROI divides total net profit by total money staked. PASS has zero profit and no defined ROI. Policy-only rows have no forecast scores.

DQN evaluation requires a versioned checkpoint with training provenance. A real-data checkpoint is excluded from any fold that overlaps its training observations or cannot be matched to its dataset. Its action and stake are evaluated separately from probability forecasts. Paired profit per observation allows comparison with PASS.

Intervals resample training runs and circular temporal blocks within sessions. Each temporal draw is shared across seeds, so repeating a baseline under several seeds does not create extra independent observations. Paired comparisons reuse the same observations and resampled indices. They are conditional on this dataset and assume approximately stationary dependence within blocks. They do not establish future profitability. The JSON export includes rows, configuration, seeds, dataset SHA-256, source/session/spin IDs, fold partitions, model status and executed devices.

Calibration uses separate inference. Adaptive ECE has no percentile-bootstrap interval: that construction can exclude zero for a perfectly calibrated predictor. Evaluation JSON schema 3 replaces `ece_interval`, `classwise_ece_interval`, and `top_k_ece_intervals` with `calibration_bounds`. These bounds cover the mean absolute cumulative conditional residuals within fixed, equal-width bins, averaged over the configured runs and channels. They use a Hilbert-space martingale bound with a summable error allocation over dyadic time horizons. They are conservative, simultaneous over time and the declared model/metric family, and do not assume independent seed replicas. All configured runs must cover the same outcomes; otherwise inference is unavailable. Forecasts, rankings and inclusion decisions must precede each outcome, and the configuration must be fixed in advance. The bounds do not estimate population l2-ECE or certify joint calibration. The JSON records the target, bin edges, error allocation, assumptions and unique observation count. See the derivation in [the implementation plan](RESEARCH_IMPLEMENTATION_PLAN.md).

`joint_calibration` adds descriptive total variation within fixed cells of the full 37-class forecast vector and a sequential likelihood diagnostic. Its alternative learns from previous labels in those cells. Run likelihood processes are averaged, so duplicated seeds do not multiply evidence. The null requires each announced vector to equal the conditional outcome law given the monitored past and current forecasts; this is stronger than calibration given only the forecast. A rejection may reveal dependence as well as a forecast mismatch. Non-rejection does not certify calibration. The method shares the predetermined calibration error budget and reports impossible-support outcomes explicitly without nonfinite JSON numbers.

The GUI Backtest button uses this same harness and its defaults. Passive replay can settle any roulette action from an observed result without propensity weighting, provided bets do not affect outcomes or which outcomes were observed. The legacy OPE section lists additional logging needed for action-dependent IPS/DR applications.

## Train and resume agents

```powershell
python train.py --episodes 100 --seed 42 --device cpu --unit-stake 1
python train.py --use-real-data --episodes 100 --seed 42 --device cuda
python train.py --resume models/roulette_agent.pt --episodes 100 --seed 42 --device cpu
python test.py --model models/roulette_agent.pt --use-real-data --device cpu
python train_hh.py --agent-type qlearning --episodes 100 --base-bet 1
python test_hh.py --model models/hh_agent_final.json --base-bet 1
python compare_agents.py --base-bet 1
```

Real-data DQN training reserves the last 20% of each session chronologically. Testing requires the checkpoint's exact dataset partition and visits the reserved outcomes once, using the preceding history only as context. Missing data or checkpoints produce errors; training and testing never silently switch to synthetic data.

Simulation seeds cover environment outcomes and agent randomness. Real sequences end without wrapping. Episode termination suppresses DQN bootstrapping; truncation preserves it. Next-action masks apply when selecting the target action. An environment validates stakes before consuming an outcome.

Hyper-heuristic strategies pass their actual stake to the environment, including progression amounts. Evaluation updates their history and bankroll with learning disabled. Training evaluation uses independent copies.

Version-2 checkpoints save online and target networks, optimizer, replay, exploration state, counters, RNG state, observation contract and metadata. Fuzzy and hyper-heuristic checkpoints also preserve controller/strategy state. Writes are atomic. Neural files load with `weights_only=True`; tabular HH uses JSON. Legacy `.pkl` and older neural checkpoints must be retrained.

Resume requires matching training configuration. Exact continuation is tested on the same device/software stack; it is not guaranteed across CPU/GPU or PyTorch releases. Agent training checkpoints are taken at episode boundaries. Experimental physics remains an explicit environment option and does not model a measured real wheel.

## Statistical diagnostics and capture

```powershell
python roulette_cli.py monitor --sessions 1 2 --state models/wheel-monitor.json --output reports/wheel-monitor.json
python roulette_cli.py randomness --session 1 --resamples 9999 --seed 42 --fdr-method by
python roulette_cli.py heatmaps --session 1 --output-dir reports/heatmaps/session_1
```

`monitor` maintains sequential Dirichlet-mixture evidence against conditional uniformity. Declare the session family before inspecting outcomes and reuse the same state file. The total error budget is divided across sessions and restart indices `r` as `alpha / (sessions * (r+1) * (r+2))`; the first segment receives half of each session's budget. Repeated calls count only new spin IDs. Changed, deleted or reordered recorded outcomes are rejected. `--reset` spends the next allocation and starts an empty segment while preserving deduplication. Starting another state file does not replenish the statistical budget.

The report includes confidence-region projections for pocket probabilities and candidates whose simultaneous lower bound exceeds the straight-bet break-even probability `1/36`. These bounds require fixed conditional probabilities throughout the monitored segment. Rejecting uniformity does not establish a persistent or profitable advantage. HH training with `--enable-bias-detection` consumes the sequential signal and preserves its monitor in checkpoints; older bias-enabled checkpoints without this state cannot resume.

Sparse frequency tables use multinomial Monte Carlo; sufficiently populated tables use chi-square asymptotics. Temporal dependence and scanned drift statistics use whole-sequence permutations, recalculating the scan in each permutation. Entropy drift compares windows with one another, so a stable nonuniform distribution does not automatically imply drift.

Benjamini–Yekutieli correction is the default for dependent families of tests. BH is an explicit alternative where supported. Heatmap alerts use exact binomial p-values; z-scores describe severity. Neither correction removes serial dependence within a spin sequence or protects unrestricted repeated monitoring. Drift inference is labeled exploratory when the dependence screen rejects its null. Sector null probabilities reflect their actual pocket counts.

The legacy number filter now uses exact binomial intervals with Bonferroni allocation across 37 pockets and compares lower bounds with `1/36`. This is fixed-sample inference under IID outcomes; use the persistent monitor for repeated looks. Its Wilson intervals remain descriptive. The OU helper reports a regression of overlapping window frequencies and the IID overlap null: lag-one correlation `(window-1)/window` and drift coefficient `1/window`. Its `expected_next` is a window-frequency extrapolation, not a next-spin probability.

OCR observations retain event IDs, timestamps, confidence and pending/accepted state. History reconciliation uses sequence overlap and preserves repeated numbers when they represent separate events. Ambiguous or low-confidence observations remain pending for review; they are not silently inserted. A green background alone is not recognized as zero.

Capture advances its local history only after persistence acknowledges the event. Retrying an event does not create another spin. Stop cancels waits and prevents a restart while an earlier worker is alive. Capture-region settings live in `data/capture_config.json`.

## Data and verification

SQLite schema version 2 enables foreign keys on every connection, cascades session deletion, validates spin values and links predictions only to spins from their own session. `data/roulette_dump.sql` contains the empty schema. The database is created on first use; an older schema requires an explicit reset rather than an implicit migration.

```powershell
python scripts/run_checks.py --profile core
python scripts/run_checks.py --profile full
python scripts/verify_runtime.py --device cpu --capture-import
python scripts/verify_runtime.py --device cuda --capture-import
python -m pip check
```

Tests use temporary databases and model files. Runtime checks actually fit LSTM/ExtraTrees, train DQN, reload checkpoints and compare the next DQN learning step. They do not validate OCR accuracy against a live site, production capture timing, or predictive advantage on new real observations.

## Sources and earlier work

Statistical and implementation references:

- [Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264): multiple runs and uncertainty in RL evaluation.
- [Nixon et al., Measuring Calibration in Deep Learning](https://arxiv.org/abs/1904.01685): calibration definitions and binning choices.
- [Vaicenavicius et al., Evaluating Model Calibration in Classification](https://proceedings.mlr.press/v89/vaicenavicius19a.html): statistical limits of calibration evaluation.
- [Small and Tse, Predicting the Outcome of Roulette](https://arxiv.org/abs/1204.6412): physical prediction uses measured wheel/ball dynamics.
- [SciPy permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [false discovery control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html), [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html), and [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Earlier repository influences include [FAIRS-Roulette-Player](https://github.com/CTCycle/FAIRS-roulette-player), [NeuralRoulette-AI](https://github.com/devddine/NeuralRoulette-AI), and the [review of RL-based hyper-heuristics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11232579/). These references motivate experiments; they do not validate this application's forecasts.

## License

MIT
