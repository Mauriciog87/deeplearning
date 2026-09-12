# Research implementation plan

Baseline: `7f6d915`. Scope: implement the recommendations from the paper audit, verify each stage, and commit completed stages. Work directly without orchestration. Publication is separate from these local commits.

## Acceptance and delivery

- Preserve chronological prediction, dataset provenance, settlement, capture configuration, and existing user data.
- Reproduce reported statistical failures before fixing them. Test mathematical identities independently of implementation details.
- Separate descriptive metrics, fixed-sample inference, sequential inference, and experimental methods in APIs and reports.
- Label the actual estimand, assumptions, error budget, reference data, and method version. Never transfer a paper's guarantees to a different algorithm.
- Use seeded synthetic scenarios to measure coverage, false alarms, power, and detection delay. Export scenario definitions, raw results, uncertainty, and software/configuration provenance.
- Compare predictive methods chronologically with frozen choices and separate calibration data. Real-data superiority remains unestablished until eligible sessions exist; missing data must be reported explicitly.
- Make a commit after each coherent, verified stage. Record commands and results here as stages finish.

## Work packages

| Stage | Changes | Observable acceptance |
|---|---|---|
| 1. Calibration | Define aggregation over training runs; fix adaptive-bin boundaries and replica dependence; replace invalid percentile ECE intervals with inference valid for its stated target; distinguish confidence, classwise, top-set, and joint calibration; correct paper attributions. | Perfectly calibrated synthetic forecasts are not systematically excluded by intervals. Repeating identical runs leaves metrics and uncertainty unchanged. Distinct run calibration errors cannot cancel. Joint-calibration counterexamples remain detectable by the joint diagnostic. |
| 2. Sequential evidence | Implement Dirichlet-mixture multinomial evidence against a declared null, probability confidence sequences, durable monitor state, predictable updates, and explicit error allocation across streams/restarts. Integrate with the analyzer, reports, and HH inputs. | Sequential updates equal the closed-form mixture likelihood ratio. Save/reload matches uninterrupted execution. Reprocessing an event cannot increase evidence. Uniform simulations respect the declared monitoring error budget within simulation uncertainty. |
| 3. Decision and legacy statistics | Add a full-information expected-value policy with PASS; settle all 47 counterfactual actions through the shared domain. Correct profitability thresholds and simultaneous uncertainty. Describe rolling-frequency OU fits as descriptive; expose the overlap null and remove unsupported claims. | All standard bets have uniform expectation -1/37 per unit and PASS has zero. Known biased probabilities select the correct affordable action. A lower bound above 1/37 but below 1/36 cannot establish a profitable full-number bet. IID overlap produces no claim of predictive dependence. |
| 4. Forecast representation and recalibration | Add categorical LSTM inputs while retaining the ordinal form as an explicit ablation. Provide chronological calibration partitions and optional recalibration comparators, including temperature scaling, multiclass linear log odds, and normalization-aware isotonic calibration. | Input/output/checkpoint contracts survive training and resume. No calibrator sees test labels. Full 37-class vectors remain valid, including unseen classes. Reports compare log loss, Brier, calibration, and actual betting exposure on the same outcomes. |
| 5. Change detection experiments | Implement and compare an e-detector, reference-conditional conformal detection, and PITMonitor. Keep unknown-reference uncertainty, ties, start-time mixtures, and reset budgets explicit. | Formula/reference tests pass. Scenarios distinguish fixed bias, abrupt/gradual change, stationary dependence, and calibration improvement. Reports distinguish anytime false-alarm probability from average run length and alarm time from estimated change location. |
| 6. Online recalibration experiment | Implement an optional Blackwell/ORCA-style multiclass recalibration experiment with bounded-score comparison, predictable updates, and measured optimization residuals. | Updates use past labels only. Probability vectors remain valid. The reported optimization bound distinguishes certified oracle steps from approximate steps. No unconditional calibration guarantee is claimed for an approximate optimizer. |
| 7. Research benchmark and integration | Add a reproducible benchmark CLI spanning uniform, biased, dependent, drift, calibration, and policy scenarios. Compare all added methods with current baselines. Integrate relevant production defaults and expose experimental methods explicitly. | Unit/regression tests, command smoke tests, seeded statistical campaigns, and checkpoint/integration checks pass. Results include uncertainty and failures. Core import paths keep optional heavy dependencies lazy where applicable. |
| 8. Completion audit | Reconcile the implementation with every audit recommendation, refresh README/research notes, and record limitations and measured results. | Every recommendation has a code/test/result reference or its originally recommended exclusion, with no unimplemented work silently marked complete. Working tree is clean after final commit. |

## Scientific contracts

- Calibration intervals must name the target they cover. A conservative interval for predictable binned residuals is not Sun et al.'s population l2-ECE interval. IID-only diagnostics must require the corresponding evaluation design rather than silently run on adaptive walk-forward rows.
- Seed replicas share observed outcomes. Report per-run calibration and aggregate without treating replicas as new observations. The same protection applies to resampling and interval construction.
- A test against conditional uniformity does not establish profitable bias, stationarity, or a guaranteed future edge. Absence of rejection does not establish fairness.
- A change detector against an estimated reference must account for reference uncertainty. Repeated restarts or multiple streams require explicit error allocation.
- Confidence sequences for fixed probabilities require a stable conditional probability model; they do not estimate an arbitrary changing next-spin distribution.
- Monitoring PIT stability differs from testing correct calibration and from detecting deterioration. Categorical PITs require a declared ordering and randomized handling of discrete mass.
- Expected-value replay is appropriate only when actions do not affect outcomes or their observation. The full-information policy includes bankroll constraints and PASS.
- OU fits on overlapping frequency windows are descriptive. Compare them with the known overlap behavior under IID outcomes.

## Paper-to-method tracking

| Source | Intended use |
|---|---|
| [Agarwal et al.](https://arxiv.org/abs/2108.13264) | Paired, multi-run evaluation; our temporal adaptation receives separate coverage checks. |
| [Posocco and Bonnefoy](https://arxiv.org/abs/2109.03480), [Sun et al.](https://arxiv.org/abs/2408.08998), [Vaicenavicius et al.](https://proceedings.mlr.press/v89/vaicenavicius19a.html) | Correct calibration targets and inference; distinguish exact reproductions from alternative methods. |
| [Lindon and Malek](https://arxiv.org/abs/2011.03567), [Ryu and Wornell](https://arxiv.org/abs/2402.03683) | Multinomial mixture evidence and confidence-sequence comparison. |
| [Shin et al.](https://arxiv.org/abs/2203.03532), [conditional CTMs](https://arxiv.org/abs/2602.13848), [PITMonitor](https://arxiv.org/abs/2603.13156) | Distinct, optional change-detection experiments. |
| [Marx et al.](https://arxiv.org/abs/2409.19157) | Online recalibration prototype with explicit approximation limits. |
| [Multiclass LLO](https://arxiv.org/abs/2602.18573), [normalization-aware isotonic calibration](https://arxiv.org/abs/2512.09054) | Optional chronological recalibration comparators. |
| [Salirrosas](https://arxiv.org/abs/1609.09601) | Chronological evaluation; reassess OU and profitability claims. |

## Exclusions retained from the audit

- No physical prediction without the required physical measurements.
- No TimesFM, Chronos, or MOMENT integration before demonstrating useful signal and a need beyond simpler categorical models.
- No IPS/DR machinery for passive full-information replay. Action-dependent applications would require a different logging and identification design.
- No automatic promotion of experimental detectors or recalibrators based on a single favorable run.
- No claim of real-world predictive improvement from synthetic data or an empty database.

## Progress

Entries below record what was complete at each commit. A statement that work remained pending belongs to that historical stage; the completion audit records the final status.

- Planning: inspected the clean baseline and linked worktree. The current database is preserved. Implementation and verification are pending.
- Stage 1a: regression tests reproduced seed-replica dependence, cancellation between different run errors, and the 40-observation bin boundary failure. Fixed per-run aggregation, bin boundaries, and run-tagged reliability data. `C:/Python312/python.exe -B -m unittest test_calibration_contract test_temporal_evaluation test_temporal_bootstrap -v`: 9 tests passed. Interval replacement and joint diagnostics remain pending.
- Stage 1b: removed percentile intervals for adaptive ECE. Evaluation schema 3 exposes fixed-bin conditional-residual bounds, method/target metadata, a predetermined model/metric error allocation, and unavailable status for incomplete seed cohorts. Tested zero-calibration coverage, replica invariance, known large errors, invalid inputs, and 100 dependent synthetic sequences at four observation counts. `C:/Python312/python.exe -B -m unittest test_calibration_contract test_evaluation_harness test_temporal_evaluation test_temporal_bootstrap -v`: 26 tests passed. Joint calibration monitoring remains pending.
- Stage 2a: implemented an equal-weight mixture of three Dirichlet multinomial likelihood processes, projection of its confidence region onto arbitrary pocket subsets, explicit stream/restart error budgets, idempotent event IDs, and atomic JSON persistence. Sequential products match the analytic gamma-function expression; projected endpoints match the likelihood boundary. Save/reload preserves the exact null vector and continuation. Tested 100 uniform streams with 1,000 looks each and a separate large-bias scenario. `C:/Python312/python.exe -B -m unittest test_sequential_inference test_cli_integrity -v`: 11 tests passed, including evaluation JSON and CLI integrity. Analyzer/HH/CLI monitor integration remains pending.

- Stage 2b: integrated durable evidence with `WheelBiasAnalyzer`, HH training/checkpoints, and the `monitor` CLI. Session families and restart spending are explicit; the CLI checks append-only spin provenance and protects its database/state paths. Recommendations require a simultaneous lower probability bound above `1/36`. Analyzer and CLI tests cover reset spending, duplicate events, state reload, HH signal routing, configuration changes and output-path protection. A real CPU CLI smoke test trained one 60-step episode and resumed for two more; the saved monitor contained all 180 observations. This does not establish a real-wheel advantage.

`C:/Python312/python.exe -B -m unittest test_cli_integrity test_sequential_analyzer -v`: 11 tests passed, including changed/deleted-history rejection.

- Stage 3: added full-information rewards and expected-value policies for all 47 actions, with PASS, bankroll constraints and optional simultaneous confidence-region bounds. Walk-forward evaluation records their decisions before consuming outcomes. Corrected both legacy probability helpers to use `1/36` and exact fixed-sample family intervals; retained Wilson intervals as descriptive fields. The OU helper exposes the analytical IID overlap null and no longer invents a positive drift coefficient for a constant series. `C:/Python312/python.exe -B -m unittest test_expected_value test_evaluation_harness test_calibration_contract test_cli_integrity test_temporal_evaluation test_settlement_environment -v`: 46 tests passed.

- Stage 1c: added a full-vector cell diagnostic and predictable alternative likelihood process. The sequential null is explicitly conditional predictive correctness, stronger than calibration given only the forecast. A 6,000-row counterexample has zero confidence and classwise ECE but joint-cell TV 0.1 and rejection. Formula, null-simulation, save/replay, impossible-support, missing-cohort and shared-seed tests pass. The evaluation manifest includes the joint diagnostic in the predetermined model/metric budget. This is an alternative diagnostic, not a claimed reproduction of a paper's omnibus calibration test.

`C:/Python312/python.exe -B -m unittest test_joint_calibration test_calibration_contract test_evaluation_harness test_cli_integrity -v`: 34 tests passed.

- Stage 4a: made categorical one-hot LSTM inputs the default, retained ordinal inputs as an explicit engine/evaluation/CLI ablation, and versioned the checkpoint input contract. Existing scalar checkpoints remain ordinal. CPU loading now respects the selected CPU device. `C:/Python312/python.exe -B -m unittest test_lstm_representation test_prediction_contract test_cli_integrity -v`: 16 tests passed, including exact continuation for both representations and legacy scalar loading. Recalibration comparators and their chronological partitions remain pending.

- Stage 4b: implemented optional temperature, MCLLO and normalization-aware isotonic comparators with chronological fit/calibration/test partitions. The manifest stores calibration provenance, parameters and optimization diagnostics; base and calibrated models see identical test outcomes. Formula, gradient, independent holdout, unseen-class, permutation, state and no-test-label-leakage tests pass. The fixed-block isotonic objective uses the convex reformulation below; its numerical optimality gap is reported rather than assuming solver convergence certifies a solution. Benchmark-wide comparisons remain pending.

Stage 4b verification: `C:/Python312/python.exe -B -m unittest test_recalibration test_temporal_evaluation test_cli_integrity -v`: 18 tests passed. The subsequently added recalibration CLI export test passed separately. A class-permutation test first exposed normalization roundoff moving a value across a fitted isotonic threshold; preserving already normalized inputs fixed the failure.

- Stage 5a: implemented optional categorical e-SR/e-CUSUM mixtures, a reference-conditional CTM variant using corrected linear bets, and PITMonitor with independent tie randomization, sequential ranks, histogram bets, full start-time mixture and post-alarm localization. Every detector supports idempotent replay and JSON state. `C:/Python312/python.exe -B -m unittest test_change_detection -v`: 7 tests passed, covering closed-form processes, 50 null streams with 500 looks each, discrete scores/atoms, shifts and calibration improvement. Full benchmark comparisons and CLI integration remain pending.

- Stage 6a: implemented an optional finite-basis Blackwell/ORCA-style recalibrator. Its epigraph optimizer enumerates all 37 adversarial outcomes; predictions and oracle bounds are fixed before the label. The payoff combines RBF-weighted categorical residuals with Brier/2 regret against one baseline. Positive oracle residuals enter the empirical bound, so failed halfspace steps are not presented as guarantees. Tests cover analytic gradients, adversarial outcomes under a one-iteration optimizer, forecast-before-label ordering, pending/completed-state replay and inconsistent saved residuals. Benchmark and evaluation integration remain pending.

`C:/Python312/python.exe -B -m unittest test_online_recalibration -v`: 4 tests passed, including gradients with a positive regret component.

- Stage 6b: exposed online recalibration through evaluation and CLI options. State persists across folds and is isolated by session/run; each forecast row stores its pre-outcome oracle bound. The manifest includes final diagnostics and replayable state. `C:/Python312/python.exe -B -m unittest test_online_recalibration test_cli_integrity -v`: 13 tests passed.

- Stage 7a: implemented the research CLI and eight reproducible scenario generators. Trial output retains exact configuration, seeds, data/source hashes, raw measurements, uncertainty, method assumptions, calibration/fit partitions and optimizer reports. The benchmark compares full-information policies, forecast/calibration metrics, all detectors and optional learned/online models on shared outcomes. KT is the single Dirichlet(1/2) categorical construction in Ryu and Wornell Section 3.1, using the existing verified multinomial engine. `C:/Python312/python.exe -B -m unittest test_research_benchmark -v`: 5 tests passed. Full campaigns and completion audit remain pending.

### Stage 7 campaign protocol

The core import regression now blocks Torch, scikit-learn, GUI and OCR modules in a fresh subprocess and executes the default research benchmark. `C:/Python312/python.exe -B -m unittest test_core_imports -v`: 2 tests passed. The core test profile includes the newly added statistical and research modules.

The following finite campaigns are fixed before inspecting their results. Each uses seed `20260912`, independent trial sequences and paired methods. Reference and calibration partitions contain 300 observations each. The core campaign uses 20 trials of 1,000 test observations in all eight scenarios. The recalibration campaign uses five trials of 300 test observations in all eight scenarios, with temperature, MCLLO, normalized isotonic and online recalibration (30 oracle iterations). The representation campaign uses five trials of 300 test observations in uniform, dependent and fixed-bias scenarios, comparing ExtraTrees and both LSTM representations on CPU with three epochs and hidden size 32. These small model campaigns assess implementation and effect direction; they do not support a definitive model ranking.

No seed is dropped for producing an alarm, loss or optimizer failure. Monte Carlo intervals are reported with their finite-sample precision. The separate mathematical regression campaigns already include 100 uniform streams with 1,000 observations for multinomial monitoring and 50 null streams with 500 observations for change detectors. Those checks are not pooled with overlapping benchmark seeds as independent evidence.

For uniformity monitors, the filtration contains past outcomes. In the overconfidence scenario, each current forecast contains information about a freshly drawn latent probability vector, but the next outcome remains uniform given past outcomes alone. The joint forecast diagnostic conditions on the current forecast and therefore has a different null. Reference/PIT monitors test score-distribution stability: stable miscalibration is a null case, while calibration improvement can be an alternative.

## Calibration-bound derivation


This is an alternative construction, not a reproduction of the population l2-ECE interval. The source concentration inequality is [Pinelis, Theorem 3.5](https://arxiv.org/abs/1208.2200v2). It bounds the maximum norm of a Hilbert-space martingale with increments of norm at most `L` through horizon `H` by `2 exp(-r^2 / (2 H L^2))`. The current arXiv text includes the paper's corrections; this construction uses the bounded-increment theorem, not its Bernstein variants.

For `R` configured runs on the same `n` outcomes, each prediction chooses one of `B` fixed bins before observing the result. Form a vector of residuals indexed by run and bin, scaled by `1/sqrt(R)`. Subtract its conditional mean to obtain a martingale difference of norm at most one. For `C` categorical channels, scale by `1/sqrt(R C)` instead; the shared one-hot outcome gives an increment bound `sqrt(2/C)`. This uses actual shared outcomes, not independence between runs or classes.

At observation count `n`, take `j = ceil(log2(n))` and horizon `H = 2^j`. Allocate `alpha_j = alpha / ((j+1)(j+2))`. These allocations sum to `alpha`, so a union bound over the maximal inequalities gives coverage at every time. Divide the resulting norm bound by `n`.

The desired statistic is the average over runs/channels of the sum of absolute bin residual means, with each bin total divided by `n`. By the reverse triangle inequality and Cauchy-Schwarz, its difference from the same statistic of conditional residual means is at most the norm bound times the square root of the average number of occupied bins. Unoccupied bins have zero observed and conditional residual sums because bin membership is predictable; using this observed support does not require selecting a new concentration event. Clip to the valid range `[0,1]`, or `[0,2/C]` for classwise categorical error.

The error budget is divided over the predetermined model family and confidence/classwise/top-set summaries. The construction is conditional on the configured training runs and does not support selecting configurations after inspecting results. Independent sessions must still be ordered so each prediction and inclusion decision precedes its outcome. No stationarity is needed for the stated conditional-residual target; translating it into a population calibration parameter would need additional assumptions.

## Normalized isotonic reformulation

For the positive, fixed-block version of [NA-FIR Equation 4](https://arxiv.org/abs/2512.09054), write each block value as `g_b = exp(z_b)`. The mean objective is `F(z) = mean_i(logsumexp_j(z_{b(i,j)}) - z_{b(i,y_i)})`. This exactly equals the original normalized multiclass negative log likelihood. It is convex in `z`, and the order constraints become linear: `z_b <= z_{b+1}`. A common additive shift cancels, so the implementation fixes the last value to zero and restricts all values to `[-30,0]` for a finite feasible domain. PAVA supplies the fixed blocks and initialization, not the final normalized solution.

For a feasible numerical solution `z`, convexity gives `F(z)-F* <= grad(F(z)) dot z - min_v grad(F(z)) dot v`. The monotone-box vertices have a prefix at -30 and the remaining values at zero, with the last value fixed. Thus the linear minimization uses the largest nonnegative prefix sum of the gradient. Tests compare this expression with an independent linear-program solve, check the gradient by finite differences, and verify equality with the paper's original objective. The bound concerns the restricted fixed-block problem and is computed in floating point. It does not certify population calibration, optimal block selection, or the unbounded positive-function problem. This is a derivation used for the implementation; a targeted literature search did not establish that it is new.

## Change-detector contracts

- E-SR uses `R_t=(R_(t-1)+1)L_t`, with a fixed mixture over 37 pocket-boost alternatives; e-CUSUM uses `C_t=max(1,C_(t-1))L_t`. Each likelihood factor has conditional null expectation one. The e-SR mixture minus time is a martingale, while e-CUSUM is dominated by its e-SR counterpart. Threshold `A` therefore controls average run length at `A`, as in [Shin et al.](https://arxiv.org/abs/2203.03532v4). It does not bound the probability of ever alarming. The finite-horizon test campaign uses the consequence `P(T<=H)<=H/A`.
- The reference monitor uses [CTM Equation 7](https://arxiv.org/abs/2602.13848v2), `b=1+eta*(p_hat-.5)-abs(eta)*epsilon`, with a fixed mixture of seven bounded bets instead of the paper's smoothed ONS update. A DKW event controls both CDF values and left limits. Randomizing within ECDF atoms gives `abs(p_hat-U)<=epsilon` for a true randomized PIT `U`. On that reference event, each factor is nonnegative and has conditional expectation at most one. The reported total error is the monitoring allocation plus the DKW failure allocation. This variant has power against mean ECDF shifts larger than reference uncertainty; it is not an omnibus distribution test.
- PITMonitor uses independent random tie keys, randomized sequential ranks, predictable histogram densities and weights `w_t=1/(t(t+1))`. Its reported e-process is the active recurrence plus unstarted mass `1/(t+1)`, as used in the proof of [PITMonitor Theorem 1](https://arxiv.org/abs/2603.13156v1). Validity requires IID PIT values and the rank filtration, not merely stationary marginal frequencies. Post-alarm localization compares rank segments with uniformity using the source's Dirichlet(1/2) prior. This estimate is neither the alarm time nor a confidence interval. A fixed categorical ordering and independent PIT randomization are explicit inputs.

## Online recalibration residual bound

The experimental payoff is `(phi(p) outer (one_hot(y)-p), Brier(p,y)/2-Brier(base,y)/2)`, with nonnegative normalized RBF features `phi`. The target is zero calibration residual and nonpositive regret. Projecting the cumulative payoff onto this cone leaves its full calibration component and the positive part of its regret component. The oracle minimizes the maximum inner product with that residual over all 37 possible outcomes, using normalized past sums.

Each payoff has squared norm at most 3. If the achieved pre-outcome halfspace upper bound on round `t` is `u_t`, the distance of the average payoff from the cone is at most `sqrt(3/n + 2*sum_t((t-1)*max(u_t,0))/n^2)`. The implementation reports this bound and the actual distance. This deterministic accounting remains valid when the optimizer fails to make a step nonpositive; it does not imply convergence if positive residuals persist. It controls the stated finite feature basis, not all measurable calibration conditions. [Marx et al.](https://arxiv.org/abs/2409.19157) supplies the approachability and approximate-oracle framework; the categorical basis and solver here are explicit adaptations.

## Completion audit

| Recommendation | Implementation and verification evidence |
|---|---|
| Correct ECE aggregation, ties and dependence between seed replicas | `temporal_statistics.py`, `evaluation_harness.py`; regressions in `test_calibration_contract.py` and `test_temporal_bootstrap.py` reproduce cancellation, bin-boundary and replica failures. |
| Replace invalid calibration intervals and distinguish joint calibration | `calibration.py` states the alternative conditional-residual target and family budget; `joint_calibration.py` separates a vector-cell diagnostic from its stronger sequential null. Tests cover calibrated/dependent streams, the joint counterexample and unavailable cohorts. |
| Add durable sequential evidence and simultaneous probability bounds | `sequential_inference.py`, analyzer, `monitor` CLI and HH checkpoints; closed-form likelihood/projection tests, null simulations, replay/deduplication, restart spending and changed-history rejection. The CPU HH training/resume smoke retained all 180 observations. |
| Evaluate all actions and correct profitability/OU claims | `expected_value.py`, shared settlement and harness; tests cover all 47 actions, uniform expectations, PASS, bankroll, the `1/36` threshold and the analytical IID rolling-window overlap null. |
| Compare categorical and ordinal LSTM inputs | `lstm_predictor.py`, engine and CLI; both representations preserve train/save/resume behavior and explicit CPU selection. Old scalar checkpoints remain ordinal. The representation campaign measures held-out results. |
| Add chronological recalibrators | `recalibration.py` and harness partitions; temperature, regularized MCLLO and fixed-block normalized isotonic expose fitted states and solver diagnostics. Tests independently verify gradients, the normalized objective, the convex gap bound, unseen classes and exclusion of test labels. |
| Compare optional change detectors with correct guarantees | `change_detection.py` and research CLI; e-SR/e-CUSUM ARL differs from the CTM/PIT probability-of-alarm contracts. Tests cover atoms, full start-time mixture, replay, shifts and null simulations. The benchmark retains violated-assumption cases and censored delays. |
| Evaluate bounded-score online recalibration | `online_recalibration.py`, harness and CLI; a finite RBF basis and all-outcome oracle residual accounting replace any unconditional full-calibration claim. Tests verify gradients, adversarial outcomes, chronology and state. |
| Make comparisons reproducible and keep optional dependencies optional | `research_benchmark.py`, research CLI and tests retain seeds, hashes, partitions, raw measurements, paired metrics and uncertainty. The core profile includes the new statistical modules; an isolated subprocess blocks optional ML/GUI/OCR imports while actually running the default benchmark. |
| Reassess prior RL and OPE choices | Double DQN action selection/evaluation, masks, termination and continuation match their tested contracts. Passive full-information replay needs no propensity weighting under its stated action-independence assumptions. |
| Reconcile source attributions and exclusions | README and `arxiv_paper_memory.md` distinguish exact formulas from adaptations. Physical prediction, unsupported TSFM additions and unnecessary IPS/DR remain excluded for the reasons in the original audit. No experiment is automatically promoted. |

The measured campaign results and full-suite outcome are recorded below. Real-data validation cannot establish superiority at this point: a read-only audit found zero sessions, spins and predictions. Capture configuration SHA-256 was `8292b5b495d2dd5b89557ec7586c75065c11f589bc29b32cdafd6e79041a0c72`; none of the research changes modifies that configuration or the user database.

### Core campaign results

Command: `C:/Python312/python.exe -B roulette_cli.py research --seed 20260912 --trials 20 --observations 1000 --output research/results/core-20260912.json`.

The command exited successfully after 1,238.6 seconds. [The JSON artifact](research/results/core-20260912.json) contains 160 trials and 160,000 test observations, plus their separate reference/calibration prefixes. All 14 recorded runtime source hashes match the checkout. No trial was removed.

| Measurement | Result |
|---|---|
| Uniform-null alarms | 0/20 for multinomial, KT, e-SR, e-CUSUM, reference CTM and joint diagnostic; 1/20 for PITMonitor. Exact 95% intervals are [0, 0.1684] and [0.0013, 0.2487], respectively. These finite samples are consistent with the contracts but cannot establish a 5% bound by themselves. |
| Fixed-bias detection | Multinomial, KT, e-SR, e-CUSUM and joint diagnostic each alarmed in 20/20 trials, interval [0.8316, 1]. PIT/reference monitor stability, so fixed bias already present in the reference is a null case for them. |
| Abrupt-change detection after the change | Multinomial 20/20, KT 5/20, e-SR/e-CUSUM 20/20 and PITMonitor 5/20. PITMonitor also had two pre-change alarms, reported separately. Multinomial and e-SR restricted mean delays were 223.4 and 72.55 observations. |
| Gradual-change detection after the change | Multinomial 9/20, KT 0/20, e-SR/e-CUSUM 20/20 and PITMonitor 0/20. Multinomial and KT each had one pre-change alarm. Nondetections remain in restricted-delay estimates. |
| Reference CTM power | 0/20 alarms for abrupt, gradual and calibration-improvement alternatives with the fixed 300-observation reference and chosen scalar PIT score. The finite-reference correction and limited betting family are conservative; this experiment gives no basis to promote the variant. |
| Calibration improvement | PITMonitor detected 19/20 changes afterward and had one early alarm. An alarm therefore cannot be labeled deterioration without a separate direction/quality analysis. |
| Probability confidence regions | Both methods covered the fixed true vector at every observation in 20/20 trials of each applicable scenario: uniform, fixed bias and calibration improvement. Mean final pocket width under uniformity was 0.0722 for the mixture and 0.1445 for KT. No fixed-probability coverage claim is made for dependent or changing conditional laws. |
| Calibration residual bounds | Confidence, classwise and top-five conditional-residual targets were covered at all four checked counts in every trial of each scenario. This is a finite simulation check of the separately proved bound, not a population ECE estimate. |
| Uniform-wheel policies | Frequency-based EV bet on every observation: mean realized profit -42.4 units, approximate 95% interval [-146.61, 61.81], and known conditional expected profit -27.027. The confidence-bound policy chose PASS throughout and earned zero. All PASS counterfactual totals were zero. |

The strong bias scenario deliberately sets one pocket's probability to 0.15. Large positive simulated profits there illustrate the decision/settlement contract and must not be extrapolated to a real wheel. The mixture's improvement over KT and the e-detectors' change sensitivity are specific to these alternatives, priors and horizons; the trial intervals are not simultaneous over all comparisons. Reference/PIT results under stationary dependence remain labeled as violated assumptions.

### Recalibration campaign results

Command: `C:/Python312/python.exe -B roulette_cli.py research --seed 20260912 --trials 5 --observations 300 --recalibrators temperature mcllo normalized_isotonic --include-online --online-iterations 30 --output research/results/recalibration-20260912.json`.

The command completed 40 trials in 470.2 seconds. [The artifact](research/results/recalibration-20260912.json) retains all fitted parameters and measurements; all recorded runtime hashes match the checkout. All 120 offline fits were available and their solvers reported success. Numerical success did not automatically certify optimality: four of 40 isotonic fits exceeded the declared `1e-6` gap threshold, with maximum gap `2.9196e-5`; their `optimization_certified` fields remain false.

In the overconfidence scenario, paired held-out log-loss differences versus the same uncalibrated forecast were:

| Method | Mean difference | Approximate 95% interval |
|---|---:|---:|
| Temperature | -0.99083 | [-1.18421, -0.79746] |
| MCLLO | -0.85782 | [-1.04918, -0.66646] |
| Normalized isotonic | -0.84643 | [-0.98689, -0.70598] |
| Online finite-basis variant | +11.85566 | [10.00195, 13.70936] |

Temperature had the largest mean reduction in this deliberately temperature-distorted scenario; that is not a general ranking. On uniform data, MCLLO worsened log loss by 0.05218, interval [0.01158, 0.09278], demonstrating calibration-set overfitting. On fixed bias it improved the uniform base, while temperature and isotonic could not separate identical class scores.

The online variant worsened mean log loss in all eight scenarios, including +1.01660 [0.95361, 1.07959] under uniformity. Of 12,000 pre-outcome oracle steps, 11,790 had a nonpositive computed bound; the maximum positive residual was 0.012254. Every final cone distance stayed within its reported bound, with maximum observed distance 0.026122. This does not rescue predictive quality: a finite-basis, finite-horizon bound involving Brier/2 does not control log loss, especially near zero assigned probabilities. The unfavorable result is retained and the variant stays opt-in; it is not recommended for deployment based on this campaign.

### Representation campaign results

Command: `C:/Python312/python.exe -B roulette_cli.py research --seed 20260912 --trials 5 --observations 300 --scenarios uniform dependence fixed_bias --learned-models extra_trees lstm_one_hot lstm_ordinal --device cpu --neural-epochs 3 --output research/results/representation-20260912.json`.

The command completed 15 trials in 73.4 seconds. All 45 model fits were available; runtime hashes match the checkout. [The artifact](research/results/representation-20260912.json) records chronological partitions, scores, exposure and model fit reports. Paired differences below use the five shared trial outcomes in each scenario, with Student-t intervals:

| Scenario | One-hot minus ordinal log loss | Approximate 95% interval |
|---|---:|---:|
| Uniform | -0.00215 | [-0.00963, 0.00534] |
| Stationary dependence | +0.00613 | [-0.00680, 0.01906] |
| Fixed bias | +0.00080 | [-0.01258, 0.01418] |

Every interval contains zero. One-hot remains the explicit categorical input contract, but this short campaign does not establish superiority over ordinal inputs. The fixed-bias frequency baseline had mean log loss 3.50284, compared with 3.50644/3.50563 for the LSTMs and 3.71581 for ExtraTrees. Under uniformity, the fair forecast remained preferable to these fitted alternatives. Neither larger neural models nor foundation-model integration is justified by these results.

The three campaigns are not pooled as independent replications: some scenarios reuse seeded prefixes. All model choices and budgets were fixed before inspecting outcomes.

### Final verification and disposition

`C:/Python312/python.exe -B scripts/run_checks.py --profile full` completed successfully: **162 tests passed in 68.103 seconds**, without skipped tests. This includes formula/gradient checks, null simulations, temporal partitions, settlement, model train/save/resume, CLI exports, persisted monitors and the isolated optional-dependency import regression. The three actual research CLI campaigns also exited successfully. Source hashes matched for every campaign; datasets and capture settings were preserved.

All eight work packages are complete. Scientific limitations are results of the audit, not claims of successful external validation: the database has no eligible real observations; the fixed-reference variant lacked power in the chosen scenarios; the online variant harmed log loss; four isotonic fits missed the numerical certificate threshold; and the representation comparison did not establish superiority. These findings remain in the exported artifacts and the experimental methods remain opt-in. No additional implementation is justified automatically by this campaign.

Delivery consists of incremental local commits, the implementation, tests, source notes and all three result artifacts. Publication is outside this goal's authorization.
