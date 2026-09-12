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
