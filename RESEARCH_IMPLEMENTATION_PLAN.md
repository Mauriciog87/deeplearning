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
