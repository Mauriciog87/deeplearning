import base64
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API_ROOT = "https://api.github.com"
ORG = "google-research"
OUTPUT_PATH = Path("google_research_repo_memory.txt")
PER_PAGE = 100
MAX_PAGES = 30
DEEP_DIVE_LIMIT = 30
CURATED_SECTION = """Curated second-pass findings from parallel agent review:
The live GitHub inventory had 348 public repositories on 2026-06-16. With per_page=100, only pages 1-4 had content. Pages 5-12 were checked conceptually from pagination and are empty unless the organization adds more repositories later.

Use this curated section before the automatic ranked inventory below. The automatic scores are useful for search, but they overvalue generic RL, sequence, or benchmark keywords that do not necessarily transfer to roulette.

Highest-value candidates for rl-roulette:

1. google-research/rliable
URL: https://github.com/google-research/rliable
Curated score: 9.2
Category: evaluation and statistical confidence
Why it matters: The local project already has walk-forward evaluation, top-k, ROI, log-loss, Brier, and calibration metrics. rliable is directly useful for robust aggregate metrics, bootstrap confidence intervals, interquartile mean, and probability of improvement across folds/runs.
Integration path: Start by copying the evaluation ideas, not the whole package. Add confidence intervals and paired model comparison around existing ModelEvaluationSummary outputs.
Acceptance test: The report should show CIs for top-k hit rate, ROI, Brier/log-loss, and probability that a candidate beats fair or rolling_frequency across folds.
Risk: Evaluation rigor may reveal that no predictor has signal. That is a good outcome and should not be hidden.

2. google-research/robustness_metrics
URL: https://github.com/google-research/robustness_metrics
Curated score: 8.8
Category: calibration, uncertainty, OOD diagnostics
Why it matters: The local engine emits probability distributions over 37 numbers. Calibration and uncertainty checks are more relevant than raw hit rate.
Integration path: Add reliability diagrams, ECE/MCE variants, negative log-likelihood summaries, confidence bucket analysis, and maybe OOD/stability checks by session or time window.
Acceptance test: Existing predictors should have per-model calibration diagnostics and confidence buckets that can expose overconfident false positives.
Risk: Dependencies may be heavier than needed. Prefer lightweight local implementations of the metrics first.

3. google-research/rl-reliability-metrics
URL: https://github.com/google-research/rl-reliability-metrics
Curated score: 8.5
Category: RL evaluation reliability
Why it matters: The repo has DQN and strategy evaluation. RL results are noisy and easy to overfit to a backtest.
Integration path: Borrow reliability concepts for repeated training/evaluation runs, seed variance, stability, drawdown, and bootstrap comparisons.
Acceptance test: DQN or future policy changes should be accepted only if they improve probabilistic and profit metrics with uncertainty bounds across repeated runs.
Risk: Some metrics may be overkill until the project has enough independent sessions.

4. google-research/spade_anomaly_detection
URL: https://github.com/google-research/spade_anomaly_detection
Curated score: 8.2
Category: anomaly detection
Why it matters: For roulette, anomaly/bias detection is more defensible than direct prediction. This aligns with BiasAwarePredictor and future heatmaps.
Integration path: Treat as inspiration for a dedicated anomaly track: detect abnormal windows by number, sector, dozen, column, color, parity, entropy, and chi-square residuals.
Acceptance test: On synthetic biased wheels, the anomaly module should flag injected drift earlier than simple chi-square without raising many false alarms on shuffled fair data.
Risk: Avoid training on future data or tuning anomaly thresholds on the test window.

5. google-research/timesfm
URL: https://github.com/google-research/timesfm
Curated score: 8.0
Category: future time-series forecasting
Why it matters: Useful once the SQLite DB has many well-labeled spins and metadata. It should forecast derived temporal features, not raw roulette numbers as continuous values.
Integration path: Optional lazy-loaded experimental predictor over rolling features: number frequencies, sector heat, entropy, deviation from fair probability, color/parity/dozen/column rates, and metadata covariates if available.
Acceptance test: Must beat fair, rolling_frequency, last_n, hot, random, bias, and consensus in walk-forward log-loss, Brier, calibration, and top-k before enabling anywhere normal.
Risk: Heavy dependency and checkpoint download. The most likely useful output is analysis/heatmaps/anomaly priors, not next-number prediction.

6. google-research/soft-dtw-divergences
URL: https://github.com/google-research/soft-dtw-divergences
Curated score: 7.8
Category: drift and window similarity
Why it matters: Comparing rolling windows of derived distributions can identify regime changes or cluster similar sessions.
Integration path: Build features per window and compare windows/sessions with soft-DTW-like distances. Use results for drift dashboards and anomaly context.
Acceptance test: Synthetic shifted windows should separate from shuffled fair windows more reliably than simple Euclidean distance on raw counts.
Risk: Continuous time-series assumptions need care; roulette numbers should be encoded as categorical distributions or wheel-sector features.

7. google-research/dice_rl and google-research/deep_ope
URLs: https://github.com/google-research/dice_rl and https://github.com/google-research/deep_ope
Curated score: 7.6
Category: off-policy evaluation
Why it matters: Backtesting betting policies can be biased if policies are selected retrospectively. OPE concepts help evaluate policies without fooling ourselves.
Integration path: Use as conceptual input for policy evaluation and train/test protocol design. Full import is probably unnecessary.
Acceptance test: New betting policies should be evaluated with strict walk-forward, fixed policy selection rules, and null/permutation baselines.
Risk: Real casino outcomes are not generated by our logging policy in the same way recommender/RL datasets are, so apply concepts cautiously.

8. google-research/gpax, google-research/hyperbo, google-research/optformer
URLs: https://github.com/google-research/gpax, https://github.com/google-research/hyperbo, https://github.com/google-research/optformer
Curated score: 7.2
Category: uncertainty and hyperparameter optimization
Why it matters: These can help tune LSTM/DQN/ExtraTrees/BiasAware and quantify uncertainty, but they are not direct roulette predictors.
Integration path: Defer until the evaluation harness has confidence intervals. Then use Bayesian optimization ideas for hyperparameters and ensemble weights.
Acceptance test: HPO must be nested inside walk-forward or use held-out validation to avoid picking lucky parameters.
Risk: JAX stacks and black-box HPO can increase complexity and overfitting.

9. google-research/weatherbenchX and google-research/weatherbench2
URLs: https://github.com/google-research/weatherbenchX and https://github.com/google-research/weatherbench2
Curated score: 7.0
Category: forecast evaluation architecture
Why it matters: Useful as architecture inspiration for reproducible forecast evaluation, aggregation, and metrics, especially if the project grows into many datasets/sessions.
Integration path: Borrow structure for result aggregation and benchmark reports. Do not import weather-specific code.
Acceptance test: The roulette evaluation harness should support repeatable folds, model ordering, summary tables, and persisted per-row predictions.
Risk: Domain mismatch is high; keep this as design inspiration.

10. google-research/unique-randomizer
URL: https://github.com/google-research/unique-randomizer
Curated score: 6.8
Category: discrete candidate sampling
Why it matters: Could help sample unique top-k candidates from model distributions without replacement.
Integration path: Only consider if the project needs stochastic candidate generation from calibrated probabilities. A small local sampler may be enough.
Acceptance test: Sampling should preserve model probabilities while avoiding duplicate candidates and should not improve metrics by accidental leakage.
Risk: Low core value unless top-k candidate diversity becomes a real problem.

Useful but lower-priority references:
- google-research/flood-forecasting: sequence forecasting and walk-forward patterns; good inspiration, not direct roulette code.
- google-research/zapbench: forecasting benchmark patterns; useful for notebooks/evaluation design.
- google-research/discs: discrete sampling and MCMC concepts for 0-36 distributions.
- google-research/deep_representation_one_class: one-class anomaly detection ideas.
- google-research/fast-soft-sort: differentiable ranking/top-k ideas if training custom neural ranking losses.
- google-research/long-range-arena, perceiver-ar, meliad, t5x: long-context sequence modeling ideas, but likely too heavy for this repo now.

Recommended implementation order:
1. Strengthen evaluation first: rliable-style CIs, robustness_metrics-style calibration, and RL reliability summaries.
2. Expand heatmap UX only when needed: inline image previews, richer anomaly tests, and rolling drift comparisons.
3. Add better experiment discipline: fixed walk-forward protocols, permutation/null tests, and no retrospective strategy selection.
4. Only then prototype optional TimesFM over derived features when the DB has enough labeled spins and metadata.
5. Keep all large external research stacks optional and lazy-loaded. Prefer reimplementing small metric ideas locally over adding heavy dependencies.

Implementation status after 2026-06-16 pass:
Implemented now:
- Added bootstrap confidence intervals to the local walk-forward evaluation summary.
- Added calibration bins per model, complementing existing ECE.
- Added paired model-vs-fair comparisons with improvement deltas and probability of improvement.
- Added CLI PNG heatmaps for table residuals, wheel sectors, rolling number residuals, and rolling category residuals.
- Added anomaly summaries over numbers, table categories, and wheel sectors with configurable z-score thresholds.
- Added session drift heatmaps over encoded distributions and a basic GUI action that generates per-session heatmap reports.
- Kept the implementation dependency-free, local, and compatible with existing evaluate_walk_forward callers.

Deep dive decisions:
- rliable: implement ideas locally now. Confidence intervals and paired comparisons are the correct first step.
- robustness_metrics: implement lightweight calibration buckets now. Reliability diagrams and OOD/session stability can come later.
- rl-reliability-metrics: partially implemented now through repeated-row summaries and uncertainty bounds. Repeated training seed studies remain future work.
- spade_anomaly_detection: lightweight anomaly alerts over derived categorical/sector features are implemented. Future work is richer entropy, chi-square, and metadata-aware alerting.
- timesfm: future optional experiment only after more labeled data. Use derived features and covariates, not raw 0-36 sequences as continuous values.
- soft-dtw-divergences: first session drift similarity over encoded distributions is implemented. Future work is rolling-window soft-DTW-style comparisons.
- dice_rl and deep_ope: future evaluation discipline if policy selection gets more complex; do not import now.
- gpax, hyperbo, optformer: future HPO/uncertainty ideas after the current harness is stable.
- weatherbenchX/weatherbench2: design inspiration only for benchmark organization.
- unique-randomizer: future local sampling helper only if stochastic top-k candidate diversity becomes a real need."""


POSITIVE_KEYWORDS = {
    "time series": 4.5,
    "forecast": 4.5,
    "forecasting": 4.5,
    "temporal": 3.5,
    "sequence": 3.0,
    "sequential": 3.0,
    "anomaly": 4.0,
    "detect": 1.5,
    "detection": 1.5,
    "heatmap": 3.0,
    "visualization": 2.0,
    "probabilistic": 4.0,
    "uncertainty": 3.5,
    "calibration": 4.0,
    "statistics": 3.0,
    "statistical": 3.0,
    "causal": 2.0,
    "reinforcement": 4.0,
    "rl": 3.0,
    "bandit": 4.0,
    "monte carlo": 3.0,
    "bayesian": 3.0,
    "tabular": 3.0,
    "boost": 2.5,
    "gbm": 3.0,
    "random forest": 2.5,
    "classification": 2.0,
    "clustering": 1.5,
    "embedding": 1.0,
    "graph": 1.0,
    "simulation": 2.0,
    "synthetic": 1.5,
    "evaluation": 2.0,
    "benchmark": 2.0,
    "metrics": 2.0,
}

NEGATIVE_KEYWORDS = {
    "language model": 2.0,
    "llm": 2.0,
    "nlp": 2.0,
    "text": 1.5,
    "image": 2.0,
    "vision": 2.0,
    "video": 2.0,
    "robot": 1.5,
    "robotics": 1.5,
    "audio": 1.5,
    "speech": 1.5,
    "music": 1.5,
    "protein": 1.5,
    "genomics": 1.5,
    "quantum": 1.5,
}


def request_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "rl-roulette-research-scan",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.raw",
            "User-Agent": "rl-roulette-research-scan",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_repositories():
    repos = []
    page_counts = []
    for page in range(1, MAX_PAGES + 1):
        url = (
            f"{API_ROOT}/orgs/{ORG}/repos"
            f"?per_page={PER_PAGE}&page={page}&sort=full_name&direction=asc"
        )
        data = request_json(url)
        if not data:
            break
        repos.extend(data)
        page_counts.append((page, len(data)))
        time.sleep(0.25)
    return repos, page_counts


def text_for(repo):
    pieces = [
        repo.get("name") or "",
        repo.get("description") or "",
        " ".join(repo.get("topics") or []),
        repo.get("language") or "",
    ]
    return " ".join(pieces).lower().replace("_", " ").replace("-", " ")


def matched_keywords(text, weights):
    return [(keyword, weight) for keyword, weight in weights.items() if keyword in text]


def score_repo(repo):
    text = text_for(repo)
    positive = matched_keywords(text, POSITIVE_KEYWORDS)
    negative = matched_keywords(text, NEGATIVE_KEYWORDS)
    score = 1.0 + sum(weight for _, weight in positive) - sum(weight for _, weight in negative)
    language = (repo.get("language") or "").lower()
    if language == "python":
        score += 1.0
    elif language in {"jupyter notebook", "r"}:
        score += 0.5
    elif language in {"html", "css", "javascript", "typescript"}:
        score -= 0.5
    if repo.get("archived"):
        score -= 2.0
    stars = int(repo.get("stargazers_count") or 0)
    score += min(1.2, math.log10(stars + 1) / 2.0)
    updated = repo.get("updated_at") or ""
    if updated >= "2025-01-01":
        score += 0.6
    elif updated >= "2023-01-01":
        score += 0.3
    score = max(0.0, min(10.0, score))
    return round(score, 1), positive, negative


def fetch_readme(repo):
    branch = repo.get("default_branch") or "master"
    candidates = ["README.md", "README", "readme.md", "Readme.md"]
    for name in candidates:
        url = f"{API_ROOT}/repos/{repo['full_name']}/contents/{name}?ref={branch}"
        try:
            data = request_json(url)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            return ""
        except urllib.error.URLError:
            return ""
        if isinstance(data, dict) and data.get("content"):
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except (ValueError, TypeError):
                return ""
    return ""


def compact_readme_summary(readme):
    lines = []
    for raw in readme.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[!") or line.startswith("<!--"):
            continue
        if len(line) > 260:
            line = line[:257] + "..."
        lines.append(line)
        if len(lines) >= 8:
            break
    return " ".join(lines)


def classify_fit(repo, score, positive):
    words = {keyword for keyword, _ in positive}
    if score >= 8.0:
        return "strong"
    if score >= 6.5:
        return "promising"
    if score >= 5.0:
        return "watch"
    if words:
        return "weak"
    return "low"


def integration_note(repo, positive):
    words = {keyword for keyword, _ in positive}
    if {"time series", "forecast", "forecasting", "temporal"} & words:
        return "Candidate for derived temporal features, rolling heatmaps, anomaly intervals, or forecast-based priors."
    if {"reinforcement", "rl", "bandit"} & words:
        return "Candidate for betting policy selection, exploration control, or strategy meta-learning."
    if {"probabilistic", "uncertainty", "calibration", "bayesian"} & words:
        return "Candidate for probability calibration, uncertainty-aware predictions, or confidence diagnostics."
    if {"statistics", "statistical", "evaluation", "benchmark", "metrics"} & words:
        return "Candidate for evaluation harness, statistical tests, or model comparison rigor."
    if {"heatmap", "visualization"} & words:
        return "Candidate for visual diagnostics over numbers, sectors, sessions, or time windows."
    if {"tabular", "boost", "gbm", "classification"} & words:
        return "Candidate for supervised tabular features if enough labeled spins and session metadata exist."
    return "No direct path identified from metadata."


def line_for_repo(item):
    repo = item["repo"]
    positive = ", ".join(keyword for keyword, _ in item["positive"][:6]) or "none"
    negative = ", ".join(keyword for keyword, _ in item["negative"][:4]) or "none"
    description = (repo.get("description") or "").replace("\n", " ").strip()
    if len(description) > 180:
        description = description[:177] + "..."
    return (
        f"{item['score']:>4.1f} | {item['fit']:<9} | {repo['full_name']} | "
        f"lang={repo.get('language') or 'unknown'} | stars={repo.get('stargazers_count') or 0} | "
        f"updated={repo.get('updated_at', '')[:10]} | +[{positive}] | -[{negative}] | {description}"
    )


def build_report(repos, page_counts):
    scored = []
    for repo in repos:
        score, positive, negative = score_repo(repo)
        scored.append(
            {
                "repo": repo,
                "score": score,
                "positive": positive,
                "negative": negative,
                "fit": classify_fit(repo, score, positive),
                "integration": integration_note(repo, positive),
            }
        )
    scored.sort(
        key=lambda item: (
            item["score"],
            int(item["repo"].get("stargazers_count") or 0),
            item["repo"]["full_name"],
        ),
        reverse=True,
    )
    deep_items = scored[:DEEP_DIVE_LIMIT]
    for item in deep_items:
        item["readme"] = compact_readme_summary(fetch_readme(item["repo"]))
        time.sleep(0.25)
    lines = []
    lines.append("Google Research repository memory for rl-roulette")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Organization: https://github.com/{ORG}")
    lines.append("")
    lines.append("Local fit context:")
    lines.append(
        "The local project predicts European roulette outcomes from a 0-36 history using LSTM, DQN, ExtraTrees, BiasAware, consensus, and simple baselines. Useful imports should improve derived time-series analysis, anomaly detection, heatmaps, probability calibration, RL strategy selection, or walk-forward evaluation."
    )
    lines.append("")
    lines.append("Scoring:")
    lines.append("0-3 low relevance, 4-4.9 weak, 5-6.4 watch, 6.5-7.9 promising, 8-10 strong.")
    lines.append("Scores are first-pass metadata/readme signals, not proof of ROI.")
    lines.append("")
    lines.append(CURATED_SECTION)
    lines.append("")
    lines.append("Inventory summary:")
    lines.append(f"Repositories scanned: {len(repos)}")
    lines.append("Page counts: " + ", ".join(f"p{page}={count}" for page, count in page_counts))
    lines.append("")
    lines.append("Second-pass shortlist:")
    for index, item in enumerate(deep_items, 1):
        repo = item["repo"]
        lines.append("")
        lines.append(f"{index}. {repo['full_name']} | score={item['score']} | fit={item['fit']}")
        lines.append(f"URL: {repo['html_url']}")
        lines.append(f"Description: {repo.get('description') or 'No description'}")
        lines.append(f"Language: {repo.get('language') or 'unknown'} | stars={repo.get('stargazers_count') or 0} | updated={repo.get('updated_at', '')[:10]} | archived={repo.get('archived')}")
        lines.append(f"Positive signals: {', '.join(keyword for keyword, _ in item['positive']) or 'none'}")
        lines.append(f"Negative signals: {', '.join(keyword for keyword, _ in item['negative']) or 'none'}")
        lines.append(f"Potential integration: {item['integration']}")
        lines.append("Risk: Treat as experimental until it beats fair, rolling_frequency, last_n, hot, random, bias, and consensus in walk-forward log-loss, Brier, calibration, and top-k.")
        lines.append(f"README signal: {item.get('readme') or 'README not found or not readable in first pass.'}")
    lines.append("")
    lines.append("Full first-pass ranked inventory:")
    for item in scored:
        lines.append(line_for_repo(item))
    lines.append("")
    lines.append("Next manual review targets:")
    lines.append("1. Re-score shortlist with direct README/API examples and dependency checks.")
    lines.append("2. Separate candidates into analysis-only, predictor-plugin, evaluation, and visualization tracks.")
    lines.append("3. Add no dependency to the core app until a local walk-forward experiment proves improvement.")
    return "\n".join(lines) + "\n"


def main():
    repos, page_counts = fetch_repositories()
    report = build_report(repos, page_counts)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} with {len(repos)} repositories")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"scan failed: {error}", file=sys.stderr)
        raise
