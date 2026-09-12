from collections import defaultdict
from math import ceil

import numpy as np


def adaptive_bins(confidences, outcomes, max_bins=10, min_bin_size=20):
    confidence = np.asarray(confidences, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if len(confidence) == 0:
        return []
    order = np.argsort(confidence, kind='stable')
    confidence, observed = confidence[order], observed[order]
    target_bins = min(max_bins, max(1, len(confidence) // min_bin_size))
    boundaries = [0]
    for target in np.linspace(0, len(confidence), target_bins + 1)[1:-1]:
        split = int(np.searchsorted(confidence, confidence[min(int(target), len(confidence) - 1)], side='right'))
        if split - boundaries[-1] >= min_bin_size and len(confidence) - split >= min_bin_size:
            boundaries.append(split)
    boundaries.append(len(confidence))
    return [(float(confidence[left]), float(confidence[right - 1]), right - left,
             float(confidence[left:right].mean()), float(observed[left:right].mean()))
            for left, right in zip(boundaries, boundaries[1:])]


def calibration_error(confidences, outcomes, max_bins=10):
    bins = adaptive_bins(confidences, outcomes, max_bins)
    count = sum(item[2] for item in bins)
    return sum(size * abs(accuracy - confidence) for _, _, size, confidence, accuracy in bins) / count if count else None


def block_bootstrap_indices(rows, resamples, seed, block_length=None):
    groups = defaultdict(lambda: defaultdict(dict))
    session_indices = defaultdict(set)
    for position, row in enumerate(rows):
        groups[row.run_id][row.session_id][row.index] = position
        session_indices[row.session_id].add(row.index)
    structures = {}
    for session_id, indices in sorted(session_indices.items()):
        ordered = sorted(indices)
        chunks = []
        current = []
        for index in ordered:
            if current and index != current[-1] + 1:
                chunks.append(current)
                current = []
            current.append(index)
        if current:
            chunks.append(current)
        size = len(ordered)
        length = min(size, block_length or ceil(size ** (1 / 3)))
        weights = np.asarray([len(chunk) for chunk in chunks], dtype=float) / size
        structures[session_id] = size, length, chunks, weights
    run_ids = sorted(groups)
    temporal_rng = np.random.default_rng(seed)
    run_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for _ in range(resamples):
        sampled_sessions = {}
        for session_id, (size, length, chunks, weights) in structures.items():
            sample = []
            while len(sample) < size:
                chunk = chunks[int(temporal_rng.choice(len(chunks), p=weights))]
                start = int(temporal_rng.integers(len(chunk)))
                sample.extend(chunk[(start + offset) % len(chunk)] for offset in range(min(length, len(chunk))))
            sampled_sessions[session_id] = sample[:size]
        sampled = []
        for run_id in run_rng.choice(run_ids, size=len(run_ids), replace=True):
            for session_id, sample in sampled_sessions.items():
                positions = groups[run_id].get(session_id, {})
                sampled.extend(positions[index] for index in sample if index in positions)
        yield np.asarray(sampled, dtype=int)
