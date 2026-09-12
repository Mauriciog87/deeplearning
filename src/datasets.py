from dataclasses import dataclass
import hashlib
import json

from .settlement import validate_number


@dataclass(frozen=True)
class EvaluationSession:
    session_id: str
    numbers: tuple[int, ...]
    spin_ids: tuple[str, ...] = ()
    source: str = 'provided'

    def __post_init__(self):
        object.__setattr__(self, 'session_id', str(self.session_id))
        object.__setattr__(self, 'numbers', tuple(validate_number(n) for n in self.numbers))
        ids = tuple(str(value) for value in self.spin_ids) if self.spin_ids else tuple(str(i) for i in range(len(self.numbers)))
        if len(ids) != len(self.numbers) or len(set(ids)) != len(ids):
            raise ValueError('Every observation must have a unique spin ID within its session')
        object.__setattr__(self, 'spin_ids', ids)


def as_sessions(data):
    values = list(data)
    if not values:
        return []
    sessions = values if isinstance(values[0], EvaluationSession) else [EvaluationSession('provided', tuple(values))]
    if any(not isinstance(session, EvaluationSession) for session in sessions):
        raise ValueError('Do not mix sessions and bare roulette numbers')
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError('Session identifiers must be unique')
    return sessions


def dataset_manifest(sessions):
    records = [{'session_id': session.session_id, 'source': session.source,
                'spin_ids': list(session.spin_ids), 'numbers': list(session.numbers)} for session in sessions]
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'sessions': records}


def training_partition(data, history_size=20):
    sessions = as_sessions(data)
    segments = []
    partitions = []
    for session in sessions:
        boundary = int(len(session.numbers) * .8)
        if boundary <= history_size or boundary >= len(session.numbers):
            raise ValueError(f'Session {session.session_id} is too short for an 80/20 chronological split')
        segments.append(list(session.numbers[:boundary]))
        partitions.append({'session_id': session.session_id, 'boundary': boundary,
                           'train_spin_ids': list(session.spin_ids[:boundary]),
                           'test_spin_ids': list(session.spin_ids[boundary:])})
    if not segments:
        raise ValueError('No real sessions are available')
    return segments, {'dataset': dataset_manifest(sessions), 'partitions': partitions}


def held_out_segments(data, partition, history_size=20):
    sessions = as_sessions(data)
    if dataset_manifest(sessions)['sha256'] != partition['dataset']['sha256']:
        raise ValueError('Dataset does not match the checkpoint partition')
    by_id = {session.session_id: session for session in sessions}
    result = []
    for record in partition['partitions']:
        if set(record['train_spin_ids']) & set(record['test_spin_ids']):
            raise ValueError('Training and test observations overlap')
        session = by_id[record['session_id']]
        boundary = record['boundary']
        if (not history_size <= boundary < len(session.numbers)
                or list(session.spin_ids[:boundary]) != record['train_spin_ids']
                or list(session.spin_ids[boundary:]) != record['test_spin_ids']):
            raise ValueError('Test partition does not match the checkpoint')
        result.append(list(session.numbers[boundary - history_size:]))
    return result
