import json
import os
from pathlib import Path
import random
import tempfile
from collections import deque

import numpy as np


SCHEMA_VERSION = 2
OBSERVATION_VERSION = 'history-gain-ratio-v2'


def plain_state(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return [plain_state(item) for item in value]
    return value


def capture_rng(include_torch=True):
    state = {'python': plain_state(random.getstate()), 'numpy': plain_state(np.random.get_state())}
    if include_torch:
        import torch
        state['torch'] = torch.get_rng_state()
        state['cuda'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def restore_rng(state):
    python = state['python']
    random.setstate((python[0], tuple(python[1]), python[2]))
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if 'torch' in state:
        import torch
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def atomic_save(path, state, *, neural=True):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            if neural:
                import torch
                torch.save(plain_state(state), stream)
            else:
                stream.write(json.dumps(plain_state(state), allow_nan=False).encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model_type, *, neural=True, device='cpu'):
    if neural:
        import torch
        state = torch.load(path, map_location=device, weights_only=True)
    else:
        state = json.loads(Path(path).read_text(encoding='utf-8'))
    if state.get('schema_version') != SCHEMA_VERSION or state.get('model_type') != model_type:
        raise ValueError('Unsupported checkpoint schema or model type; retrain the model')
    if state.get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('Checkpoint observation contract does not match this version')
    return state


def seed_everything(seed, device='auto'):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return 'cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu'
