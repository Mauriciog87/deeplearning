import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--capture-import', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    from src.agents.dqn_agent import DQNAgent
    from src.checkpoints import seed_everything
    from src.utils.lstm_predictor import LSTMPredictor
    from src.utils.predictor import ExtraTreesPredictor

    seed_everything(42, args.device)
    history = np.random.default_rng(42).integers(0, 37, 80).tolist()
    lstm = LSTMPredictor(device=args.device)
    fit = lstm.fit(history, epochs=2, batch_size=16)
    if not lstm.is_trained:
        raise RuntimeError(f'LSTM training failed: {fit}')
    probabilities = lstm.predict_proba(history)
    np.testing.assert_allclose(np.sum(probabilities), 1, atol=1e-6)
    trees = ExtraTreesPredictor(seed=42)
    trees.fit(history)
    np.testing.assert_allclose(np.sum(trees.predict_proba(history)), 1, atol=1e-6)
    agent = DQNAgent(hidden_size=16, embedding_dim=4, batch_size=4, device=args.device)
    for index in range(8):
        observed = np.asarray(history[index:index + 20])
        following = np.asarray(history[index + 1:index + 21])
        agent.remember(observed, 1, 46, 0, following, 1, False, np.ones(47, dtype=bool))
    if not np.isfinite(agent.train()):
        raise RuntimeError('DQN training produced nonfinite loss')
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = str(Path(folder) / 'lstm.pt')
        lstm.save(checkpoint)
        restored = LSTMPredictor(device=args.device)
        restored.load(checkpoint)
        np.testing.assert_allclose(restored.predict_proba(history), probabilities, rtol=0, atol=0)
        checkpoint = str(Path(folder) / 'dqn.pt')
        agent.save(checkpoint)
        expected_loss = agent.train()
        expected = {key: value.clone() for key, value in agent.q_network.state_dict().items()}
        restored_agent = DQNAgent(device=args.device)
        restored_agent.load(checkpoint)
        if restored_agent.train() != expected_loss:
            raise AssertionError('Checkpoint did not resume the same learning step')
        for key, value in expected.items():
            torch.testing.assert_close(restored_agent.q_network.state_dict()[key], value, rtol=0, atol=0)
    if args.capture_import:
        from importlib import import_module
        for module in ('src.capture.roulette_monitor', 'src.gui.app', 'easyocr', 'torchvision'):
            import_module(module)
    print(json.dumps({'device': str(agent.device), 'torch': torch.__version__,
                      'cuda_runtime': torch.version.cuda, 'lstm_fit': True,
                      'extra_trees_fit': True, 'dqn_resume_exact': True,
                      'capture_gui_imports': args.capture_import}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
