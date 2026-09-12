import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('core', 'full'), default='full')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    tests = ['discover', '-s', str(root), '-p', 'test_*.py']
    if args.profile == 'core':
        tests = ['test_core_imports', 'test_cli_integrity', 'test_database_integrity', 'test_evaluation_harness',
                 'test_randomness', 'test_multiple_testing', 'test_heatmaps',
                 'test_statistical_inference', 'test_settlement_environment', 'test_calibration_contract',
                 'test_joint_calibration', 'test_sequential_inference', 'test_sequential_analyzer',
                 'test_expected_value', 'test_change_detection', 'test_online_recalibration', 'test_research_benchmark']
    return subprocess.run([sys.executable, '-m', 'unittest', *tests], cwd=root, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
