from importlib import import_module


_EXPORTS = {
    "DQNAgent": "dqn_agent",
    "FuzzyAdaptiveDQN": "fuzzy_adaptive",
    "FuzzyEpsilonController": "fuzzy_adaptive",
    "ActionQualityEstimator": "fuzzy_adaptive",
    "SuccessfulActionsEstimator": "fuzzy_adaptive",
    "HistoricalTrendAnalyzer": "fuzzy_adaptive",
    "ExplorationDiversityEstimator": "fuzzy_adaptive",
    "BehavioralAgent": "behavioral",
    "GamblersFallacyAgent": "behavioral",
    "HotHandAgent": "behavioral",
    "RandomAgent": "behavioral",
    "MixedBehaviorAgent": "behavioral",
    "SectorBiasAgent": "behavioral",
    "BehaviorType": "behavioral",
    "BetType": "behavioral",
    "create_agent": "behavioral",
    "compare_behavioral_agents": "behavioral",
    "HyperHeuristicAgent": "hyper_heuristic",
    "DQNHyperHeuristic": "hyper_heuristic",
    "LLHType": "hyper_heuristic",
    "LowLevelHeuristic": "hyper_heuristic",
    "HotNumbersLLH": "hyper_heuristic",
    "ColdNumbersLLH": "hyper_heuristic",
    "SectorBettingLLH": "hyper_heuristic",
    "MartingaleLLH": "hyper_heuristic",
    "FibonacciLLH": "hyper_heuristic",
    "PassActionLLH": "hyper_heuristic"
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module('.' + module, __name__), name)
    globals()[name] = value
    return value
