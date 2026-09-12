from importlib import import_module


_EXPORTS = {
    "plot_training_results": "visualization",
    "plot_comparison": "visualization",
    "get_wheel_neighbors": "near_miss",
    "get_table_neighbors": "near_miss",
    "get_wheel_distance": "near_miss",
    "get_table_distance": "near_miss",
    "is_near_miss_wheel": "near_miss",
    "is_near_miss_table": "near_miss",
    "compute_near_miss_features": "near_miss",
    "get_near_miss_zones": "near_miss",
    "encode_wheel_position": "near_miss",
    "compute_wheel_sector_frequencies": "near_miss",
    "NearMissTracker": "near_miss",
    "EUROPEAN_WHEEL_ORDER": "near_miss",
    "WHEEL_POSITION": "near_miss",
    "TABLE_LAYOUT": "near_miss",
    "BiasLevel": "bias_detection",
    "BiasResult": "bias_detection",
    "WheelBiasReport": "bias_detection",
    "chi_square_test": "bias_detection",
    "sector_bias_test": "bias_detection",
    "runs_test": "bias_detection",
    "color_bias_test": "bias_detection",
    "neighbor_clustering_test": "bias_detection",
    "detect_hot_cold_numbers": "bias_detection",
    "generate_bias_report": "bias_detection",
    "print_bias_report": "bias_detection",
    "WheelBiasAnalyzer": "bias_detection",
    "classify_bias": "bias_detection",
    "LSTMPredictor": "predictor",
    "LSTMNetwork": "predictor",
    "RoulettePredictor": "predictor",
    "ExtraTreesPredictor": "predictor",
    "PredictionWithConfidence": "predictor",
    "PROBABILITY_THRESHOLD": "predictor",
    "FAIR_PROBABILITY": "predictor",
    "get_optimal_device": "predictor"
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module('.' + module, __name__), name)
    globals()[name] = value
    return value
