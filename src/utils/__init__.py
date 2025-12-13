from .visualization import plot_training_results, plot_comparison
from .near_miss import (
    get_wheel_neighbors,
    get_table_neighbors,
    get_wheel_distance,
    get_table_distance,
    is_near_miss_wheel,
    is_near_miss_table,
    compute_near_miss_features,
    get_near_miss_zones,
    encode_wheel_position,
    compute_wheel_sector_frequencies,
    NearMissTracker,
    EUROPEAN_WHEEL_ORDER,
    WHEEL_POSITION,
    TABLE_LAYOUT
)
from .bias_detection import (
    BiasLevel,
    BiasResult,
    WheelBiasReport,
    chi_square_test,
    sector_bias_test,
    runs_test,
    color_bias_test,
    neighbor_clustering_test,
    detect_hot_cold_numbers,
    generate_bias_report,
    print_bias_report,
    WheelBiasAnalyzer,
    classify_bias
)
from .predictor import (
    LSTMPredictor,
    LSTMNetwork,
    RoulettePredictor,
    ExtraTreesPredictor,
    PredictionWithConfidence,
    PROBABILITY_THRESHOLD,
    FAIR_PROBABILITY,
    get_optimal_device
)

__all__ = [
    'plot_training_results', 
    'plot_comparison',
    # Near-miss utilities
    'get_wheel_neighbors',
    'get_table_neighbors',
    'get_wheel_distance',
    'get_table_distance',
    'is_near_miss_wheel',
    'is_near_miss_table',
    'compute_near_miss_features',
    'get_near_miss_zones',
    'encode_wheel_position',
    'compute_wheel_sector_frequencies',
    'NearMissTracker',
    'EUROPEAN_WHEEL_ORDER',
    'WHEEL_POSITION',
    'TABLE_LAYOUT',
    # Bias detection
    'BiasLevel',
    'BiasResult',
    'WheelBiasReport',
    'chi_square_test',
    'sector_bias_test',
    'runs_test',
    'color_bias_test',
    'neighbor_clustering_test',
    'detect_hot_cold_numbers',
    'generate_bias_report',
    'print_bias_report',
    'WheelBiasAnalyzer',
    'classify_bias',
    'LSTMPredictor',
    'LSTMNetwork',
    'RoulettePredictor',
    'ExtraTreesPredictor',
    'PredictionWithConfidence',
    'PROBABILITY_THRESHOLD',
    'FAIR_PROBABILITY',
    'get_optimal_device'
]
