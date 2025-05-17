import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

TRAINING_CONFIG = {
    'epochs': 10, # максимальное число эпох обучения
    'stop_loss_levels': np.arange(0.004, 0.04, 0.001), # уровни стоп-лосса (x100 - получим % от текущей цены)
    'take_profit_multipliers': np.arange(2, 7, 1), # множители для уровней тейк-профит (x100 - получим % от текущей цены)
    'scaler':  MinMaxScaler(feature_range=(0, 1)),
    'backward_window': 30, # кол-во свечей, предшествующих текущей, для анализа ретроспективы
    'forward_window': 30, # кол-во свечей, следующих за текущей, для прогноза динамики развития цены
    'windows': np.arange(10, 110, 10), #окна, в пределах которых формируются технические индикаторы
    'training_csv_file': os.path.join('historical_data', 'BTCUSDT', '3m', 'monthly', 'combined_csv.csv'),
    'evaluating_csv_file': os.path.join('historical_data', 'BTCUSDT', '3m', 'daily', 'BTCUSDT-3m-2025-05-16.csv'),
    'chart_patterns': {
        'lookback_window': 20,  # Количество свечей для анализа паттернов
        'min_pattern_length': 5  # Минимальная длина паттерна
    }
}

TRAINING_CONFIG['percentage_pairs'] = [
    (sl, sl * multiplier)
    for sl in TRAINING_CONFIG['stop_loss_levels']
    for multiplier in TRAINING_CONFIG['take_profit_multipliers']
]