import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

timeframe = 3

TRAINING_CONFIG = {
    'timeframe': timeframe,
    'epochs': 10, # максимальное число эпох обучения
    'stop_loss_levels': np.arange(0.003, 0.06, 0.001), # уровни стоп-лосса (x100 - получим % от текущей цены)
    'take_profit_multipliers': np.arange(2, 6, 1), # множители для уровней тейк-профит (x100 - получим % от текущей цены)
    'scaler':  MinMaxScaler(feature_range=(0, 1)),
    'backward_window': 30, # кол-во свечей, предшествующих текущей, для анализа ретроспективы
    'forward_window': 90, # кол-во свечей, следующих за текущей, для прогноза динамики развития цены
    'windows': np.arange(5, 25, 5), #окна, в пределах которых формируются технические индикаторы
    'training_csv_file': os.path.join('historical_data', 'BTCUSDT', f'{timeframe}m', 'monthly', 'combined_csv.csv'),
    'evaluating_csv_file': os.path.join('historical_data', 'BTCUSDT', f'{timeframe}m', 'daily', f'BTCUSDT-{timeframe}m-2025-05-16.csv'),
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