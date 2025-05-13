import numpy as np
from sklearn.preprocessing import MinMaxScaler

TRAINING_CONFIG = {
    'epochs': 1, # максимальное число эпох обучения
    'stop_loss_levels': np.arange(0.001, 0.02, 0.001), # уровни стоп-лосса (x100 - получим % от текущей цены)
    'take_profit_multipliers': np.arange(2, 6, 1), # множители для уровней тейк-профит (x100 - получим % от текущей цены)
    'scaler':  MinMaxScaler(feature_range=(0, 1)),
    'backward_window': 60, # кол-во свечей, предшествующих текущей, для анализа ретроспективы
    'forward_window': 180, # кол-во свечей, следующих за текущей, для прогноза динамики развития цены
    #'training_csv_file': 'c:\\Users\\roman\\PycharmProjects\\Bybit\\historical_data\\BTCUSDT\\monthly\\combined_csv.csv',
    'training_csv_file': 'BTCUSDT-1m-2025-04.csv',
    'evaluating_csv_file': 'BTCUSDT-1m-2025-05-07.csv',
}

TRAINING_CONFIG['percentage_pairs'] = [
    (sl, sl * multiplier)
    for sl in TRAINING_CONFIG['stop_loss_levels']
    for multiplier in TRAINING_CONFIG['take_profit_multipliers']
]