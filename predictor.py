import json
import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from torch.utils.data import DataLoader, TensorDataset

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 128, batch_first=True)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self._apply_bn(x, self.bn1)

        x, _ = self.lstm2(x)
        x = self._apply_bn(x, self.bn2)

        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Последний шаг
        x = self.bn3(x)
        x = self.dropout(x)

        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x.view(x.size(0), 2, -1)

    def _apply_bn(self, x: torch.Tensor, bn_layer: nn.Module) -> torch.Tensor:
        return bn_layer(x.permute(0, 2, 1)).permute(0, 2, 1)


class CryptoModelPredictor:
    def __init__(self, config_path: Optional[str] = None):
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[LSTMModel] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns: Optional[List[str]] = None
        self.training_config = self._load_config(config_path)

    def _load_config(self, path: Optional[str]) -> Dict:
        """Загружает конфиг из файла или использует значения по умолчанию"""
        default_config = {
            'windows': [5, 10, 20, 50],
            'forward_window': 20,
            'backward_window': 60,
            'percentage_pairs': [
                (0.01, 0.02),
                (0.02, 0.04),
                (0.03, 0.06)
            ]
        }
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default_config

    def load_trading_data_from_csv(self, file_path: str) -> None:
        """Загружает и очищает данные из CSV"""
        try:
            self.df = pd.read_csv(file_path)
            self.df['open_time'] = pd.to_datetime(self.df['open_time'], unit='ms')
            cols_to_drop = [col for col in [
                'close_time', 'quote_volume', 'count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ] if col in self.df.columns]
            self.df.drop(cols_to_drop, axis=1, inplace=True)
            logger.info(f"Данные загружены. Строк: {len(self.df)}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")
            raise

    def create_features(self) -> None:
        """Генерация всех фичей и целевых переменных"""
        if self.df is None:
            raise ValueError("Данные не загружены")

        self._add_time_features()
        self._add_technical_indicators()
        self._generate_targets()

    def _add_time_features(self) -> None:
        """Добавляет временные фичи с циклическим кодированием"""
        dt = self.df['open_time'].dt
        self.df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        self.df['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)

    def _add_technical_indicators(self) -> None:
        """Добавляет технические индикаторы"""
        close = self.df['close']
        volume = self.df['volume']

        # MACD
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()

        # OBV
        self.df['obv'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

        # Остальные индикаторы
        for window in self.training_config['windows']:
            self.df[f'sma_{window}'] = SMAIndicator(close=close, window=window).sma_indicator()
            self.df[f'ema_{window}'] = EMAIndicator(close=close, window=window).ema_indicator()
            self.df[f'rsi_{window}'] = RSIIndicator(close=close, window=window).rsi()

            bb = BollingerBands(close=close, window=window)
            self.df[f'bb_bbm_{window}'] = bb.bollinger_mavg()
            self.df[f'bb_bbh_{window}'] = bb.bollinger_hband()
            self.df[f'bb_bbl_{window}'] = bb.bollinger_lband()
            self.df[f'obv_ma_{window}'] = self.df['obv'].rolling(window=window).mean()

        self.df = self.df.bfill().ffill().dropna()

    def _generate_targets(self) -> None:
        """Генерирует целевые переменные для обучения"""
        min_samples = self.training_config['forward_window']
        if len(self.df) < min_samples:
            raise ValueError(f"Нужно минимум {min_samples} свечей")

        close_prices = self.df['close'].values
        new_cols = {}

        for sl_pct, tp_pct in self.training_config['percentage_pairs']:
            new_cols[f'long_success_sl_{sl_pct:.4f}_tp_{tp_pct:.4f}'] = np.zeros(len(self.df))
            new_cols[f'short_success_sl_{sl_pct:.4f}_tp_{tp_pct:.4f}'] = np.zeros(len(self.df))

        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

        # Векторизованный расчет
        for i in range(len(self.df) - min_samples):
            current = close_prices[i]
            future = close_prices[i + 1: i + min_samples + 1]

            for sl, tp in self.training_config['percentage_pairs']:
                # LONG
                tp_hit = (future >= current * (1 + tp)).any()
                sl_hit = (future <= current * (1 - sl)).any()
                self.df.at[i, f'long_success_sl_{sl:.4f}_tp_{tp:.4f}'] = int(tp_hit and not sl_hit)

                # SHORT
                tp_hit = (future <= current * (1 - tp)).any()
                sl_hit = (future >= current * (1 + sl)).any()
                self.df.at[i, f'short_success_sl_{sl:.4f}_tp_{tp:.4f}'] = int(tp_hit and not sl_hit)

    def prepare_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для валидации"""
        required = self.training_config['backward_window'] + self.training_config['forward_window']
        if len(self.df) < required:
            raise ValueError(f"Нужно минимум {required} свечей")

        if self.feature_columns is None:
            self.feature_columns = [
                col for col in self.df.columns
                if not col.startswith(('long_success', 'short_success'))
                   and col not in ['open_time', 'close_time']
            ]

        data = self.df[self.feature_columns].values
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=np.nanmedian(data, axis=0))

        self.scaler.fit(data)
        scaled = self.scaler.transform(data)

        X, y_long, y_short = [], [], []
        for i in range(len(self.df) - required):
            X.append(scaled[i:i + self.training_config['backward_window']])

            long_probs = [
                self.df[f'long_success_sl_{sl:.4f}_tp_{tp:.4f}'].iloc[i + self.training_config['backward_window']]
                for sl, tp in self.training_config['percentage_pairs']
            ]
            short_probs = [
                self.df[f'short_success_sl_{sl:.4f}_tp_{tp:.4f}'].iloc[i + self.training_config['backward_window']]
                for sl, tp in self.training_config['percentage_pairs']
            ]

            y_long.append(long_probs)
            y_short.append(short_probs)

        return np.array(X), np.stack((y_long, y_short), axis=1)

    def load_model(self, model_folder: str) -> None:
        """Загружает модель и связанные данные"""
        with open(os.path.join(model_folder, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)

        with open(os.path.join(model_folder, 'training_config.pkl'), 'rb') as f:
            self.training_config = pickle.load(f)

        self.model = self._build_model(len(self.feature_columns))
        self.model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pth')))
        self.model.to(device)
        self.model.eval()

        self.scaler.min_ = np.load(os.path.join(model_folder, 'scaler_min.npy'))
        self.scaler.scale_ = np.load(os.path.join(model_folder, 'scaler_scale.npy'))

    def _build_model(self, input_size: int) -> LSTMModel:
        """Создает новую модель"""
        output_size = 2 * len(self.training_config['percentage_pairs'])
        return LSTMModel(input_size, output_size).to(device)

    def predict_last_30_candles(self) -> Dict:
        """Прогнозирует для последних 30 свечей"""
        required = self.training_config['backward_window'] + 30
        if len(self.df) < required:
            raise ValueError(f"Нужно минимум {required} свечей")

        data = self.df[self.feature_columns].iloc[-required:].values
        data = np.nan_to_num(data, nan=np.nanmedian(data, axis=0))
        scaled = self.scaler.transform(data)

        X = np.array([
            scaled[i:i + self.training_config['backward_window']]
            for i in range(30)
        ])

        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X).to(device)).cpu().numpy()

        def safe_max(arr):
            return float(np.max(arr)) if len(arr) > 0 else 0.0

        long_probs = preds[:, 0].mean(axis=0)
        short_probs = preds[:, 1].mean(axis=0)

        result = {
            'long': {},
            'short': {},
            'max_values': {
                'long': {'max_sl': 0.0, 'max_tp': 0.0},
                'short': {'max_sl': 0.0, 'max_tp': 0.0}
            }
        }

        # Заполняем вероятности
        for i, (sl, tp) in enumerate(self.training_config['percentage_pairs']):
            key = f"SL_{sl:.2%}_TP_{tp:.2%}"
            result['long'][key] = float(long_probs[i])
            result['short'][key] = float(short_probs[i])

        # Вычисляем максимальные значения для prob > 0.55
        long_sl_values = [
            sl for i, (sl, _) in enumerate(self.training_config['percentage_pairs'])
            if long_probs[i] > 0.55
        ]
        long_tp_values = [
            tp for i, (_, tp) in enumerate(self.training_config['percentage_pairs'])
            if long_probs[i] > 0.55
        ]

        short_sl_values = [
            sl for i, (sl, _) in enumerate(self.training_config['percentage_pairs'])
            if short_probs[i] > 0.55
        ]
        short_tp_values = [
            tp for i, (_, tp) in enumerate(self.training_config['percentage_pairs'])
            if short_probs[i] > 0.55
        ]

        result['max_values']['long']['max_sl'] = safe_max(long_sl_values)
        result['max_values']['long']['max_tp'] = safe_max(long_tp_values)
        result['max_values']['short']['max_sl'] = safe_max(short_sl_values)
        result['max_values']['short']['max_tp'] = safe_max(short_tp_values)

        return result

    def run_prediction_pipeline(self, csv_path: str, model_folder: str) -> Dict:
        """Полный цикл прогнозирования"""
        logger.info(f"Запуск прогноза для {csv_path}")

        self.load_model(model_folder)
        self.load_trading_data_from_csv(csv_path)
        self.create_features()

        prediction = self.predict_last_30_candles()

        # Логируем результаты
        logger.info("\nРезультаты для LONG:")
        for strat, prob in prediction['long'].items():
            if prob > 0.55:
                logger.info(f"{strat}: {prob:.2%}")

        logger.info("\nРезультаты для SHORT:")
        for strat, prob in prediction['short'].items():
            if prob > 0.55:
                logger.info(f"{strat}: {prob:.2%}")

        logger.info("\nМаксимальные значения:")
        logger.info(
            f"LONG: SL={prediction['max_values']['long']['max_sl']:.2%}, TP={prediction['max_values']['long']['max_tp']:.2%}")
        logger.info(
            f"SHORT: SL={prediction['max_values']['short']['max_sl']:.2%}, TP={prediction['max_values']['short']['max_tp']:.2%}")

        return prediction


if __name__ == "__main__":
    predictor = CryptoModelPredictor()
    prediction = predictor.run_prediction_pipeline(
        csv_path=os.path.join("downloads", "BOBAUSDT_1m_2025-05-16-20-19.csv"),
        model_folder=os.path.join("models", "model-4month-16-05-2025 11-37-14")
    )