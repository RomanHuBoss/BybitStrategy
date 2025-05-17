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

# Проверка GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Настройка логгирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


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
    def __init__(self, model_folder, threshold=0.7):
        self.model_folder = model_folder
        self.threshold = threshold
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.training_config = None

        self.load_model(self.model_folder)

    def load_model(self, model_folder: str) -> None:
        """Загружает модель и связанные данные"""
        with open(os.path.join(model_folder, 'training_config.pkl'), 'rb') as f:
            self.training_config = pickle.load(f)

        if 'percentage_pairs' not in self.training_config:
            raise ValueError("Конфигурация обучения должна содержать 'percentage_pairs'.")

        with open(os.path.join(model_folder, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)

        self.model = self._build_model(len(self.feature_columns))
        self.model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pth')))
        self.model.to(device)
        self.model.eval()

        self.scaler.min_ = np.load(os.path.join(model_folder, 'scaler_min.npy'))
        self.scaler.scale_ = np.load(os.path.join(model_folder, 'scaler_scale.npy'))

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
            print(f"Данные загружены. Строк: {len(self.df)}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")
            raise

    def create_features(self) -> None:
        """Генерация всех фичей и целевых переменных"""
        if self.df is None:
            raise ValueError("Данные не загружены")

        self._add_time_features()
        self._add_technical_indicators()
        self._add_chart_patterns()

    def _add_chart_patterns(self) -> None:
        """Добавление признаков графических паттернов"""
        if self.df is None:
            raise ValueError("Данные не загружены")

        # Создаем временный словарь для новых колонок
        new_columns = {
            'head_shoulders': np.zeros(len(self.df)),
            'double_top': np.zeros(len(self.df)),
            'double_bottom': np.zeros(len(self.df)),
            'triangle_ascending': np.zeros(len(self.df)),
            'triangle_descending': np.zeros(len(self.df)),
            'wedge_rising': np.zeros(len(self.df)),
            'wedge_falling': np.zeros(len(self.df))
        }

        # Анализируем ценовые движения для выявления паттернов
        close_prices = self.df['close'].values
        high_prices = self.df['high'].values
        low_prices = self.df['low'].values

        # Параметры для обнаружения паттернов
        lookback = self.training_config['chart_patterns']['lookback_window']  # Количество свечей для анализа паттерна
        min_pattern_length = self.training_config['chart_patterns']['min_pattern_length']  # Минимальная длина паттерна в свечах

        for i in range(lookback, len(self.df)):
            current_window_high = high_prices[i - lookback:i]
            current_window_low = low_prices[i - lookback:i]
            current_window_close = close_prices[i - lookback:i]

            # 1. Голова и плечи (Head and Shoulders)
            left_shoulder = np.argmax(current_window_high[:lookback // 3])
            head = np.argmax(current_window_high[lookback // 3:2 * lookback // 3]) + lookback // 3
            right_shoulder = np.argmax(current_window_high[2 * lookback // 3:]) + 2 * lookback // 3

            neckline = (current_window_low[left_shoulder] + current_window_low[right_shoulder]) / 2

            if (left_shoulder < head > right_shoulder and
                    current_window_high[left_shoulder] < current_window_high[head] > current_window_high[
                        right_shoulder] and
                    abs(current_window_high[left_shoulder] - current_window_high[right_shoulder]) < 0.01 *
                    current_window_high[head] and
                    current_window_close[-1] < neckline):
                new_columns['head_shoulders'][i] = 1

            # 2. Двойная вершина (Double Top)
            first_top = np.argmax(current_window_high[:lookback // 2])
            second_top = np.argmax(current_window_high[lookback // 2:]) + lookback // 2
            valley = np.argmin(current_window_low[first_top:second_top]) + first_top

            if (first_top < second_top and
                    abs(current_window_high[first_top] - current_window_high[second_top]) < 0.01 * current_window_high[
                        first_top] and
                    current_window_close[-1] < current_window_low[valley]):
                new_columns['double_top'][i] = 1

            # 3. Двойное дно (Double Bottom)
            first_bottom = np.argmin(current_window_low[:lookback // 2])
            second_bottom = np.argmin(current_window_low[lookback // 2:]) + lookback // 2
            peak = np.argmax(current_window_high[first_bottom:second_bottom]) + first_bottom

            if (first_bottom < second_bottom and
                    abs(current_window_low[first_bottom] - current_window_low[second_bottom]) < 0.01 *
                    current_window_low[first_bottom] and
                    current_window_close[-1] > current_window_high[peak]):
                new_columns['double_bottom'][i] = 1

            # 4. Восходящий треугольник (Ascending Triangle)
            resistance = np.max(current_window_high)
            higher_lows = all(
                current_window_low[j] > current_window_low[j - 1] for j in range(1, len(current_window_low)))

            if (higher_lows and
                    np.std(current_window_high) < 0.005 * resistance and
                    current_window_close[-1] > resistance):
                new_columns['triangle_ascending'][i] = 1

            # 5. Нисходящий треугольник (Descending Triangle)
            support = np.min(current_window_low)
            lower_highs = all(
                current_window_high[j] < current_window_high[j - 1] for j in range(1, len(current_window_high)))

            if (lower_highs and
                    np.std(current_window_low) < 0.005 * support and
                    current_window_close[-1] < support):
                new_columns['triangle_descending'][i] = 1

            # 6. Восходящий клин (Rising Wedge)
            if (all(current_window_high[j] > current_window_high[j - 1] for j in range(1, len(current_window_high))) and
                    all(current_window_low[j] > current_window_low[j - 1] for j in
                        range(1, len(current_window_low))) and
                    (current_window_close[-1] < current_window_low[-1])):
                new_columns['wedge_rising'][i] = 1

            # 7. Нисходящий клин (Falling Wedge)
            if (all(current_window_high[j] < current_window_high[j - 1] for j in range(1, len(current_window_high))) and
                    all(current_window_low[j] < current_window_low[j - 1] for j in
                        range(1, len(current_window_low))) and
                    (current_window_close[-1] > current_window_high[-1])):
                new_columns['wedge_falling'][i] = 1

        # Конвертируем словарь в DataFrame и объединяем с основным
        new_columns_df = pd.DataFrame(new_columns)
        self.df = pd.concat([self.df, new_columns_df], axis=1)

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

    def _build_model(self, input_size: int) -> LSTMModel:
        """Создает новую модель"""
        output_size = 2 * len(self.training_config['percentage_pairs'])
        return LSTMModel(input_size, output_size).to(device)

    def predict_last_candles(self) -> Dict:
        """Прогнозирует для последних свечей"""
        required = 2 * self.training_config['backward_window']
        if len(self.df) < required:
            raise ValueError(f"Нужно минимум {required} свечей")

        data = self.df[self.feature_columns].iloc[-required:].values
        data = np.nan_to_num(data, nan=np.nanmedian(data, axis=0))
        scaled = self.scaler.transform(data)

        X = np.array([
            scaled[i:i + self.training_config['backward_window']]
            for i in range(self.training_config['backward_window'])
        ])

        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X).to(device)).cpu().numpy()

        def safe_median(arr):
            return float(np.median(arr)) if len(arr) > 0 else 0.0

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
            },
            'median_values': {
                'long': {'median_sl': 0.0, 'median_tp': 0.0},
                'short': {'median_sl': 0.0, 'median_tp': 0.0}
            },
            'chart_patterns': self._get_last_chart_patterns()  # Добавляем информацию о паттернах
        }

        # Заполняем вероятности
        for i, (sl, tp) in enumerate(self.training_config['percentage_pairs']):
            key = f"SL_{sl:.2%}_TP_{tp:.2%}"
            result['long'][key] = float(long_probs[i])
            result['short'][key] = float(short_probs[i])

        # Вычисляем значения для prob > threshold
        long_sl_values = [
            sl for i, (sl, _) in enumerate(self.training_config['percentage_pairs'])
            if long_probs[i] > self.threshold
        ]
        long_tp_values = [
            tp for i, (_, tp) in enumerate(self.training_config['percentage_pairs'])
            if long_probs[i] > self.threshold
        ]

        short_sl_values = [
            sl for i, (sl, _) in enumerate(self.training_config['percentage_pairs'])
            if short_probs[i] > self.threshold
        ]
        short_tp_values = [
            tp for i, (_, tp) in enumerate(self.training_config['percentage_pairs'])
            if short_probs[i] > self.threshold
        ]

        # Максимальные значения
        result['max_values']['long']['max_sl'] = safe_max(long_sl_values)
        result['max_values']['long']['max_tp'] = safe_max(long_tp_values)
        result['max_values']['short']['max_sl'] = safe_max(short_sl_values)
        result['max_values']['short']['max_tp'] = safe_max(short_tp_values)

        # Медианные значения
        result['median_values']['long']['median_sl'] = safe_median(long_sl_values)
        result['median_values']['long']['median_tp'] = safe_median(long_tp_values)
        result['median_values']['short']['median_sl'] = safe_median(short_sl_values)
        result['median_values']['short']['median_tp'] = safe_median(short_tp_values)

        return result

    def _get_last_chart_patterns(self) -> Dict[str, bool]:
        """Возвращает обнаруженные графические паттерны на последней свече"""
        if self.df is None or len(self.df) == 0:
            return {}

        last_row = self.df.iloc[-1]
        patterns = {
            'head_shoulders': bool(last_row['head_shoulders']),
            'double_top': bool(last_row['double_top']),
            'double_bottom': bool(last_row['double_bottom']),
            'triangle_ascending': bool(last_row['triangle_ascending']),
            'triangle_descending': bool(last_row['triangle_descending']),
            'wedge_rising': bool(last_row['wedge_rising']),
            'wedge_falling': bool(last_row['wedge_falling'])
        }

        # Фильтруем только активные паттерны
        active_patterns = {name: active for name, active in patterns.items() if active}
        return active_patterns

    def run_csv_prediction_pipeline(self, csv_path: str) -> Dict:
        """Полный цикл прогнозирования"""
        print(f"Запуск прогноза для {csv_path}")

        self.load_trading_data_from_csv(csv_path)
        self.create_features()

        df_copy =self.df.copy()

        for i in range(0, len(df_copy)):
            first_number = i
            last_number = i + self.training_config['backward_window'] * 2

            if last_number > len(df_copy):
                continue

            self.df = df_copy[first_number: first_number + last_number]
            prediction = self.predict_last_candles()

            for direction in ['long', 'short']:
                if any(prob > self.threshold for prob in prediction[direction].values()):
                    print('---')
                    print(f"Результаты для {direction}:")
                    print(f"Свеча от даты {self.df['open_time'].iloc[-1]}")

                    for strat, prob in prediction[direction].items():
                        if prob > self.threshold:
                            print(f"{strat}: {prob:.2%}")

                    print(f"Максимальные значения (prob > {self.threshold * 100}%):")
                    print(
                        f"{direction}: SL={prediction['max_values'][direction]['max_sl']:.2%}, TP={prediction['max_values'][direction]['max_tp']:.2%}")

                    print(f"Медианные значения (prob > {self.threshold * 100}%):")
                    print(
                        f"{direction}: SL={prediction['median_values'][direction]['median_sl']:.2%}, TP={prediction['median_values'][direction]['median_tp']:.2%}")

                    #Выводим обнаруженные паттерны
                    if prediction['chart_patterns']:
                        print("\nОбнаруженные графические паттерны:")
                        for pattern, active in prediction['chart_patterns'].items():
                            if active:
                                print(f"- {pattern}")
                    else:
                        print("\nГрафические паттерны не обнаружены")


if __name__ == "__main__":
    model_folder = os.path.join("models", "3m-60forward-10backward-17-05-2025 22-17-14")
    predictor = CryptoModelPredictor(model_folder=model_folder, threshold=0.6)


    prediction = predictor.run_csv_prediction_pipeline(
        #csv_path=os.path.join("historical_data", "BTCUSDT", "3m", "daily", "BTCUSDT-3m-2025-05-05.csv"),
        csv_path=os.path.join("downloads", "OBTUSDT_3m_2025-05-17-20-31.csv"),
    )