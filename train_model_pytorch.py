import json
import os
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training_config import TRAINING_CONFIG
from datetime import datetime

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 128, batch_first=True)

        # Normalization and dropout
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        # Dense layers
        self.fc1 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self._apply_bn(x, self.bn1)

        x, _ = self.lstm2(x)
        x = self._apply_bn(x, self.bn2)

        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Last timestep only
        x = self.bn3(x)
        x = self.dropout(x)

        # Dense layers
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x.view(x.size(0), 2, -1)

    def _apply_bn(self, x, bn_layer):
        """Helper for batch norm application on LSTM outputs"""
        return bn_layer(x.permute(0, 2, 1)).permute(0, 2, 1)


class CryptoModelTrainer:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

    # Основные методы для работы с внутренним датафреймом
    def load_trading_data_from_csv(self, file_path) -> None:
        """Загрузка и предварительная обработка внутреннего датафрейма"""
        try:
            self.df = pd.read_csv(file_path)
            self.df['open_time'] = pd.to_datetime(self.df['open_time'], unit='ms')
            self.df.drop(
                ['close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'],
                axis=1, inplace=True)
            print("Данные успешно загружены. Доступно строк:", len(self.df))
        except Exception as e:
            raise ValueError(f"Ошибка загрузки данных: {str(e)}")

    def create_features(self) -> None:
        """Полная обработка внутреннего датафрейма (фичи + индикаторы)"""
        if self.df is None:
            raise ValueError("Внутренний датафрейм не загружен")

        self._add_time_features_to_df()
        self._add_technical_indicators_to_df()
        self._add_chart_patterns_to_df()
        self._analyze_price_movements_on_df()

    def _add_chart_patterns_to_df(self) -> None:
        """Добавление признаков графических паттернов к внутреннему датафрейму"""
        if self.df is None:
            raise ValueError("Внутренний датафрейм не загружен")

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
        lookback = TRAINING_CONFIG['chart_patterns']['lookback_window']  # Количество свечей для анализа паттерна
        min_pattern_length = TRAINING_CONFIG['chart_patterns']['min_pattern_length']  # Минимальная длина паттерна в свечах

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

    # Приватные методы для работы только с внутренним датафреймом
    def _add_time_features_to_df(self) -> None:
        """Добавление временных фичей к внутреннему датафрейму"""
        self.df['day_of_week'] = self.df['open_time'].dt.dayofweek
        self.df['month'] = self.df['open_time'].dt.month
        self.df['hour'] = self.df['open_time'].dt.hour

        # Циклическое кодирование
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        self.df.drop(['day_of_week', 'month', 'hour'], axis=1, inplace=True)

    def _add_technical_indicators_to_df(self) -> None:
        """Добавление технических индикаторов к внутреннему датафрейму"""
        # Создаем временный словарь для новых колонок
        new_columns = {}

        # MACD
        macd = MACD(close=self.df['close'], window_slow=26, window_fast=12, window_sign=9)
        new_columns['macd'] = macd.macd()
        new_columns['macd_signal'] = macd.macd_signal()
        new_columns['macd_diff'] = macd.macd_diff()

        # OBV
        new_columns['obv'] = OnBalanceVolumeIndicator(
            close=self.df['close'],
            volume=self.df['volume']
        ).on_balance_volume()

        # Добавляем индикаторы для каждого окна
        for window in TRAINING_CONFIG['windows']:
            new_columns[f'sma_{window}'] = SMAIndicator(close=self.df['close'], window=window).sma_indicator()
            new_columns[f'ema_{window}'] = EMAIndicator(close=self.df['close'], window=window).ema_indicator()
            new_columns[f'rsi_{window}'] = RSIIndicator(close=self.df['close'], window=window).rsi()

            bb = BollingerBands(close=self.df['close'], window=window)
            new_columns[f'bb_bbm_{window}'] = bb.bollinger_mavg()
            new_columns[f'bb_bbh_{window}'] = bb.bollinger_hband()
            new_columns[f'bb_bbl_{window}'] = bb.bollinger_lband()

            new_columns[f'obv_ma_{window}'] = new_columns['obv'].rolling(window=window).mean()

        # Конвертируем словарь в DataFrame и объединяем с основным
        new_columns_df = pd.DataFrame(new_columns)
        self.df = pd.concat([self.df, new_columns_df], axis=1)

        # Заполняем пропуски
        self.df = self.df.bfill().ffill().dropna()

        if len(self.df) == 0:
            raise ValueError("После добавления индикаторов DataFrame стал пустым")

    def _analyze_price_movements_on_df(self) -> None:
        """Анализ ценовых движений на внутреннем датафрейме"""
        if len(self.df) < TRAINING_CONFIG['forward_window']:
            raise ValueError(
                f"Недостаточно данных для анализа. Требуется минимум {TRAINING_CONFIG['forward_window']} свечей, доступно {len(self.df)}")

        close_prices = self.df['close'].values
        high_prices = self.df['high'].values
        low_prices = self.df['low'].values
        max_analysis_index = len(close_prices) - TRAINING_CONFIG['forward_window']

        # Создаем словарь для новых столбцов
        new_columns = {}

        # Инициализируем все новые столбцы нулями
        for sl_pct, tp_pct in TRAINING_CONFIG['percentage_pairs']:
            new_columns[f'long_success_sl_{sl_pct:.4f}_tp_{tp_pct:.4f}'] = np.zeros(len(self.df))
            new_columns[f'short_success_sl_{sl_pct:.4f}_tp_{tp_pct:.4f}'] = np.zeros(len(self.df))

        # Конвертируем словарь в DataFrame и объединяем с основным
        new_columns_df = pd.DataFrame(new_columns)
        self.df = pd.concat([self.df, new_columns_df], axis=1)

        # Теперь заполняем значения
        for sl_pct, tp_pct in TRAINING_CONFIG['percentage_pairs']:
            long_col = f'long_success_sl_{sl_pct:.4f}_tp_{tp_pct:.4f}'
            short_col = f'short_success_sl_{sl_pct:.4f}_tp_{tp_pct:.4f}'

            for i in range(max_analysis_index):
                current_price = close_prices[i]
                future_highs = high_prices[i + 1:i + TRAINING_CONFIG['forward_window'] + 1]
                future_lows = low_prices[i + 1:i + TRAINING_CONFIG['forward_window'] + 1]

                # LONG позиция
                long_tp = current_price * (1 + tp_pct)
                long_sl = current_price * (1 - sl_pct)

                # Проверяем, достигнут ли TP (по максимальной цене) или SL (по минимальной цене)
                for high, low in zip(future_highs, future_lows):
                    if high >= long_tp:  # TP достигнут по high
                        self.df.at[i, long_col] = 1
                        break
                    if low <= long_sl:  # SL достигнут по low
                        break

                # SHORT позиция
                short_tp = current_price * (1 - tp_pct)
                short_sl = current_price * (1 + sl_pct)

                # Проверяем, достигнут ли TP (по минимальной цене) или SL (по максимальной цене)
                for high, low in zip(future_highs, future_lows):
                    if low <= short_tp:  # TP достигнут по low
                        self.df.at[i, short_col] = 1
                        break
                    if high >= short_sl:  # SL достигнут по high
                        break

    def prepare_train_test_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""

        # сколько свечей нужно для обучения
        data_size_needed_to_work = TRAINING_CONFIG['backward_window'] + TRAINING_CONFIG['forward_window']

        if len(self.df) < data_size_needed_to_work:
            raise ValueError(f"Нужно минимум {data_size_needed_to_work} строк, доступно {len(self.df)}")

        # Сохраняем список признаков для последующего использования
        self.feature_columns = [
            col for col in self.df.columns
            if not col.startswith(('long_success', 'short_success'))
               and col not in ['open_time', 'close_time']
        ]

        # Заполнение NaN и масштабирование
        self.df[self.feature_columns] = self.df[self.feature_columns].fillna(self.df[self.feature_columns].median())
        scaled_data = self.scaler.fit_transform(self.df[self.feature_columns])

        # Создание последовательностей
        X, y_long, y_short = [], [], []
        for i in range(len(self.df) - data_size_needed_to_work):
            X.append(scaled_data[i:i + TRAINING_CONFIG['backward_window']])

            current_long_probs = []
            current_short_probs = []

            for sl, tp in TRAINING_CONFIG['percentage_pairs']:
                long_col = f'long_success_sl_{sl:.4f}_tp_{tp:.4f}'
                short_col = f'short_success_sl_{sl:.4f}_tp_{tp:.4f}'

                current_long_probs.append(self.df[long_col].iloc[i + TRAINING_CONFIG['backward_window']])
                current_short_probs.append(self.df[short_col].iloc[i + TRAINING_CONFIG['backward_window']])

            y_long.append(current_long_probs)
            y_short.append(current_short_probs)

        return train_test_split(
            np.array(X),
            np.stack((np.array(y_long), np.array(y_short)), axis=1),
            test_size=0.2,
            random_state=42
        )

    def prepare_validation_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для валидации модели (без разделения на train/test)"""
        # Проверка минимального количества данных
        data_size_needed_to_work = TRAINING_CONFIG['backward_window'] + TRAINING_CONFIG['forward_window']

        if len(self.df) < data_size_needed_to_work:
            raise ValueError(f"Нужно минимум {data_size_needed_to_work} строк, доступно {len(self.df)}")

        # Если feature_columns еще не определены, устанавливаем их
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in self.df.columns
                if not col.startswith(('long_success', 'short_success'))
                   and col not in ['open_time', 'close_time']
            ]

        # Получаем значения без имен признаков
        data_values = self.df[self.feature_columns].values

        # Заполнение NaN и масштабирование
        data_values = np.nan_to_num(data_values, nan=np.nanmedian(data_values, axis=0))
        scaled_data = self.scaler.transform(data_values)  # Теперь работаем с array, а не DataFrame

        # Создание последовательностей
        X, y_long, y_short = [], [], []
        for i in range(len(self.df) - data_size_needed_to_work):
            X.append(scaled_data[i:i + TRAINING_CONFIG['backward_window']])

            current_long_probs = []
            current_short_probs = []

            for sl, tp in TRAINING_CONFIG['percentage_pairs']:
                long_col = f'long_success_sl_{sl:.4f}_tp_{tp:.4f}'
                short_col = f'short_success_sl_{sl:.4f}_tp_{tp:.4f}'

                current_long_probs.append(self.df[long_col].iloc[i + TRAINING_CONFIG['backward_window']])
                current_short_probs.append(self.df[short_col].iloc[i + TRAINING_CONFIG['backward_window']])

            y_long.append(current_long_probs)
            y_short.append(current_short_probs)

        X = np.array(X)
        y = np.stack((y_long, y_short), axis=1)

        return X, y

    def build_model(self, input_size: int) -> nn.Module:
        """Создание модели LSTM"""
        output_size = 2 * len(TRAINING_CONFIG['percentage_pairs'])
        model = LSTMModel(input_size=input_size, output_size=output_size)
        return model.to(device)

    def train_model(self, X_train, X_test, y_train, y_test):
        # Конвертируем данные в тензоры НА CPU
        train_data = TensorDataset(
            torch.FloatTensor(X_train),  # Оставляем на CPU
            torch.FloatTensor(y_train)  # Оставляем на CPU
        )

        # DataLoader с pin_memory=True только для CPU тензоров
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            pin_memory=True  # Разрешаем pinning только для CPU данных
        )

        # Инициализация модели и перенос на устройство
        self.model = LSTMModel(
            input_size=X_train.shape[2],
            output_size=2 * len(TRAINING_CONFIG['percentage_pairs'])
        ).to(device)

        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()

        for epoch in range(TRAINING_CONFIG['epochs']):
            self.model.train()
            epoch_loss = 0

            for x_batch, y_batch in train_loader:
                # Переносим батч на устройство (GPU если доступно)
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            # Валидация
            val_loss, val_acc = self._validate(X_test, y_test)
            print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss / len(train_loader):.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    def _validate(self, X, y):
        self.model.eval()
        val_data = TensorDataset(
            torch.FloatTensor(X).to(device),
            torch.FloatTensor(y).to(device)
        )
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        total_loss = 0
        correct_per_pair = torch.zeros(len(TRAINING_CONFIG['percentage_pairs']),
                                       device=device)  # Для каждой пары (sl, tp)
        total_per_pair = torch.zeros(len(TRAINING_CONFIG['percentage_pairs']),
                                     device=device)  # Общее количество для каждой пары

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = self.model(x_batch)
                loss = nn.BCELoss()(outputs, y_batch)
                total_loss += loss.item()

                predicted = (outputs > 0.5).float()

                # Считаем правильные предсказания для каждой пары (sl, tp)
                for i in range(len(TRAINING_CONFIG['percentage_pairs'])):
                    correct_per_pair[i] += (predicted[:, :, i] == y_batch[:, :, i]).sum()
                    total_per_pair[i] += y_batch[:, :, i].numel()

        # Вычисляем accuracy для каждой пары
        val_acc_per_pair = (correct_per_pair / total_per_pair).cpu().numpy()

        # Выводим результаты
        for i, (sl, tp) in enumerate(TRAINING_CONFIG['percentage_pairs']):
            print(f"Val Acc (sl={sl:.4f}, tp={tp:.4f}): {val_acc_per_pair[i]:.4f}")

        # Средняя accuracy по всем парам (опционально)
        mean_val_acc = val_acc_per_pair.mean()
        return total_loss / len(val_loader), mean_val_acc

    def save_model(self, model_folder: str = '') -> None:
        """Сохранение модели и всех связанных данных"""
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        torch.save(self.model.state_dict(), os.path.join(model_folder, 'model.pth'))
        np.save(os.path.join(model_folder, 'scaler_min.npy'), self.scaler.min_)
        np.save(os.path.join(model_folder, 'scaler_scale.npy'), self.scaler.scale_)

        with open(os.path.join(model_folder, 'training_config.pkl'), 'wb') as f:
            pickle.dump(TRAINING_CONFIG, f)

        with open(os.path.join(model_folder, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)

    def load_model(self, model_folder: str = '') -> None:
        """Загрузка модели и всех связанных данных"""
        with open(os.path.join(model_folder, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)

        # Сначала создаем модель с правильной архитектурой
        self.model = self.build_model(input_size=len(self.feature_columns))
        # Затем загружаем веса
        self.model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pth')))
        self.model.to(device)

        self.scaler.min_ = np.load(os.path.join(model_folder, 'scaler_min.npy'))
        self.scaler.scale_ = np.load(os.path.join(model_folder, 'scaler_scale.npy'))

        global TRAINING_CONFIG
        with open(os.path.join(model_folder, 'training_config.pkl'), 'rb') as f:
            TRAINING_CONFIG = pickle.load(f)

    def evaluate_on_new_data(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.model.eval()
        criterion = nn.BCELoss()

        total_loss = 0.0
        correct_per_pair = torch.zeros(len(TRAINING_CONFIG['percentage_pairs']), device=device)
        total_per_pair = torch.zeros(len(TRAINING_CONFIG['percentage_pairs']), device=device)

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                predicted = (outputs >= 0.6).float()

                for i in range(len(TRAINING_CONFIG['percentage_pairs'])):
                    correct_per_pair[i] += (predicted[:, :, i] == targets[:, :, i]).sum()
                    total_per_pair[i] += targets[:, :, i].numel()

        # Accuracy для каждой пары
        test_acc_per_pair = (correct_per_pair / total_per_pair).cpu().numpy()

        # Вывод результатов
        print(f"Test Loss: {total_loss / len(test_loader.dataset):.4f}")
        for i, (sl, tp) in enumerate(TRAINING_CONFIG['percentage_pairs']):
            print(f"Test Acc (sl={sl:.4f}, tp={tp:.4f}): {test_acc_per_pair[i]:.4f}")

        # Средняя accuracy (опционально)
        mean_test_acc = test_acc_per_pair.mean()
        print(f"Mean Test Accuracy: {mean_test_acc:.4f}")

    def run_training_pipeline(self, csv_path, model_folder) -> None:
        """Запуск пайплайна для обучения или оценки обученной модели"""
        print(f"Подготовка к обучению модели на данных {csv_path}...")
        print("1. Загрузка данных")
        self.load_trading_data_from_csv(csv_path)

        print("2. Насыщение датафрейма фичами")
        self.create_features()

        print("3. Формирование наборов данных для передачи в модель")
        X_train, X_test, y_train, y_test = self.prepare_train_test_split()

        print("4. Обучение модели")
        self.train_model(X_train, X_test, y_train, y_test)

        print(f"5. Сохранение модели в папку {model_folder}")
        self.save_model(model_folder=model_folder)

        print("Готово! Модель обучена и сохранена.")

    def run_evaluating_pipeline(self, csv_path, model_folder) -> None:
        print(f"Подготовка к тестированию обученной модели из папки {model_folder} на данных {csv_path}...")

        print("1. Загрузка модели")
        self.load_model(model_folder=model_folder)

        print("2. Загрузка данных из датафрейма")
        self.load_trading_data_from_csv(csv_path)

        print("3. Насыщение датафрейма фичами")
        self.create_features()

        print("4. Формирование наборов данных для передачи в модель")
        X_test, y_test = self.prepare_validation_data()

        print("5. Проверка обученной модели на других данных")
        self.evaluate_on_new_data(X_test, y_test)

        print("Готово! Модель протестирована.")


if __name__ == "__main__":
    trainer = CryptoModelTrainer()

    model_folder = os.path.join('models',
            f"{TRAINING_CONFIG['timeframe']}m-{TRAINING_CONFIG['backward_window']}backward-{TRAINING_CONFIG['forward_window']}forward-{datetime.now():%d-%m-%Y %H-%M-%S}"
    )

    trainer.run_training_pipeline(csv_path=TRAINING_CONFIG['training_csv_file'],
                                      model_folder=model_folder)
    trainer.run_evaluating_pipeline(csv_path=TRAINING_CONFIG['evaluating_csv_file'],
                                     model_folder=model_folder)