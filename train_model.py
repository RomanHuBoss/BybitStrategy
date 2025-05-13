import os
import pandas as pd
import numpy as np
import json
import pickle
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from training_config import TRAINING_CONFIG
import warnings

#warnings.filterwarnings('ignore')

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
            self.df.drop(['close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'],
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
        self._analyze_price_movements_on_df()


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
        windows = np.arange(10, TRAINING_CONFIG['backward_window'], 10)

        for window in windows:
            self.df[f'sma_{window}'] = SMAIndicator(close=self.df['close'], window=window).sma_indicator()
            self.df[f'ema_{window}'] = EMAIndicator(close=self.df['close'], window=window).ema_indicator()
            self.df[f'rsi_{window}'] = RSIIndicator(close=self.df['close'], window=window).rsi()

            bb = BollingerBands(close=self.df['close'], window=window)
            self.df[f'bb_bbm_{window}'] = bb.bollinger_mavg()
            self.df[f'bb_bbh_{window}'] = bb.bollinger_hband()
            self.df[f'bb_bbl_{window}'] = bb.bollinger_lband()

        macd = MACD(close=self.df['close'], window_slow=26, window_fast=12, window_sign=9)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()

        self.df['obv'] = OnBalanceVolumeIndicator(
            close=self.df['close'],
            volume=self.df['volume']
        ).on_balance_volume()

        for window in windows:
            self.df[f'obv_ma_{window}'] = self.df['obv'].rolling(window=window).mean()

        self.df = self.df.bfill().ffill().dropna()

        if len(self.df) == 0:
            raise ValueError("После добавления индикаторов DataFrame стал пустым")

    def _analyze_price_movements_on_df(self) -> None:
        """Анализ ценовых движений на внутреннем датафрейме"""
        if len(self.df) < TRAINING_CONFIG['forward_window']:
            raise ValueError(
                f"Недостаточно данных для анализа. Требуется минимум {TRAINING_CONFIG['forward_window']} свечей, доступно {len(self.df)}")

        close_prices = self.df['close'].values
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
                next_prices = close_prices[i + 1:i + TRAINING_CONFIG['forward_window'] + 1]

                # LONG позиция
                long_tp = current_price * (1 + tp_pct)
                long_sl = current_price * (1 - sl_pct)

                for j, price in enumerate(next_prices):
                    if price >= long_tp:
                        self.df.at[i, long_col] = 1
                        break
                    elif price <= long_sl:
                        break

                # SHORT позиция
                short_tp = current_price * (1 - tp_pct)
                short_sl = current_price * (1 + sl_pct)

                for j, price in enumerate(next_prices):
                    if price <= short_tp:
                        self.df.at[i, short_col] = 1
                        break
                    elif price >= short_sl:
                        break

        self.df.to_csv("tmp.csv")


    def prepare_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        X = np.array(X)
        y_long = np.array(y_long)
        y_short = np.array(y_short)

        return train_test_split(
            np.array(X),
            np.stack((y_long, y_short), axis=1),
            test_size=0.2,
            random_state=42
        )

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Создание модели LSTM"""
        model = Sequential([
            Input(shape=input_shape),  # Add this as the first layer
            LSTM(256, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2 * len(TRAINING_CONFIG['percentage_pairs']), activation='sigmoid'),
            Reshape((2, len(TRAINING_CONFIG['percentage_pairs'])))
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> Any:
        """Обучение модели"""
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        callbacks = [
            ModelCheckpoint('best_model.keras', save_best_only=True),
            EarlyStopping(patience=10, restore_best_weights=True)
        ]

        return self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

    def save_model(self, model_folder: str = '', model_name: str = 'model.keras') -> None:
        """Сохранение модели и всех связанных данных"""
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.model.save(os.path.join(model_folder, model_name))
        np.save(os.path.join(model_folder, 'scaler_min.npy'), self.scaler.min_)
        np.save(os.path.join(model_folder, 'scaler_scale.npy'), self.scaler.scale_)

        with open(os.path.join(model_folder, 'training_config.pkl'), 'wb') as f:
            pickle.dump(TRAINING_CONFIG, f)

        with open(os.path.join(model_folder, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)

    def load_model(self, model_folder: str = '', model_name: str = 'model.keras') -> None:
        """Загрузка модели и всех связанных данных"""
        self.model = load_model(os.path.join(model_folder, model_name))
        self.scaler.min_ = np.load(os.path.join(model_folder, 'scaler_min.npy'))
        self.scaler.scale_ = np.load(os.path.join(model_folder, 'scaler_scale.npy'))

        global TRAINING_CONFIG
        with open(os.path.join(model_folder, 'training_config.pkl'), 'rb') as f:
            TRAINING_CONFIG = pickle.load(f)

        with open(os.path.join(model_folder, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)

    def evaluate_on_new_data(self, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> None:

        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def run_pipeline(self, csv_path, mode = "training") -> None:
        """Запуск пайплайна для обучения или оценки обученной модели"""

        if mode == 'training':
            print(f"Подготовка к обучению модели на данных {csv_path}...")
        elif mode == 'evaluating':
            print(f"Подготовка к тестированию обученной модели на данных {csv_path}...")

        print("1. Загрузка данных")
        self.load_trading_data_from_csv(csv_path)

        print("2. Насыщение датафрейма фичами")
        self.create_features()

        print("3. Формирование наборов данных для передачи в модель")
        train_test_split = self.prepare_train_test_split()

        if mode == "training":
            print("4. Обучение модели")
            self.train_model(*train_test_split)

            print("5. Сохранение модели")
            self.save_model(model_folder=os.path.join('models', f"{datetime.now():%d-%m-%Y %H-%M-%S}"))

            print("Готово! Модель обучена и сохранена.")

        elif mode == "evaluating":
            print("4. Проверка обученной модели на других данных")
            self.evaluate_on_new_data(*train_test_split)

            print("Готово! Модель протестирована.")
        else:
            raise ValueError("Неизвестный режим работы")


if __name__ == "__main__":
    trainer = CryptoModelTrainer()
    trainer.run_pipeline(csv_path=TRAINING_CONFIG['training_csv_file'], mode="training")
    trainer.run_pipeline(csv_path=TRAINING_CONFIG['evaluating_csv_file'], mode="evaluating")
