import os
import json
import numpy as np
import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from typing import Dict, Any, List, Optional, Generator

TRAINING_CONFIG = {
    'epochs': 4, # число эпох обучения
    'stop_loss_levels': np.arange(0.001, 0.02, 0.001), # уровни стоп-лосса (x100 - получим % от текущей цены)
    'take_profit_multipliers': np.arange(2, 6, 1), # множители для уровней тейк-профит (x100 - получим % от текущей цены)
    'scaler':  MinMaxScaler(feature_range=(0, 1)),
    'backward_window': 30, # кол-во свечей, предшествующих текущей, для анализа ретроспективы
    'forward_window': 120, # кол-во свечей, следующих за текущей, для прогноза динамики развития цены
}

TRAINING_CONFIG['percentage_pairs'] = [
    (sl, sl * multiplier)
    for sl in TRAINING_CONFIG['stop_loss_levels']
    for multiplier in TRAINING_CONFIG['take_profit_multipliers']
]



class CryptoModelPredictor:
    def __init__(self, model_folder: str):
        """
        Инициализация прогнозировщика с загрузкой модели и связанных данных.

        Args:
            model_folder: Путь к папке с сохраненной моделью и файлами конфигурации.
        """
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.percentage_pairs = None
        self.feature_columns = None
        self.required_raw_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']

        self.load_model(model_folder)

    def load_model(self, model_folder: str) -> None:
        """Загружает модель и все связанные данные из указанной папки"""
        if not os.path.exists(model_folder):
            raise ValueError(f"Папка {model_folder} не существует")

        # Загрузка модели
        model_path = os.path.join(model_folder, 'model.keras')
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели не найден: {model_path}")
        self.model = load_model(model_path)

        # Загрузка scaler
        scaler_min_path = os.path.join(model_folder, 'scaler_min.npy')
        scaler_scale_path = os.path.join(model_folder, 'scaler_scale.npy')
        if not (os.path.exists(scaler_min_path) and os.path.exists(scaler_scale_path)):
            raise ValueError("Не найдены файлы scaler")
        self.scaler.min_ = np.load(scaler_min_path)
        self.scaler.scale_ = np.load(scaler_scale_path)

        # Загрузка percentage_pairs
        pairs_path = os.path.join(model_folder, 'percentage_pairs.npy')
        if not os.path.exists(pairs_path):
            raise ValueError(f"Не найден файл с парами sl/tp: {pairs_path}")
        self.percentage_pairs = np.load(pairs_path, allow_pickle=True)

        # Загрузка feature_columns
        features_path = os.path.join(model_folder, 'feature_columns.json')
        if not os.path.exists(features_path):
            raise ValueError(f"Не найден файл с feature columns: {features_path}")
        with open(features_path, 'r') as f:
            self.feature_columns = json.load(f)

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет временные и технические индикаторы к данным.

        Args:
            data: DataFrame с базовыми колонками: open_time, open, high, low, close, volume

        Returns:
            DataFrame с добавленными фичами
        """
        df = data.copy()

        # 1. Добавляем временные фичи
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['day_of_week'] = df['open_time'].dt.dayofweek
            df['month'] = df['open_time'].dt.month
            df['hour'] = df['open_time'].dt.hour

            # Циклическое кодирование
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            df.drop(['day_of_week', 'month', 'hour'], axis=1, inplace=True)

        # 2. Добавляем технические индикаторы
        windows = np.arange(10, TRAINING_CONFIG['backward_window'], 10)

        for window in windows:
            # Скользящие средние
            df[f'sma_{window}'] = SMAIndicator(close=df['close'], window=window).sma_indicator()
            df[f'ema_{window}'] = EMAIndicator(close=df['close'], window=window).ema_indicator()

            # RSI
            df[f'rsi_{window}'] = RSIIndicator(close=df['close'], window=window).rsi()

            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=window)
            df[f'bb_bbm_{window}'] = bb.bollinger_mavg()
            df[f'bb_bbh_{window}'] = bb.bollinger_hband()
            df[f'bb_bbl_{window}'] = bb.bollinger_lband()

        # MACD
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd().fillna(0)
        df['macd_signal'] = macd.macd_signal().fillna(0)
        df['macd_diff'] = macd.macd_diff().fillna(0)



        # OBV
        df['obv'] = OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()

        for window in windows:
            df[f'obv_ma_{window}'] = df['obv'].rolling(window=window).mean()

        # Удаляем возможные NaN и возвращаем только нужные колонки
        df = df.bfill().ffill()
        if self.feature_columns:
            df = df[self.feature_columns]
        return df.dropna()

    def predict(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Делает прогноз на основе уже обработанных данных с фичами.

        Args:
            processed_data: Словарь с обработанными данными {feature: [values]}

        Returns:
            Словарь с прогнозами для long/short позиций
        """
        if self.model is None:
            raise ValueError("Модель не загружена")

        # Проверяем наличие всех фичей
        missing_features = set(self.feature_columns) - set(processed_data.keys())
        if missing_features:
            raise ValueError(f"Отсутствуют фичи: {missing_features}")

        # Подготавливаем данные для модели
        input_array = np.array([processed_data[feature][-TRAINING_CONFIG['backward_window']:] for feature in self.feature_columns]).T
        scaled_input = self.scaler.transform(input_array)

        # Делаем прогноз
        predictions = self.model.predict(np.array([scaled_input]), verbose=0)

        # Форматируем результат
        result = {'long': {}, 'short': {}}
        for i, (sl, tp) in enumerate(self.percentage_pairs):
            sl_key = f"sl_{sl:.4f}_tp_{tp:.4f}"
            result['long'][sl_key] = float(predictions[0, 0, i])
            result['short'][sl_key] = float(predictions[0, 1, i])

        return result

    def prepare_and_predict(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Полный пайплайн: преобразует сырые данные, добавляет фичи и делает прогноз.

        Args:
            raw_data: Словарь с сырыми данными свечей (должен содержать open_time, open, high, low, close, volume)

        Returns:
            Результат прогноза в формате {'long': {sl_tp: prob}, 'short': {sl_tp: prob}}
        """
        # Проверяем сырые данные
        missing_columns = set(self.required_raw_columns) - set(raw_data.keys())
        if missing_columns:
            raise ValueError(f"В сырых данных отсутствуют колонки: {missing_columns}")

        # Конвертируем в DataFrame
        df = pd.DataFrame(raw_data)

        # Добавляем фичи
        df_with_features = self.add_features(df)

        # Конвертируем обратно в словарь и делаем прогноз
        processed_data = {col: df_with_features[col].values for col in self.feature_columns}
        return self.predict(processed_data)

    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о загруженной модели"""
        return {
            'feature_columns': self.feature_columns,
            'percentage_pairs': [{'sl': float(sl), 'tp': float(tp)}
                                 for sl, tp in self.percentage_pairs],
            'input_shape': self.model.input_shape[1:] if self.model else None,
            'required_raw_columns': self.required_raw_columns
        }

    def get_last_prediction(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Упрощенный метод для получения последнего прогноза (без хранения состояния)

        Args:
            raw_data: Словарь с историей свечей (минимум TRAINING_CONFIG['backward_window'])

        Returns:
            Последний прогноз для всех пар sl/tp
        """
        return self.prepare_and_predict(raw_data)

    def process_csv_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Обрабатывает CSV-файл с историческими данными и возвращает генератор прогнозов для каждой свечи.

        Args:
            file_path: Путь к CSV-файлу с данными

        Yields:
            Словарь с прогнозами для каждой свечи после накопления достаточного количества данных
        """
        # Читаем CSV-файл
        df = pd.read_csv(file_path)
        print(f"\nВсего свечей для обработки {len(df)}")

        # Проверяем наличие необходимых колонок
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"В CSV-файле отсутствуют колонки: {missing_columns}")

        # Конвертируем open_time в int (на случай если в CSV он записан как строка)
        df['open_time'] = df['open_time'].astype(int)

        # Сортируем по времени (на всякий случай)
        df = df.sort_values('open_time')

        # Накопленный буфер данных
        data_buffer = {col: [] for col in required_columns}

        for _, row in df.iterrows():
            # Добавляем новую свечу в буфер
            for col in required_columns:
                data_buffer[col].append(row[col])

            # Удаляем старые данные, если буфер превышает TRAINING_CONFIG['backward_window'] свечей
            for col in required_columns:
                if len(data_buffer[col]) > TRAINING_CONFIG['backward_window']:
                    data_buffer[col] = data_buffer[col][-TRAINING_CONFIG['backward_window']:]

            # Когда накопилось достаточно данных, делаем прогноз
            if len(data_buffer['open_time']) == TRAINING_CONFIG['backward_window']:
                try:
                    predictions = self.prepare_and_predict(data_buffer)
                    yield {
                        'open_time': row['open_time'],
                        'close': row['close'],
                        'predictions': predictions
                    }
                except Exception as e:
                    print(f"Ошибка при обработке свечи {row['open_time']}: {str(e)}")
                    continue


# Инициализация прогнозировщика
predictor = CryptoModelPredictor("models/12-05-2025 04-55-55")

# Получение информации о модели
print("Информация о модели:", predictor.get_model_info())


# Пример 2: Обработка CSV-файла
csv_file = "BTCUSDT_1m_2025-05-12-22-48.csv"
if os.path.exists(csv_file):
    print(f"\nОбработка файла {csv_file}...")

    # Счетчик для ограничения вывода в примере
    counter = 0
    #max_examples = 5

    # результаты обработки
    for result in predictor.process_csv_file(csv_file):
        # Выводим информацию о свече и лучших рекомендациях

        best_long = max(result['predictions']['long'].items(), key=lambda x: x[1])
        best_short = max(result['predictions']['short'].items(), key=lambda x: x[1])

        threshold = 0.65
        if best_long[1] > threshold or best_short[1] > threshold:
            print(f"\nВыводится результат №{counter}")
            print(f"\nСвеча {pd.to_datetime(result['open_time'], unit='ms')} (цена закрытия: {result['close']})")
            if best_long[1] > threshold:
                print(f"Лучшая long стратегия: {best_long[0]} с вероятностью {best_long[1]:.2%}")
                print(result['predictions']['long'])

            if best_short[1] > threshold:
                print(f"Лучшая short стратегия: {best_short[0]} с вероятностью {best_short[1]:.2%}")
                print(result['predictions']['short'])



        counter += 1
else:
    print(f"\nФайл {csv_file} не найден, пример обработки CSV пропущен")