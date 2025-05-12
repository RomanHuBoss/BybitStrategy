import os
import requests
import zipfile
from datetime import datetime
from urllib.parse import urljoin

def download_and_extract_zip_files(base_url, currency_pair, period, timeframe, start_year, start_month, start_day, output_folder):
    """
    Скачивает и распаковывает ZIP-файлы с указанного месяца и года до текущего.

    :param base_url: Базовый URL сайта с файлами
    :param currency_pair: Код валютной пары (например, 'BTCSTUSDT')
    :period: период, за который информация сохраняется в файлы (daily - ежедневно, monthly - ежемесячно)
    :param timeframe: Таймфрейм (например, '1m')
    :param start_year: Год начала скачивания
    :param start_month: Месяц начала скачивания
    :param output_folder: Папка для сохранения распакованных файлов
    """

    def download_unpack_and_save(zip_url, zip_path, extracted_folder_path, extracted_filename):
        if not os.path.exists(extracted_folder_path):
            os.makedirs(extracted_folder_path, exist_ok=True)

        # Проверяем, не был ли файл уже распакован
        if os.path.exists(os.path.join(extracted_folder_path, extracted_filename)):
            print(f"Файлы {extracted_filename} уже распакован в {extracted_folder_path}, пропускаем.")
            return

        # Пытаемся скачать файл
        print(f"Попытка скачать {zip_url}...")
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()  # Проверяем на ошибки HTTP

            # Сохраняем ZIP-файл
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Пытаемся распаковать
            print(f"Распаковка {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder_path)
                print(f"Файлы успешно распакованы в {extracted_folder_path}")

                # Удаляем ZIP-файл после успешной распаковки
                os.remove(zip_path)
            except zipfile.BadZipFile:
                print(f"Ошибка: {zip_path} поврежден или не является ZIP-файлом")
            except Exception as e:
                print(f"Ошибка при распаковке {zip_path}: {str(e)}")

        except requests.exceptions.RequestException as e:
            print(f"Не удалось скачать файл {zip_url}: {str(e)}")
        except Exception as e:
            print(f"Неожиданная ошибка при обработке {zip_url}: {str(e)}")


    # Создаем папку для распакованных файлов, если её нет
    os.makedirs(output_folder, exist_ok=True)

    # Получаем текущий год и месяц
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day

    # Перебираем все месяцы от начального до текущего
    for year in range(start_year, current_year + 1):
        # Определяем начальный и конечный месяцы для каждого года
        start_m = start_month if year == start_year else 1
        end_m = current_month if year == current_year else 12

        for month in range(start_m, end_m + 1):
            if period == "monthly":
                # Формируем имя файла для ежемесячной статистики
                # https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-03.zip

                zip_internal_folder = f"/{period}/klines/{currency_pair}/{timeframe}/"
                extracted_filename = zip_filename = f"{currency_pair}-{timeframe}-{year}-{month:02d}.zip"
                zip_url = base_url + zip_internal_folder + zip_filename
                zip_path = os.path.join(output_folder, currency_pair, zip_filename)
                extracted_folder_path = os.path.join(output_folder, currency_pair, period)
                download_unpack_and_save(zip_url, zip_path, extracted_folder_path, extracted_filename)
            else:
                # Формируем имя файла для ежедневной статистики
                #https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-05-07.zip

                start_d = start_day if year == start_year and month == start_month else 1
                end_d = current_day if year == current_year and month == current_month else 31

                for day in range(start_d, end_d):
                    zip_internal_folder = f"/{period}/klines/{currency_pair}/{timeframe}/"
                    extracted_filename = zip_filename = f"{currency_pair}-{timeframe}-{year}-{month:02d}-{day:02d}.zip"
                    zip_url = base_url + zip_internal_folder + zip_filename
                    zip_path = os.path.join(output_folder, currency_pair, zip_filename)
                    extracted_folder_path = os.path.join(output_folder, currency_pair, period)
                    download_unpack_and_save(zip_url, zip_path, extracted_folder_path, extracted_filename)

# Пример использования
if __name__ == "__main__":
    #https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-05-07.zip
    import os
    BASE_URL = "https://data.binance.vision/data/futures/um"  # Замените на реальный URL
    PERIOD = "daily" # daily or monthly
    CURRENCY_PAIR = "BTCUSDT"
    TIMEFRAME = "1m"
    START_YEAR = 2023
    START_MONTH = 1
    START_DAY = 1
    OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "historical_data")

    download_and_extract_zip_files(
        base_url=BASE_URL,
        currency_pair=CURRENCY_PAIR,
        period=PERIOD,
        timeframe=TIMEFRAME,
        start_year=START_YEAR,
        start_month=START_MONTH,
        start_day=START_DAY,
        output_folder=OUTPUT_FOLDER
    )
