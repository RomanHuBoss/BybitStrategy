import os
import pandas as pd

# Укажите путь к папке с CSV-файлами
folder_path = 'C:\\Users\\roman\\PycharmProjects\\Bybit\\historical_data\\BTCUSDT\\monthly'  # Замените на ваш путь

# Получаем список всех CSV-файлов в папке
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Проверяем, что файлы найдены
if not csv_files:
    print("В указанной папке нет CSV-файлов.")
else:
    # Создаём пустой DataFrame для хранения объединённых данных
    combined_data = pd.DataFrame()

    # Читаем и объединяем все CSV-файлы
    for file in csv_files:
        if file == 'combined_csv.csv':
            continue
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Сохраняем объединённый файл
    output_path = os.path.join(folder_path, 'combined_csv.csv')
    combined_data.to_csv(output_path, index=False)
    print(f"Объединённый файл сохранён как: {output_path}")