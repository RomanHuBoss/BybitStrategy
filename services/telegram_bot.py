import telebot
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import time

# Конфигурация бота
TOKEN = ""
SERVER_URL = "https://trading.web-line.ru/bot"  # Замените на ваш URL сервера

# Инициализация бота
bot = telebot.TeleBot(TOKEN)

# Словарь для хранения активных задач
active_jobs = {}


def check_server(chat_id):
    """Функция для проверки сервера и отправки результата в чат"""
    try:
        # Делаем запрос к серверу
        response = requests.get(SERVER_URL)
        response.raise_for_status()  # Проверяем на ошибки HTTP

        # Отправляем результат в чат
        bot.send_message(
            chat_id,
            f"Результат запроса к серверу ({SERVER_URL}):\n{response.text}"
        )
    except Exception as e:
        bot.send_message(
            chat_id,
            f"Ошибка при запросе к серверу: {str(e)}"
        )


@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Обработчик команды /start"""
    bot.reply_to(
        message,
        f"Привет {message.from_user.first_name}! Я бот, который проверяет сервер каждую минуту."
    )


@bot.message_handler(commands=['start_checking'])
def start_checking(message):
    """Запускает периодическую проверку сервера"""
    chat_id = message.chat.id

    # Останавливаем предыдущую задачу, если она существует
    if chat_id in active_jobs:
        active_jobs[chat_id].remove()
        del active_jobs[chat_id]

    # Создаем планировщик
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        check_server,
        'interval',
        minutes=1,
        args=[chat_id],
        id=str(chat_id)
    )
    scheduler.start()

    # Сохраняем ссылку на планировщик
    active_jobs[chat_id] = scheduler

    bot.reply_to(
        message,
        "Проверка сервера запущена! Запросы будут выполняться каждую минуту."
    )


@bot.message_handler(commands=['stop_checking'])
def stop_checking(message):
    """Останавливает проверку сервера"""
    chat_id = message.chat.id

    if chat_id not in active_jobs:
        bot.reply_to(message, "Нет активных проверок.")
        return

    # Останавливаем планировщик
    active_jobs[chat_id].shutdown()
    del active_jobs[chat_id]

    bot.reply_to(message, "Проверка сервера остановлена.")


# Запуск бота
if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling()