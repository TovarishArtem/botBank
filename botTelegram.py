import telebot
from telebot import types

from chatBotModel import ChatbotModel

# Ваш токен от BotFather
TOKEN = '7018272532:AAGO7yA5SU6lyum1lv4P8fg7Sm4e9_x0fGU'

# Инициализация модели
chatbot_model = ChatbotModel()

# Создание экземпляра бота
bot = telebot.TeleBot(TOKEN)

# Обработка команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я чат-бот. Как я могу помочь вам сегодня?")

# Обработка команды /help
@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, "Введите сообщение, и я постараюсь вам помочь.")

# Обработка всех текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_message = message.text
    response = chatbot_model.generate_response(user_message)
    bot.reply_to(message, response)

# Запуск бота
if __name__ == '__main__':
    bot.polling(none_stop=True)