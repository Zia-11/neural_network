import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
import nest_asyncio
import asyncio

nest_asyncio.apply()
# Загрузка обученной модели
model = tf.keras.models.load_model("car_brand_model.h5")
print("Модель загружена успешно")

# Список марок автомобилей в том же порядке, в котором они были в тренировочных данных
class_labels = ['Hyundai', 'Lexus', 'Mazda', 'Mercedes', 'Opel', 'Skoda', 'Toyota', 'Volkswagen']

# Создание временной директории
temp_dir = 'temp_images'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Функция для предобработки изображения
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # Изменение размера изображения
    img = img_to_array(img)  # Преобразование изображения в массив
    img = img / 255.0  # Нормализация пикселей
    img = img.reshape(1, 150, 150, 3)  # Изменение формы массива для сети
    return img

# Функция для предсказания марки автомобиля
def predict_car_brand(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = class_labels[prediction.argmax()]  # Получение метки с наибольшим значением предсказания
    return predicted_class

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Отправь мне изображение автомобиля, и я скажу, какой это бренд.')

# Обработка изображения
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await update.message.photo[-1].get_file()
    image_path = os.path.join(temp_dir, f"{photo_file.file_id}.jpg")
    await photo_file.download_to_drive(image_path)

    # Получение предсказания
    predicted_brand = predict_car_brand(image_path)
    await update.message.reply_text(f'Марка автомобиля: {predicted_brand}')

    # Удаление временного файла
    os.remove(image_path)

# Основная функция
async def main() -> None:
    TOKEN = '7000901038:AAFWV2FFrCXGti-pwy3PuCyItWWeXnTyEtQ'
    application = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Запуск бота
    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
