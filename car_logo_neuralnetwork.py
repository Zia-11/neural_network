import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Определение пути к директориям с данными
train_dir = '/content/drive/MyDrive/Car_Brand_Logos/Train'
test_dir = '/content/drive/MyDrive/Car_Brand_Logos/Test'

# Подготовка данных
# Создание объектов ImageDataGenerator для предварительной обработки и аугментации данных обучения,
# здесь применяется масштабирование пикселей из диапазона [0, 255] в [0, 1].
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Генераторы данных обучения, которые автоматически загружают изображения из заданных директорий (train_dir, test_dir),
# масштабирует их до указанного размера (150x150 пикселей), упаковывает в батчи по 20 изображений и
# классифицирует в категориальный формат (один горячий вектор).
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

# Построение модели
# Создание последовательной модели позволяет модели добавлять слои один за другим.
model = Sequential([
    # Добавление сверточного слоя с 32 фильтрами размером 3x3 и функцией активации ReLU;
    # этот слой будет первым слоем модели, обрабатывающим входные изображения размером 150x150 пикселей с 3 каналами (RGB).
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # Слой максимального пулинга с окном 2x2
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Слой для выравнивания всех признаков в один длинный вектор, который будет использоваться в полносвязных слоях.
    Flatten(),

    # Слой dropout для предотвращения переобучения, исключает случайные 50% соединений во время обучения.
    Dropout(0.5),

    # Полносвязный слой с 512 нейронами и функцией активации ReLU.
    Dense(512, activation='relu'),

    # Выходной полносвязный слой с 8 нейронами, использующий softmax для вывода вероятностей классификации.
    Dense(8, activation='softmax')
])

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Количество шагов за эпоху
    epochs=30,  # Количество эпох
    validation_data=test_generator,
    validation_steps=50)  # Шаги валидации

# Список марок автомобилей в том же порядке, в котором они были в тренировочных данных
class_labels = ['Hyundai', 'Lexus', 'Mazda', 'Mercedes', 'Opel', 'Skoda', 'Toyota', 'Volkswagen']

# Функция для загрузки и предобработки изображения
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

# Путь к изображению
image_path = '/content/Mercedes-Benz-logo-cover.jpg'

# Получение и вывод результата
predicted_brand = predict_car_brand(image_path)
print(f'На изображении представлена марка автомобиля: {predicted_brand}')
