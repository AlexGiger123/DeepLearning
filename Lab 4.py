import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Завантаження та нормалізація CIFAR-10
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
train_data, test_data = train_data / 255.0, test_data / 255.0


# Визначення моделі AlexNet для CIFAR-10
def alexnet_model():
    net = Sequential([
        Conv2D(96, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return net


# Ініціалізація та компіляція моделі
model = alexnet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі
train_history = model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels))

# Оцінка моделі
test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=2)
print(f"Точність на тестовому наборі: {test_accuracy * 100:.2f}%")

# Імена класів CIFAR-10
labels_map = ['Птаха', 'Транспорт', 'Риба', 'Хижак', 'Травоїдний', 'Домашній', 'Амфібія', 'Гіппогриф', 'Корабель', 'Вантажівка']


# Функція для відображення результатів передбачення
def show_predictions(images, true_labels, predictions, num_images=5):
    plt.figure(figsize=(12, 5))
    for idx in range(num_images):
        plt.subplot(1, num_images, idx + 1)
        plt.xticks([]);
        plt.yticks([]);
        plt.grid(False)

        img = images[idx]
        plt.imshow(img)
        actual = labels_map[true_labels[idx][0]]
        prediction = labels_map[np.argmax(predictions[idx])]

        plt.xlabel(f"Істинно: {actual}\nПрогноз: {prediction}")
    plt.show()


# Прогнози для зображень із тестового набору
predictions = model.predict(test_data[:5])

# Відображення кількох тестових зображень із прогнозами
show_predictions(test_data, test_labels, predictions)
