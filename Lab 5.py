import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Завантаження та нормалізація CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Перетворення міток у формат one-hot
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, 10)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)

# Ініціалізація архітектури VGG-13
def build_vgg13():
    cnn_model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return cnn_model

# Створення та компіляція моделі
model = build_vgg13()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі на 10 епохах
history = model.fit(train_images, train_labels_one_hot, epochs=10, batch_size=64, validation_data=(test_images, test_labels_one_hot))

# Оцінка моделі на тестовому наборі
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot, verbose=2)
print(f"Точність на тестовому наборі: {test_acc * 100:.2f}%")

# Альтернативні назви класів для CIFAR-10
alternative_class_names = ['Птах', 'Спорткар', 'Летюча риба', 'Кіт', 'Лань', 'Пес', 'Жаба', 'Дракон', 'Вітрильник', 'Камаз']

# Функція відображення передбачень
def show_sample_predictions(num_samples=5):
    sample_indices = np.random.choice(len(test_images), num_samples, replace=False)
    sample_images = test_images[sample_indices]
    sample_true_labels = test_labels[sample_indices]
    sample_predictions = model.predict(sample_images)

    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i])
        true_class = alternative_class_names[np.argmax(sample_true_labels[i])]
        predicted_class = alternative_class_names[np.argmax(sample_predictions[i])]
        plt.title(f"Істинно: {true_class}\nПрогноз: {predicted_class}")
        plt.axis('off')
    plt.show()

show_sample_predictions()
