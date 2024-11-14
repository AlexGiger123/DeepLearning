import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Завантаження та підготовка даних
(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()

# Ресайз і нормалізація зображень
train_imgs = train_imgs.reshape(-1, 28 * 28).astype('float32') / 255
test_imgs = test_imgs.reshape(-1, 28 * 28).astype('float32') / 255

# One-hot кодування міток
train_lbls = tf.keras.utils.to_categorical(train_lbls, 10)
test_lbls = tf.keras.utils.to_categorical(test_lbls, 10)

# Побудова багатошарового перцептрона
model = Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(300, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компіляція нейронної мережі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі з виділенням частини на валідацію
history = model.fit(train_imgs, train_lbls, epochs=10, batch_size=128, validation_split=0.15)

# Оцінка точності моделі на тестових даних
loss, accuracy = model.evaluate(test_imgs, test_lbls)
print(f"Точність на тестових даних: {accuracy:.4f}")

# Прогнозування для тестового набору
preds = model.predict(test_imgs)

# Функція для відображення зображень із реальними та передбаченими мітками
def show_images(images, true_labels, predicted_labels, num=6):
    plt.figure(figsize=(12, 4))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        actual = np.argmax(true_labels[i])
        predicted = np.argmax(predicted_labels[i])
        plt.title(f"True: {actual}\nPred: {predicted}")
        plt.axis('off')
    plt.show()

# Відображення прикладів із тестового набору
show_images(test_imgs, test_lbls, preds)