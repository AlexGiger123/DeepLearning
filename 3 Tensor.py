import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Завантаження та попередня обробка даних
(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()
train_imgs = train_imgs[..., np.newaxis].astype('float32') / 255
test_imgs = test_imgs[..., np.newaxis].astype('float32') / 255

# One-hot encoding для міток
train_lbls = tf.keras.utils.to_categorical(train_lbls, num_classes=10)
test_lbls = tf.keras.utils.to_categorical(test_lbls, num_classes=10)

# Визначення моделі LeNet-5
def create_lenet5():
    model = Sequential([
        Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', padding='same', input_shape=(28, 28, 1)),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(filters=120, kernel_size=(5, 5), activation='tanh'),
        Flatten(),
        Dense(units=84, activation='tanh'),
        Dense(units=10, activation='softmax')
    ])
    return model

# Ініціалізація та компіляція моделі
model = create_lenet5()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі
history = model.fit(train_imgs, train_lbls, epochs=10, batch_size=128, validation_split=0.1)

# Оцінка моделі
loss, accuracy = model.evaluate(test_imgs, test_lbls, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Прогноз на тестових даних
predictions = model.predict(test_imgs)

# Функція для відображення зображень
def display_images(images, labels, predictions, n=5):
    plt.figure(figsize=(10, 4))
    for idx in range(n):
        plt.subplot(1, n, idx + 1)
        img = images[idx].reshape(28, 28)
        actual = np.argmax(labels[idx])
        predicted = np.argmax(predictions[idx])
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {actual}\nPred: {predicted}")
        plt.axis('off')
    plt.show()

# Відображення прикладів з тестового набору
display_images(test_imgs, test_lbls, predictions)
