import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Dense, \
    GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)



def build_inception_block(inputs, f1, f3r, f3, f5r, f5, proj):
    branch1 = Conv2D(f1, (1, 1), activation='relu', padding='same')(inputs)

    branch3 = Conv2D(f3r, (1, 1), activation='relu', padding='same')(inputs)
    branch3 = Conv2D(f3, (3, 3), activation='relu', padding='same')(branch3)

    branch5 = Conv2D(f5r, (1, 1), activation='relu', padding='same')(inputs)
    branch5 = Conv2D(f5, (5, 5), activation='relu', padding='same')(branch5)

    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv2D(proj, (1, 1), activation='relu', padding='same')(branch_pool)

    return concatenate([branch1, branch3, branch5, branch_pool])



input_img = Input(shape=(32, 32, 3))
layer = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same')(input_img)
layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)

layer = Conv2D(64, (1, 1), activation='relu', padding='same')(layer)
layer = Conv2D(192, (3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)

layer = build_inception_block(layer, 64, 96, 128, 16, 32, 32)
layer = build_inception_block(layer, 128, 128, 192, 32, 96, 64)
layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)

layer = build_inception_block(layer, 192, 96, 208, 16, 48, 64)
layer = build_inception_block(layer, 160, 112, 224, 24, 64, 64)
layer = build_inception_block(layer, 128, 128, 256, 24, 64, 64)
layer = build_inception_block(layer, 112, 144, 288, 32, 64, 64)
layer = build_inception_block(layer, 256, 160, 320, 32, 128, 128)
layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)

layer = build_inception_block(layer, 256, 160, 320, 32, 128, 128)
layer = build_inception_block(layer, 384, 192, 384, 48, 128, 128)

layer = GlobalAveragePooling2D()(layer)
output = Dense(10, activation='softmax')(layer)

googlenet_model = Model(inputs=input_img, outputs=output)
googlenet_model.summary()


googlenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_history = googlenet_model.fit(train_images, train_labels, epochs=10, batch_size=64,
                                    validation_data=(test_images, test_labels))


loss, accuracy = googlenet_model.evaluate(test_images, test_labels, verbose=0)
print(f"Тестова точність: {accuracy * 100:.2f}%")


preds = np.argmax(googlenet_model.predict(test_images[:12]), axis=1)
true = np.argmax(test_labels[:12], axis=1)

plt.figure(figsize=(12, 6))
for idx in range(12):
    plt.subplot(3, 4, idx + 1)
    plt.imshow(test_images[idx])
    plt.title(f"True: {true[idx]}, Pred: {preds[idx]}")
    plt.axis('off')
plt.show()
