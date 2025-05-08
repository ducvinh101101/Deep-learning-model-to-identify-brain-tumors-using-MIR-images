import cv2
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from keras.api.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import tensorflow as tf

# Hàm tạo bộ lọc Gabor
def gabor_filter(img, ksize=5):
    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = np.stack([img.squeeze()] * 3, axis=-1)

    img = img.astype(np.float32)
    filtered_imgs = [cv2.filter2D(img[:, :, i], cv2.CV_32F, k) for i in range(3) for k in gabor_kernels]
    return np.mean(filtered_imgs, axis=0)

# Hàm tiền xử lý ảnh
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img).astype(np.float32)
    img_array = gabor_filter(img_array)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=-1)

# Denoising AutoEncoder
def build_denoising_autoencoder():
    input_img = Input(shape=(128, 128, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Load dữ liệu ảnh cho AutoEncoder
Train_dir = "Data/Training"
Test_dir = "Data/Test"
image_paths = []
for subdir, _, files in os.walk(Train_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_paths.append(os.path.join(subdir, file))

x_train = np.array([preprocess_image(img_path) for img_path in image_paths])

# Huấn luyện AutoEncoder
autoencoder = build_denoising_autoencoder()
autoencoder.fit(x_train, x_train, epochs=1, batch_size=32, verbose=1)

# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

# Dataloader cho VGG16
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(Train_dir, target_size=(128, 128), batch_size=64, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(Test_dir, target_size=(128, 128), batch_size=64, class_mode='categorical', shuffle=False)

# Xây dựng model kết hợp
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Huấn luyện mô hình
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
H = model.fit(train_generator, epochs=200, validation_data=test_generator, verbose=1)

# Vẽ biểu đồ huấn luyện
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(H.history['loss'], label='Train Loss')
plt.plot(H.history['val_loss'], label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(H.history['accuracy'], label='Train Accuracy')
plt.plot(H.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
model.save("last.h5")
# Ma trận nhầm lẫn
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
cm = confusion_matrix(y_true, y_pred)
labels = list(train_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Báo cáo phân loại
report = classification_report(y_true, y_pred, target_names=labels)
print(report)
