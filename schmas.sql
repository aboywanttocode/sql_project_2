import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 2. Load and Preprocess Data
# -------------------------------------------------------
# CIFAR-10 contains 60,000 color images (32x32x3) in 10 classes
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Verify the shape: (32, 32, 3)
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# 3. Build CNN Model
# -------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')   # 10 classes in CIFAR-10
])

# Show architecture
model.summary()

# 4. Compile Model
# -------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train Model
# -------------------------------------------------------
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

# 6. Evaluate Model
# -------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")