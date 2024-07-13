import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load image data as TensorFlow dataset
data_dir = "data"
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(256, 256))

# Preprocess data (scale pixel values to [0,1])
data = data.map(lambda x, y: (x / 255, y))

# Split dataset into training, validation, and test sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build a CNN model with dropout layers
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(256, 256, 3)
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Dropout layer with 50% dropout rate
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display model summary
model.summary()

# Train the model with dropout regularization
logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(
    train, epochs=50, validation_data=val, callbacks=[tensorboard_callback]
)

# Plot training history (loss and accuracy)
plt.figure()
plt.plot(hist.history["loss"], color="teal", label="loss")
plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate the model on the test set
test_metrics = model.evaluate(test)
print(f"Test Loss: {test_metrics[0]}")
print(f"Test Accuracy: {test_metrics[1]}")

# Test the model with a sample image
sample_image_path = "154006829.jpg"
sample_img = cv2.imread(sample_image_path)
resize_sample_img = cv2.resize(sample_img, (256, 256))

# Preprocess the sample image and make prediction
sample_img_input = np.expand_dims(resize_sample_img / 255.0, axis=0)
prediction = model.predict(sample_img_input)

# Print the predicted class based on the prediction threshold (0.5)
predicted_class = "Sad" if prediction > 0.5 else "Happy"
print(f"Predicted class is {predicted_class}")

# Save the trained model
model.save("imageclassifier_with_dropout.h5")