# -*- coding: utf-8 -*-
"""BrainTumorClassification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/124aOxCAZMPK-NvbX7kSuPowl2hgVFdEL
"""

# Install necessary libraries
!pip install pillow numpy seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from google.colab import drive
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import array_to_img

# Mount Google Drive to save the trained model
drive.mount('/content/drive')

# Define paths to dataset on Google Drive
train_dir = '/content/drive/MyDrive/Training'
validation_dir = '/content/drive/MyDrive/Testing'

# Set up image data generators with advanced augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load VGG16 model for transfer learning
base_model = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

# Custom CNN model with VGG16 as base
def create_transfer_learning_model():
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_transfer_learning_model()

# Set up callbacks for early stopping and model checkpointing
model_save_path = '/content/drive/MyDrive/Saved/brain_tumor_model_vgg16.keras'  # Change .h5 to .keras
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Test Loss: {loss:0.5f}")
print(f"Test Accuracy: {accuracy:0.5f}")

# Plot training history
_, ax = plt.subplots(ncols=2, figsize=(15, 6))
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Model Accuracy')
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Model Loss')
ax[1].legend()
plt.show()

# Confusion matrix
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification report
class_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualization of predictions
def plot_sample_predictions(model, test_generator, class_indices, num_samples=9, figsize=(13, 12)):
    batch_images, batch_labels = next(test_generator)
    plt.figure(figsize=figsize)
    for i in range(num_samples):
        image, label = batch_images[i], batch_labels[i]
        pred = model.predict(np.expand_dims(image, axis=0))
        pred_label = np.argmax(pred, axis=1)[0]
        plt.subplot(3, 3, i + 1)
        plt.imshow(array_to_img(image))
        plt.title(f"Pred: {class_names[pred_label]} | True: {class_names[np.argmax(label)]}")
        plt.axis("off")
    plt.show()

# Visualize misclassified images
def visualize_misclassified_images(model, test_generator):
    predictions = model.predict(test_generator)
    misclassified_indices = np.where(predictions != test_generator.labels)[0]
    for i in misclassified_indices:
        image, label = test_generator[i]
        pred_label = np.argmax(model.predict(image))
        if pred_label != np.argmax(label):
            plt.imshow(array_to_img(image))
            plt.title(f"Predicted: {class_names[pred_label]}, True: {class_names[np.argmax(label)]}")
            plt.show()

plot_sample_predictions(model, validation_generator, class_indices=class_names)
visualize_misclassified_images(model, validation_generator)