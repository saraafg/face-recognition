import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from sklearn.utils.class_weight import compute_class_weight
from keras._tf_keras.keras.models import load_model
import cv2

# Paths to training and validation datasets
train_data_path = "Facial_Images\\train"
val_data_path = "Facial_Images\\validation"

# Count total images in the training and validation sets
def count_images(data_path):
    total_images = 0
    for expression in os.listdir(data_path):
        expression_path = os.path.join(data_path, expression)
        print(f"{expression}: {len(os.listdir(expression_path))} images")
        total_images += len(os.listdir(expression_path))
    return total_images

print("Total Images in Training Set:", count_images(train_data_path))
print("Total Images in Validation Set:", count_images(val_data_path))

# Define emotions based on folders in the validation set
emotions = os.listdir(val_data_path)
print("Emotions:", emotions)

# Display a sample image for each expression
i = 1
plt.figure(figsize=(14, 8))  # Adjust figure size as needed
for expression in os.listdir(train_data_path):
    expression_path = os.path.join(train_data_path, expression)
    image_path = os.path.join(expression_path, os.listdir(expression_path)[0])
    image = load_img(image_path)
    
    plt.subplot(1, 7, i)  # Adjust the grid if you have more/less than 7 classes
    plt.imshow(image)
    plt.title(expression)  # Add a title to show the emotion category
    plt.axis('off')
    i += 1

plt.show()  # Ensure this is called to display the images

# Data Augmentation and Generators
train_data_gen = ImageDataGenerator(
    rotation_range=30,        # Increased augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255            # Rescale images for better training stability
)

val_data_gen = ImageDataGenerator(rescale=1./255)

train_dataset = train_data_gen.flow_from_directory(
    train_data_path,
    shuffle=True,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=128
)

val_dataset = val_data_gen.flow_from_directory(
    val_data_path,
    shuffle=False,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=128
)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_dataset.classes),
    y=train_dataset.classes
)
class_weights = dict(enumerate(class_weights))

# Define the CNN Model
model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(48, 48, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (5, 5), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(512, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(7, activation='softmax'))

# model.summary()

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Callbacks for early stopping and saving the best model
# checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# # Train the model with class weights
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=50,
#     callbacks=[checkpoint, early_stopping],
#     class_weight=class_weights,  # Include class weights here
#     verbose=1
# )

# Save the trained model
model.save('trained_emotion_recognition_model.keras')
# Step 1: Load the chosen model
model = load_model('best_model.keras')  # Replace with 'trained_emotion_recognition_model.keras' if you prefer

# Step 2: Load and preprocess the image
image_path = "Facial_Images\\validation\sad\\231.jpg"
image = cv2.imread(image_path)  # Read the image using OpenCV
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale if not already
image = cv2.resize(image, (48, 48))  # Resize the image to 48x48 pixels
image = image.astype('float32') / 255  # Normalize the pixel values to [0, 1]
image = np.expand_dims(image, axis=-1)  # Add the channel dimension (48, 48, 1)
image = np.expand_dims(image, axis=0)  # Add the batch dimension (1, 48, 48, 1)

# Step 3: Make the prediction
output = model.predict(image)
predicted_index = np.argmax(output)
predicted_emotion = emotions[predicted_index]
print(f"Predicted Emotion: {predicted_emotion}")