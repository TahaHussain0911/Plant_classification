import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Load the CSV file
csv_path = "./archive/train.csv"
df = pd.read_csv(csv_path)

# Define directories
train_dir = "./archive/train"
val_dir = "./archive/val"
test_dir = "./archive/test"

# Image Data Generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Define the Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(30, activation='softmax')  # the output units is dependent on the data sets here i have 30 plants data thats why using 30 
                                            # i-e target.shape must be equal to output.shape
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the model on training | test | validation data sets
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator)

# Save the Model
model.save('plant_classifier.h5')
