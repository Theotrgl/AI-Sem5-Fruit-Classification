"""
This script trains a fruit classifier using the MobileNetV2 architecture.
It uses the TensorFlow library for building and training the machine learning model.
The script loads the fruit images from a directory, preprocesses them using data augmentation techniques,
and splits them into training and validation sets.
The MobileNetV2 model is then constructed by adding additional layers on top of the base model.
The model is compiled with an optimizer, loss function, and evaluation metrics.
Training is performed for a specified number of epochs, with early stopping and tensorboard logging.
The trained model is saved in the specified directory, and a TFLite model is also generated.
"""

# primary library for building and training machine learning
import tensorflow as tf

# efficient numerical operations for arrays and matrices
import numpy as np

# handling filepaths
import os
import pathlib

# regularization technique
from tensorflow.keras import regularizers

# initialize the file paths that will be used
home_path = str(pathlib.Path.home())
file_path = "/.keras/datasets/FinalFruits/"

log_dir = "/Tutorials/AI_FinalProject/logs"
base_dir = home_path + file_path
test_path = home_path + "/.keras/datasets/TestFruits/"
log_absolute_path = home_path + log_dir
saved_model_dir = "/mnt/d/Binus/S5/AI/Final_Project/Datasets/Fruits"

# Set image dimension and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 128

# Data augmentation and preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Preprocessing for test data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory( #generators for training
    test_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Generate training data from directory
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
)

# Generate validation data from directory
val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
)

# Tensorboard callback for logging and visualization
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_absolute_path)

# Get a batch of images and labels from the training generator
for image_batch, label_batch in train_generator:
    break
image_batch.shape, label_batch.shape

print(train_generator.class_indices)

# Save the class labels to a file
labels = "\n".join(sorted(train_generator.class_indices.keys()))
num_classes = len(labels)

labels_model_path = os.path.join(saved_model_dir, "labels.txt")
with open(labels_model_path, "wb") as f:
    f.write(labels.encode("utf-8"))

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Load the MobileNetV2 base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)

base_model.trainable = False

# Construct the final model by adding additional layers on top of the base model

model = tf.keras.Sequential([
    base_model,
    # Adding a convolutional layer with 64 filters of size 3x3.
    # ReLU (Rectified Linear Unit) activation function is applied after the convolution operation.
    # L2 regularization with a penalty of 0.01 is applied to the kernel weights.
    tf.keras.layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    # Dropout layer to regularize the model and prevent overfitting.
    # It randomly sets a fraction of the input units to 0 during training.
    tf.keras.layers.Dropout(0.2),
    # Global Average Pooling 2D layer to convert the 2D feature maps into a 1D feature vector.
        # This reduces the spatial dimensions and retains the most important features.
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(12, activation='softmax')
])


"""
Compile the model with optimizer, loss function, and evaluation metrics

optimizer Adam : Adam is a popular optimizer for models, combining AdaGrad and RMSProp's advantages to minimize loss function and adjust model weights and biases during training.,
loss categorical_crossentropy: Categorical crossentropy is a loss function that is used for single-label classification. This is when only one category is applicable for each data point.,
metrics accuracy: Accuracy is the fraction of predictions our model got right. It is calculated by dividing the number of correct predictions made by the model by the total number of predictions made for each class.,
metrics precision: Precision is the fraction of true positives out of all the predicted positives. It is calculated by dividing the number of true positives by the number of true positives and false positives.,
metrics recall: Recall is the fraction of true positives out of all the actual positives. It is calculated by dividing the number of true positives by the number of true positives and false negatives.
"""

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

epochs = 10  # the number of times the model is exposed to the training set

# Create an EarlyStopping callback to monitor validation loss during training
# early_stopping = tf.keras.callbacks.EarlyStopping(
#   monitor='val_loss',  # The metric to monitor (validation loss)
#   patience=3,   # Number of epochs with no improvement before stopping
#   restore_best_weights=True) # Restore model weights from the epoch with the best validation loss

# Train the model
history = model.fit(
    train_generator,  # training data generator,,
    epochs=epochs,  # the number of times the model is exposed to the training set,
    validation_data=val_generator,  # the data that the model will be validated against
    callbacks=[tensorboard_callback],
)  # callbacks that are applied at the end of each epoch


test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
print(f"Test precision: {test_precision}")
print(f"Test recall: {test_recall}")

# Save the trained model
tf.saved_model.save(model, saved_model_dir)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
# Initialize the path to the TFLite model file
tflite_model_path = os.path.join(saved_model_dir, "modelv4.tflite")
print("TFLite Model Path:", tflite_model_path)
# Write the TFLite model to designated file
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
