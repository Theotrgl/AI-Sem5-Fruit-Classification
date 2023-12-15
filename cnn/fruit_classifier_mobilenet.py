import tensorflow as tf
import numpy as np
import os
import pathlib
import glob

home_path = str(pathlib.Path.home())
file_path = '/.keras/datasets/FinalFruits/'
base_dir = home_path + file_path
saved_model_dir = '/mnt/d/Binus/S5/AI/Final_Project/Datasets/Fruits'

IMAGE_SIZE = 224
BATCH_SIZE = 64


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)
    #increase variety of trainings to generalize the model
    # rotation_range=0.2,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

log_dir = os.path.join('Tutorials/AI_FinalProject','logs')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape

print(train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))



labels_model_path = os.path.join(saved_model_dir,'labels.txt')
with open(labels_model_path, 'wb') as f:  
  f.write(labels.encode('utf-8'))

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
      base_model,
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(12, activation='softmax')
])



model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

epochs = 10

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_generator, 
                    epochs=epochs, 
                    validation_data=val_generator)
                    # callbacks=[early_stopping])


tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
tflite_model_path = os.path.join(saved_model_dir,'modelv2.tflite')
print("TFLite Model Path:", tflite_model_path)

with open(tflite_model_path, 'wb') as f:
  f.write(tflite_model)