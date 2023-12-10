import os
import pathlib
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping


CLASSES = ['freshapples', 'freshbanana', 'freshoranges','rottenapples','rottenbanana','rottenoranges']
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_image_and_label(image_path, target_size=(224, 224)): ##224 is target size for MobileNetV2
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Change channels to 3 for RGB images
    image = tf.image.convert_image_dtype(imcage, np.float32)
    image = tf.image.resize(image, target_size)
    image = preprocess_input(image)  # MobileNetV2 requires specific preprocessing

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES)  # One-hot encode.
    label = tf.dtypes.cast(label, tf.float32)

    return image, label

def build_mobile_netv2(fine_tune_at=100):
    base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    input_layer = Input(shape=(224,224,3))
    x = base_model(input_layer,training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256,activation='relu')(x)
    output = Dense(6,activation='softmax')(x) # Adjust units based on your classes
    model = Model(input_layer,output)
    return model

def prepare_dataset(dataset_path,
                    buffer_size,
                    batch_size,
                    shuffle=True):
    dataset = (tf.data.Dataset
               .from_tensor_slices(dataset_path)
               .map(load_image_and_label,
                    num_parallel_calls=AUTOTUNE))

    if shuffle:
        dataset.shuffle(buffer_size=buffer_size)

    dataset = (dataset
               .batch(batch_size=batch_size)
               .prefetch(buffer_size=buffer_size))

    return dataset

train_file_pattern = str(pathlib.Path.home() / '.keras' / 'datasets' / 'fruits' / 'dataset' / 'train' / '*' / '*.png')
test_file_pattern = str(pathlib.Path.home() / '.keras' / 'datasets' / 'fruits' / 'dataset' / 'test' / '*' / '*.png')


train_dataset_paths = [*glob.glob(train_file_pattern)]
test_dataset_paths = [*glob.glob(test_file_pattern)]

train_paths, test_paths = train_test_split(train_dataset_paths,
                                           test_size=0.2,
                                           random_state=999)
train_paths, val_paths = train_test_split(train_paths,
                                          test_size=0.2,
                                          random_state=999)

BATCH_SIZE = 32 # MobileNetV2 works well with smaller batch sizes
BUFFER_SIZE = 1024

train_dataset = prepare_dataset(train_paths,
                                buffer_size=BUFFER_SIZE,
                                batch_size=BATCH_SIZE)
validation_dataset = prepare_dataset(val_paths,
                                     buffer_size=BUFFER_SIZE,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
test_dataset = prepare_dataset(test_paths,
                               buffer_size=BUFFER_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

model = build_mobile_netv2()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

EPOCHS = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=EPOCHS,
          callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Loss: {test_loss}, accuracy: {test_accuracy}')