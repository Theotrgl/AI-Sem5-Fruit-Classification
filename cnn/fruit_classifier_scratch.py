import os
import pathlib
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy

CLASSES = ['freshapples', 'freshbanana', 'freshoranges','rottenapples','rottenbanana','rottenoranges']
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_image_and_label(image_path, target_size=(32, 32)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.float32)
    image = tf.image.resize(image, target_size)

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES)  # One-hot encode.
    label = tf.dtypes.cast(label, tf.float32)

    return image, label

def build_network():
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               strides=(1, 1))(input_layer)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)

    x = Flatten()(x)
    x = Dense(units=6)(x)
    output = Softmax()(x)

    model = Model(inputs=input_layer, outputs=output)
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

print("Number of training images:", len(train_dataset_paths))
print("Number of testing images:", len(test_dataset_paths))



BATCH_SIZE = 1024
BUFFER_SIZE = 1024

train_dataset = prepare_dataset(train_dataset_paths,
                                buffer_size=BUFFER_SIZE,
                                batch_size=BATCH_SIZE)
validation_dataset = prepare_dataset(test_dataset_paths,
                                     buffer_size=BUFFER_SIZE,
                                     batch_size=BATCH_SIZE
                                    )

# Use tf.data.Dataset.cardinality to get the number of batches
train_steps = tf.data.experimental.cardinality(train_dataset).numpy()
validation_steps = tf.data.experimental.cardinality(validation_dataset).numpy()

model = build_network()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

EPOCHS = 100
model.fit(train_dataset,
          steps_per_epoch= train_steps,
          epochs=EPOCHS,
          validation_data = validation_dataset,
          validation_steps = validation_steps,
          verbose = 1
         
          )

test_loss, test_accuracy = model.evaluate(validation_dataset, steps = train_steps)
print(f'Loss: {test_loss}, accuracy: {test_accuracy}')