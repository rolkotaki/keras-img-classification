import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 20
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMAGES_PATH = "/path/to/holiday_photos"


# creating the training and validation datasets by loading the images from the folder, based on the folder structure
# each subfolder, such as water or other, will be regarded as an image category
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    IMAGES_PATH,
    labels="inferred",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    shuffle=True,
    validation_split=0.2,
    subset="both",  # both training and validation sets
    seed=123,
    label_mode="binary"
    # we have two categories, water and other; in case of more, it can be 'categorical'
)

class_names = np.array(train_ds.class_names)
print(class_names)

# the shape of the datasets
[(image_batch, label_batch)] = train_ds.take(1)
print(image_batch.shape)
print(label_batch.shape)


# standardizing the RGB values to be within the [0-1] range
rescaling_layer = keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescaling_layer(x), y))
val_ds = val_ds.map(lambda x, y: (rescaling_layer(x), y))


# plotting ten images as an example
# for image_batch, label_batch in train_ds.take(1):
#     plt.figure(figsize=(10, 4))
#     plt.subplots_adjust(hspace=0.3)
#     for n in range(10):
#         plt.subplot(2, 5, n+1)
#         plt.imshow(image_batch[n])
#         plt.title(class_names[int(keras.backend.get_value(label_batch[n])[0])])
#         plt.axis('off')
# plt.show()
# exit(0)


# caching the dataset in the memory and prefetching
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


###########################################
#     building and training the model     #
###########################################

# for detailed explanation check the article mentioned in the README file

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32,  # how many different filters should be in the layer
                                           # each filter will be able to detect one pattern in the image
                              kernel_size=(3, 3),  # size of the window that we'll use when creating image tiles
                                                   # from each image; 3 pixels x 3 pixels
                              padding="same",  # if in the last image tile we don't have 3 pixels left,
                                               # we pad it with zeros
                              input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),  # 256 pixels x 256 pixels, 3 RGB channels;
                                                                       # input shape required only for the first layer
                              activation="relu"  # activation function
                              )
          )
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation="sigmoid"))


# compiling the model with the loss function and optimizer; in case of more categories use categorical_crossentropy
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.legacy.Adam(),
    metrics=["acc"]
)

# printing the summary of the model
model.summary()

# training the model
model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    shuffle=True
)

# saving the model as TensorFlow SavedModel format
model.save("saved_model")
