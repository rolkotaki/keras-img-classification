import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMAGES_PATH = "/path/to/holiday_photos"
CLASS_LABELS = ["other", "water"]


# loading the model
model = keras.models.load_model("saved_model")

# plotting twenty photos with the model's predictions
plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
n = 0
for image_file in os.listdir(IMAGES_PATH):
    if image_file.split('.')[-1].lower() != 'jpg':
        continue
    img = keras.preprocessing.image.load_img(os.path.join(IMAGES_PATH, image_file), target_size=(IMG_WIDTH, IMG_HEIGHT))
    image_to_test = keras.preprocessing.image.img_to_array(img)
    results = model.predict(np.expand_dims(image_to_test, axis=0))
    result = results[0]
    class_label = CLASS_LABELS[int(result)]
    plt.subplot(5, 5, n+1)
    plt.imshow(image_to_test/255)
    plt.title(class_label)
    plt.axis('off')
    n += 1
plt.show()
