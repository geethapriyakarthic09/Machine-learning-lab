# Classify a single waste image with the trained model.
# Usage:  python predict.py path/to/image.jpg
import sys
import json
import numpy as np
import tensorflow as tf

IMG_SIZE = (128, 128)

if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

model = tf.keras.models.load_model("waste_cnn.keras")
class_names = json.load(open("class_names.json"))

img = tf.keras.utils.load_img(sys.argv[1], target_size=IMG_SIZE)
arr = tf.keras.utils.img_to_array(img)
arr = np.expand_dims(arr, axis=0)

probs = model.predict(arr)[0]
best = int(np.argmax(probs))
print("Predicted class: {}  (confidence {:.1f}%)".format(class_names[best], probs[best] * 100))
