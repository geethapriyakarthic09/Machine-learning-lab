"""
Mini Project - Smart Waste Classification using CNN
Classifies a waste image (e.g. Organic / Recyclable) with a Convolutional Neural Network.

Expected folder layout (one sub-folder per class, folder name = class name):
    dataset/TRAIN/O/...   dataset/TRAIN/R/...
    dataset/TEST/O/...    dataset/TEST/R/...
(Kaggle "Waste Classification data" by techsash uses O = Organic, R = Recyclable.
 Any dataset with class sub-folders works, e.g. TrashNet's 6 classes.)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 10
TRAIN_DIR = "dataset/TRAIN"
TEST_DIR = "dataset/TEST"

# Step 1: load images from folders (80% train / 20% validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Step 2: show a few sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig("sample_images.png")
plt.close()

# Step 3: build the CNN
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),                       # pixel 0-255 -> 0-1
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),                               # reduces overfitting
    layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Step 4: train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Step 5: accuracy / loss graphs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.savefig("training_graph.png")
plt.close()

# Step 6: evaluate on the test set
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# Step 7: save the trained model
model.save("waste_cnn.keras")
print("Model saved as waste_cnn.keras")
