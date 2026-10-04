import numpy as np
import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

# Step 1: import the data (downloads LFW faces dataset the first time - needs internet)
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
print(people.data.shape)
print(people.target.shape)
image_shape = people.images[0].shape
print("people.images.shape:", people.images.shape)
print("Number of classes:", len(people.target_names))

# show some faces
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={"xticks": (), "yticks": ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.savefig("sample_faces.png")
plt.close()

# Step 2: transform the data - keep max 50 images per person, scale pixels to 0-1
mask = np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
print(X_train.shape, X_test.shape)

# Baseline: 1-nearest neighbour
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))

# PCA (whitening) to reduce 5655 pixel features to 100 components
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape:", X_train_pca.shape)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("1-nn accuracy after PCA: {:.2f}".format(knn.score(X_test_pca, y_test)))

# Step 3-4: scale and build the neural network (2 hidden layers: 300 and 100 neurons)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ann = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=300, random_state=0)

# Step 5: train and evaluate
ann.fit(X_train_scaled, y_train)
print("ANN train accuracy: {:.2f}".format(ann.score(X_train_scaled, y_train)))
print("ANN test accuracy : {:.2f}".format(ann.score(X_test_scaled, y_test)))

# Step 6: improve the model - train the ANN on the PCA features
ann_pca = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=500, random_state=0)
ann_pca.fit(X_train_pca, y_train)
print("ANN (with PCA) test accuracy: {:.2f}".format(ann_pca.score(X_test_pca, y_test)))
