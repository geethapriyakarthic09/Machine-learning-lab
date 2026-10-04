import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: import dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
print(iris_data[:5])

# Step 3: split dataset
x_train, x_test, y_train, y_test = train_test_split(
    iris_data, iris_labels, test_size=0.20, random_state=42)

# Step 4: train model
classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train, y_train)

# Step 5: predictions
y_pred = classifier.predict(x_test)
print("accuracy is")
print(classification_report(y_test, y_pred))

# Print both correct and wrong predictions (as asked in the problem statement)
print("Correct and wrong predictions:")
for i in range(len(x_test)):
    status = "CORRECT" if y_test[i] == y_pred[i] else "WRONG"
    print(f"Sample {i}: actual={iris.target_names[y_test[i]]:<11} predicted={iris.target_names[y_pred[i]]:<11} -> {status}")

# Step 6: validation
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(6, 6))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.savefig("confusion_matrix.png")
