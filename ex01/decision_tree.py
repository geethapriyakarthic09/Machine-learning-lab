import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Step 2: load Iris dataset
iris_df = pd.read_csv("Iris.csv")
print(iris_df.head())

# Step 3: explore
print(iris_df.isnull().any())
print(iris_df.dtypes)
print(iris_df.describe())

# Step 4: pair plot
sns.pairplot(iris_df.drop(columns="Id"), hue="Species")
plt.savefig("pairplot.png")
plt.close()

# Step 5: train/test split
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = iris_df[features].values
y = iris_df["Species"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=1)

# Step 6: train decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Step 7: test
prediction = clf.predict(X_test)
print(prediction)
print(classification_report(y_test, prediction))

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.savefig("confusion_matrix.png")
plt.close()

# Step 8: visualize tree
plt.figure(figsize=(14, 8))
tree.plot_tree(clf, feature_names=features, class_names=clf.classes_, filled=True, rounded=True)
plt.savefig("decision_tree.png")
plt.close()

# Step 9: new sample
x = [[4.8, 2.9, 1.3, 0.2]]
res = clf.predict(x)
print("The class predicted is --> " + str(*res))
