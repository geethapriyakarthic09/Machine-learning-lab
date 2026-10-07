import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

LABELS = ["Normal", "Fraud"]

# Step 2: load dataset
# NOTE: this folder has a small SAMPLE creditcard.csv (6000 rows). For the real
# result, download "creditcard.csv" (Kaggle - Credit Card Fraud Detection, 284807 rows)
# and replace this file. Columns: Time, V1..V28, Amount, Class (1 = fraud).
data = pd.read_csv("creditcard.csv", sep=",")

# Step 3: explore
data.info()
print("Any null values:", data.isnull().values.any())

count_classes = data["Class"].value_counts(sort=True)
count_classes.plot(kind="bar", rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.savefig("class_distribution.png")
plt.close()

fraud = data[data["Class"] == 1]
normal = data[data["Class"] == 0]
print(fraud.shape, normal.shape)
print(fraud.Amount.describe())
print(normal.Amount.describe())

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle("Amount per transaction by class")
ax1.hist(fraud.Amount, bins=50)
ax1.set_title("Fraud")
ax2.hist(normal.Amount, bins=50)
ax2.set_title("Normal")
plt.xlabel("Amount ($)")
plt.ylabel("Number of Transactions")
plt.yscale("log")
plt.savefig("amount_histogram.png")
plt.close()

# Take 10% sample (as in the manual) - keeps the program fast on the full dataset
data1 = data.sample(frac=0.1, random_state=1)
print(data1.shape, data.shape)

Fraud = data1[data1["Class"] == 1]
Valid = data1[data1["Class"] == 0]
outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))

# Correlation heatmap
corrmat = data1.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, annot=False, cmap="RdYlGn")
plt.savefig("correlation_heatmap.png")
plt.close()

# Independent and dependent features
columns = [c for c in data1.columns.tolist() if c != "Class"]
target = "Class"
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
print(X.shape)
print(Y.shape)

# Step 4: Isolation Forest and Local Outlier Factor
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X),
                                        contamination=outlier_fraction, random_state=state, verbose=0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm="auto", leaf_size=30,
                                               metric="minkowski", p=2, metric_params=None,
                                               contamination=outlier_fraction),
}

for clf_name, clf in classifiers.items():
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
    else:
        clf.fit(X)
        y_pred = clf.predict(X)
    # reshape: 0 = valid transaction, 1 = fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y, y_pred))
    print("Classification Report :")
    print(classification_report(Y, y_pred))
