import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Step 2: load dataset (columns: Category, Message)
# NOTE: this folder has a small sample spam.csv. For the full dataset,
# download the SMS Spam Collection "spam.csv" from Kaggle and replace this file.
dataset = pd.read_csv("spam.csv")

# Step 3: explore
dataset.info()
print(dataset.head())

# Step 4: X = message text, y = label (spam/ham)
X = dataset["Message"].values
y = dataset["Category"].values

sns.countplot(x="Category", data=dataset)
plt.savefig("countplot.png")
plt.close()

# Step 5: split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: text -> numbers
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Step 7: SVM
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

# Step 8: accuracy
print("Accuracy:", classifier.score(X_test, y_test))

# Try a new message
msg = ["Congratulations! You won a free prize, claim now"]
print("New message prediction:", classifier.predict(cv.transform(msg))[0])
