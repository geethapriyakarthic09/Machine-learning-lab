import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")           # remove this line if you want plots to pop up
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2-3: load and explore dataset (Review, Liked)
# NOTE: this folder has a small SAMPLE Restaurant_Reviews.tsv. Replace it with the
# real 1000-review file from your lab for the real result.
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)
print(df.head())
print(df.shape)

# Step 4: preprocessing (needs internet once, for stopwords)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean(text):
    text = re.sub(pattern="[^a-zA-Z]", repl=" ", string=text)   # remove special characters
    words = text.lower().split()                                # lowercase + tokenize
    words = [w for w in words if w not in stop_words]           # remove stop words
    return " ".join(ps.stem(w) for w in words)                  # stemming

corpus = [clean(r) for r in df["Review"]]
print(corpus[0:10])

# Step 5: bag of words
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Step 6: train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
model = RandomForestClassifier(n_estimators=35, random_state=0)
model.fit(X_train, y_train)
print("Model score:", model.score(X_test, y_test))

# Step 7: predictions and metrics
y_pred = model.predict(X_test)
print("---- Scores")
print("Accuracy score is: {}%".format(round(accuracy_score(y_test, y_pred) * 100, 2)))
print("Precision score is: {}".format(round(precision_score(y_test, y_pred), 2)))
print("Recall score is: {}".format(round(recall_score(y_test, y_pred), 2)))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.savefig("confusion_matrix.png")
plt.close()

# Hyperparameter tuning: find best n_estimators
best_accuracy = 0.0
n_estimators_val = 5
for i in np.arange(1, 30, 2):
    temp = RandomForestClassifier(n_estimators=i, random_state=0)
    temp.fit(X_train, y_train)
    score = accuracy_score(y_test, temp.predict(X_test))
    print("Accuracy score for n_estimators={} is: {}%".format(i, round(score * 100, 2)))
    if score > best_accuracy:
        best_accuracy = score
        n_estimators_val = i
print("The best accuracy is {}% with n_estimators value as {}".format(round(best_accuracy * 100, 2), n_estimators_val))

model = RandomForestClassifier(n_estimators=int(n_estimators_val), random_state=0)
model.fit(X_train, y_train)

def predict_sentiment(sample_review):
    final_review = clean(sample_review)
    temp = cv.transform([final_review]).toarray()
    return model.predict(temp)

for review in ["The food is really good here.",
               "Food was pretty bad and the service was very slow.",
               "The food was absolutely wonderful and nicely presentation"]:
    print(review, "->", "POSITIVE review." if predict_sentiment(review) else "NEGATIVE review!")
