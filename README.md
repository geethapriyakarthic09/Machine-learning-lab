# Ex 07 - Sentiment Analysis using Random Forest

**Aim:** Classify restaurant reviews as positive or negative using a Random Forest classifier.

**Steps:** load reviews -> clean text (remove symbols, lowercase, remove stop words, stemming) -> Bag of Words (CountVectorizer, 1500 features) -> train/test split (80/20) -> RandomForest (35 trees) -> accuracy/precision/recall -> confusion matrix -> tune `n_estimators` (1 to 29) -> predict new reviews.

**Run:** `pip install -r requirements.txt` then `python sentiment_random_forest.py` (needs internet once to download NLTK stopwords)

**Dataset:** `Restaurant_Reviews.tsv` here is a small SAMPLE (200 reviews, same columns: Review, Liked). Replace it with the 1000-review file from your lab for the real result.

**Output:** `output.txt` was produced with the sample data (and a simple stand-in for the NLTK stop-words/stemmer, since this machine had no internet). Accuracy is 100% only because the sample is tiny and repetitive. Run it on your system and paste your real output.

Note: the manual's loop rebuilt the stop-word set for every word (very slow); here it is built once. `random_state` is set so results repeat.
