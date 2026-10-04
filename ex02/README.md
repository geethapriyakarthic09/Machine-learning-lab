# Ex 02 - Spam Mail Detection using Support Vector Machine

**Aim:** Detect spam messages using SVM.

**Steps:** load dataset -> explore -> split X (message) / y (category) -> train/test split (80/20) -> CountVectorizer (text to numbers) -> SVC (rbf kernel) -> accuracy.

**Run:** `pip install -r requirements.txt` then `python spam_svm.py`

**Dataset:** `spam.csv` here is a small SAMPLE (100 messages) so the program runs. For the real result, replace it with the Kaggle "SMS Spam Collection" spam.csv (same columns: Category, Message). Accuracy in `output.txt` is from the sample file.

**Note:** the lab manual had X and y swapped (X = Category, y = Message); fixed here.
