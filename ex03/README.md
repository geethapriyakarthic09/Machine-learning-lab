# Ex 03 - Facial Recognition using Artificial Neural Network

**Aim:** Recognise faces from the LFW (Labeled Faces in the Wild) dataset using an ANN.

**Steps:** import data -> transform (max 50 images/person, scale 0-1) -> baseline 1-NN -> PCA (100 components) -> build ANN (hidden layers 300 and 100) -> train & evaluate -> improve with PCA features.

**Run:** `pip install -r requirements.txt` then `python face_recognition_ann.py`
(needs internet the first time, to download the LFW dataset ~200 MB)

**Output:** run it and paste your real accuracy values in `output.txt`. In the lab manual the 1-NN accuracy was 0.23 and 0.31 with PCA.

Note: the manual used old `tf.estimator` / `np.bool` code which fails on new TensorFlow/NumPy, and set `n_classes=10` although the dataset has 62 people. This version uses scikit-learn's `MLPClassifier` (a neural network) so it runs without TensorFlow.
