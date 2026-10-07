# Ex 10 - Mini Project: Smart Waste Classification using CNN

**Aim:** Automatically classify waste images (e.g. Organic / Recyclable) using a Convolutional Neural Network, so waste can be segregated correctly.

**Dataset:** Kaggle - "Waste Classification data" (techsash). Download and extract so the folders look like:
```
ex10/dataset/TRAIN/O   (organic)     ex10/dataset/TRAIN/R   (recyclable)
ex10/dataset/TEST/O                  ex10/dataset/TEST/R
```
(The dataset is large, so it is NOT uploaded to GitHub. Any dataset with one folder per class also works.)

**CNN architecture:** Input 128x128 RGB -> data augmentation (flip, rotate, zoom) -> rescale 0-1 -> 3 x [Conv2D + MaxPooling] (32, 64, 128 filters) -> Flatten -> Dense 128 -> Dropout 0.5 -> Dense softmax (one neuron per class). Optimizer: Adam, loss: sparse categorical cross-entropy, 10 epochs.

**Run**
```
pip install -r requirements.txt
python train_cnn.py                      # trains, saves graphs + waste_cnn.keras
python predict.py some_waste_image.jpg   # classify one image
```
Tip: no GPU? Run it in Google Colab (Runtime -> Change runtime type -> GPU), it is much faster.

**Files produced after training:** `sample_images.png`, `training_graph.png`, `confusion_matrix.png`, `class_names.json`, `waste_cnn.keras`

**Output:** NOT generated here (TensorFlow and the dataset were not available on my machine). Run `train_cnn.py` in your lab/Colab and paste the real accuracy output into `output.txt` and add the graph images to this folder.

**Result:** Thus the smart waste classification system using CNN was implemented successfully.
