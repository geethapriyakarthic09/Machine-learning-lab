# Ex 06 - Locally Weighted Regression (non-parametric)

**Aim:** Fit data points with Locally Weighted Regression and draw the graph (total bill vs tip).

**How it works:** for every point, nearby points get higher weights (Gaussian kernel, k = 0.5) and a separate weighted linear regression is solved. Joining all predictions gives a smooth curve.

**Run:** `pip install -r requirements.txt` then `python locally_weighted_regression.py`

**Dataset:** `10-dataset.csv` here is a SAMPLE with the same columns (total_bill, tip). Replace with your lab's tips dataset for the real graph. `lwr_plot.png` and `output.txt` are from the sample data.

Note: the manual used `np.mat` which is removed in new NumPy versions; this version uses normal arrays and gives the same idea. `pinv` (pseudo-inverse) is used so isolated points with very small weights do not crash the program.
