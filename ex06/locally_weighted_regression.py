import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you want the plot to pop up
import matplotlib.pyplot as plt

# Step 2: load dataset (columns: total_bill, tip)
# NOTE: this folder has a SAMPLE 10-dataset.csv. For the real result use the
# "tips" dataset csv given in your lab (same column names).
data = pd.read_csv("10-dataset.csv")
bill = np.array(data.total_bill)
tip = np.array(data.tip)

# Step 3: weight matrix functions
def kernel(point, X, k):
    m = X.shape[0]
    weights = np.eye(m)
    for j in range(m):
        diff = point - X[j]
        weights[j, j] = np.exp(diff @ diff.T / (-2.0 * k ** 2))
    return weights

def local_weight(point, X, y, k):
    wei = kernel(point, X, k)
    return np.linalg.pinv(X.T @ (wei @ X)) @ (X.T @ (wei @ y))

def local_weight_regression(X, y, k):
    m = X.shape[0]
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = X[i] @ local_weight(X[i], X, y, k)
    return ypred

# add a column of ones (intercept) to the feature
X = np.column_stack((np.ones(len(bill)), bill))
ypred = local_weight_regression(X, tip, 0.5)

# sort by bill so the curve is drawn left to right
order = X[:, 1].argsort()
xsort = X[order]

# Step 4: visualize
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color="green")
ax.plot(xsort[:, 1], ypred[order], color="red", linewidth=3)
plt.xlabel("Total bill")
plt.ylabel("Tip")
plt.savefig("lwr_plot.png")
print("Locally weighted regression done. Plot saved as lwr_plot.png")
print("First 5 predicted tips:", np.round(ypred[:5], 2))
