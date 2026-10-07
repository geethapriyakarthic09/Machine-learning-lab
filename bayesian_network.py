import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pgmpy renamed BayesianModel -> BayesianNetwork -> DiscreteBayesianNetwork in newer versions
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork as BayesianModel
    except ImportError:
        from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

# Step 2-3: load and explore the heart dataset
# NOTE: this folder has a SAMPLE heart.csv (same columns as the Kaggle/UCI heart
# disease dataset). Replace it with the real heart.csv from your lab.
df = pd.read_csv("heart.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())

df.drop(["oldpeak", "slope", "ca", "thal"], axis=1, inplace=True)
print(df.head())
df.info()
print(df.describe())
print(df.columns)

# Step 4: build the Bayesian network
# age, sex, exang (exercise induced angina), cp (chest pain) -> target (heart disease)
# target -> restecg (ECG result), target -> chol (cholesterol)
model = BayesianModel([("age", "target"), ("sex", "target"), ("exang", "target"),
                       ("cp", "target"), ("target", "restecg"), ("target", "chol")])

# learn the conditional probability tables (CPDs) from the data
model.fit(df, estimator=MaximumLikelihoodEstimator)

print(model.get_cpds("age"))
print(model.get_cpds("sex"))
print(model.get_cpds("exang"))
print(model.get_cpds("chol"))

# Extra: inference - probability of heart disease for one patient (optional)
try:
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    row = df.iloc[0]
    q = infer.query(variables=["target"],
                    evidence={"age": int(row["age"]), "sex": int(row["sex"]),
                              "cp": int(row["cp"]), "exang": int(row["exang"])})
    print(q)
except Exception as e:
    print("Inference skipped:", e)
