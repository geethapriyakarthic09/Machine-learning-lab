# Ex 04 - Study and Implement Amazon SageMaker

**Aim:** Use Amazon SageMaker to upload data, train, deploy and evaluate an ML model (XGBoost - will a bank customer subscribe to a term deposit?).

**Steps**
1. Create a SageMaker notebook instance (`SageMaker-Tutorial`, type `ml.t2.medium`) with an IAM role that can access S3
2. Prepare the data: create S3 bucket, download `bank_clean.csv`, split 70% train / 30% test, upload train data to S3
3. Train the model: XGBoost estimator on `ml.m4.xlarge`
4. Deploy the model to an endpoint
5. Evaluate with a confusion matrix (manual result: ~90% overall classification rate)
6. Clean up: delete endpoint, S3 bucket, notebook instance

**Code:** `sagemaker_xgboost.py` - paste each "Cell" into the SageMaker Jupyter notebook one by one.

**Output:** this needs an AWS account, so it can't run on a normal laptop. Add screenshots of your SageMaker notebook / console as `output.png`.
