# Ex 04 - Amazon SageMaker: train, deploy and evaluate an XGBoost model
# Run these cells one by one in a SageMaker Jupyter notebook (conda_python3 kernel).
# This cannot be run on a normal laptop - it needs an AWS account.

# ---------- Cell 1: libraries, role, region ----------
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from time import gmtime, strftime

role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")
print("Success - the instance is in the " + my_region + " region. Using container: " + xgboost_container)

# ---------- Cell 2: create S3 bucket (change the name to something unique) ----------
bucket_name = 'your-s3-bucket-name'   # <--- CHANGE THIS
s3 = boto3.resource('s3')
try:
    if my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(Bucket=bucket_name,
                         CreateBucketConfiguration={'LocationConstraint': my_region})
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ', e)

# ---------- Cell 3: download data and load into a dataframe ----------
try:
    urllib.request.urlretrieve(
        "https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv",
        "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ', e)

try:
    model_data = pd.read_csv('./bank_clean.csv', index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ', e)

# ---------- Cell 4: shuffle and split (70% train, 30% test) ----------
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729),
                                 [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)

# ---------- Cell 5: format training data and upload to S3 ----------
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1) \
  .to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(
    os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

# ---------- Cell 6: create the XGBoost estimator ----------
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(
    xgboost_container, role, instance_count=1, instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(bucket_name, prefix), sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6,
                        subsample=0.8, silent=0, objective='binary:logistic', num_round=100)

# ---------- Cell 7: train ----------
xgb.fit({'train': s3_input_train})

# ---------- Cell 8: deploy the model to an endpoint ----------
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# ---------- Cell 9: predict on test data ----------
from sagemaker.serializers import CSVSerializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values
xgb_predictor.serializer = CSVSerializer()
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')
predictions_array = np.fromstring(predictions[1:], sep=',')
print(predictions_array.shape)

# ---------- Cell 10: evaluate (confusion matrix) ----------
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array),
                 rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0, 0]; fn = cm.iloc[1, 0]; tp = cm.iloc[1, 1]; fp = cm.iloc[0, 1]
p = (tp + tn) / (tp + tn + fp + fn) * 100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100, tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100, fn, tp/(tp+fp)*100, tp))

# ---------- Cell 11: CLEAN UP (important - avoids AWS charges) ----------
xgb_predictor.delete_endpoint(delete_endpoint_config=True)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
# Then stop and delete the notebook instance from the SageMaker console.
