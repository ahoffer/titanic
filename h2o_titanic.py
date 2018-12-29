# Key: A key is an entry in the DKV that maps to an object in H2O.
# Frame: A Frame is a collection of Vec objects. It is a 2D array of elements.
# Vec: A Vec is a collection of Chunk objects. It is a 1D array of elements.
# Chunk: A Chunk holds a fraction of the BigData. It is a 1D array of elements.
# ModelMetrics: A collection of metrics for a given category of model.
# Model: A model is an immutable object having predict and metrics methods.
# Job: A Job is a non-blocking task that performs a finite amount of work.

import os

import h2o
from h2o import H2OFrame
from h2o.estimators import H2ORandomForestEstimator

import helpers

h2o.init()
h2o.remove_all()
train = h2o.import_file('train.csv', destination_frame='titanic_train', col_types={'Ticket': 'string'})
test = h2o.import_file('test.csv', destination_frame='titanic_test', col_types={'Ticket': 'string'})

response_name = 'Survived'
# train = H2OFrame(helpers.pre_pipeline_process_h2o(train.as_data_frame()))
# test = H2OFrame(helpers.pre_pipeline_process_h2o(test.as_data_frame()))
train = H2OFrame(helpers.pre_pipeline_process(train.as_data_frame()))
test = H2OFrame(helpers.pre_pipeline_process(test.as_data_frame()))
combined = train.drop(response_name).rbind(test)

# Predict Age
rows_missing_ages = combined['Age'].isna()
unknown_ages = combined[rows_missing_ages]
unknown_ages.pop('Age')
known_ages = combined[rows_missing_ages.logical_negation()]
age_model = H2ORandomForestEstimator(seed=42)
age_model.train(['Title', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch', 'Fare'], 'Age', training_frame=known_ages)
age_prediction = age_model.predict(unknown_ages)
age_join_frame = age_prediction.cbind(unknown_ages['PassengerId'])
train = helpers.merge_ages(train, age_join_frame)
test = helpers.merge_ages(test, age_join_frame)

train.impute()
test.impute()

# Use classification, not regression
response_name_fact = 'Survived_factor'
train[response_name_fact] = train[response_name].asfactor()
ss = train.split_frame(ratios=[0.9], seed=42)
train_split = ss[0]
valid_split = ss[1]

predictor_names = ['Pclass', 'Sex', 'Age', 'Fare', 'SocialPosition', 'Pclass']
model = H2ORandomForestEstimator(binomial_double_trees=True, max_depth=14, ntrees=45, balance_classes=True, seed=42)
model.train(predictor_names, response_name_fact, training_frame=train_split, validation_frame=valid_split)
predictions = model.predict(test)
submission = test['PassengerId']
submission['Survived'] = predictions['predict']
h2o.export_file(submission, os.getcwd() + "/submission.csv", force=True)
print(model.auc())

