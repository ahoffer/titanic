# Key: A key is an entry in the DKV that maps to an object in H2O.
# Frame: A Frame is a collection of Vec objects. It is a 2D array of elements.
# Vec: A Vec is a collection of Chunk objects. It is a 1D array of elements.
# Chunk: A Chunk holds a fraction of the BigData. It is a 1D array of elements.
# ModelMetrics: A collection of metrics for a given category of model.
# Model: A model is an immutable object having predict and metrics methods.
# Job: A Job is a non-blocking task that performs a finite amount of work.

import os

import h2o
from h2o.estimators import H2ORandomForestEstimator

h2o.init()
h2o.remove_all()
train = h2o.import_file('train.csv', destination_frame='titanic_train')
test = h2o.import_file('test.csv', destination_frame='titanic_test')
# train = h2o.get_frame('titanic_test')
# test = h2o.get_frame("test")

predictor_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# predictor_names = ['Pclass', 'Sex', 'Age']
response_name = 'Survived'
response_name_fact = 'Survived_factor'
train[response_name_fact] = train[response_name].asfactor()

train.impute()
test.impute()

rf_model = H2ORandomForestEstimator()
rf_model.train(predictor_names, response_name_fact, training_frame=train)
auc=rf_model.auc(train=True)
rf_predictions = rf_model.predict(test)
rf_submission = test['PassengerId']
rf_submission['Survived'] = rf_predictions['predict']
h2o.export_file(rf_submission, os.getcwd() + "/rf_submission.csv", force=True)
