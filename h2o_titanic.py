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
# train = h2o.get_frame('titanic_test')
# test = h2o.get_frame("test")

# predictor_names = ['Pclass', 'Sex', 'Age']
train = H2OFrame(helpers.pre_pipeline_process_h2o(train.as_data_frame()))
test = H2OFrame(helpers.pre_pipeline_process_h2o(test.as_data_frame()))
response_name = 'Survived'
response_name_fact = 'Survived_factor'
train[response_name_fact] = train[response_name].asfactor()
# predictor_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'TicketPrefix',
#                    'TicketPostfix']
predictor_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'TicketPrefix', 'TicketPostfix']

train.impute()
test.impute()

ss = train.split_frame(ratios=[0.80], seed=42)
train_split = ss[0]
valid_split = ss[1]

model = H2ORandomForestEstimator(binomial_double_trees=True, max_depth=10, ntrees=30, seed=42)
model.train(predictor_names, response_name_fact, training_frame=train_split, validation_frame=valid_split)
print(model.auc(valid=True))

predictions = model.predict(test)

submission = test['PassengerId']
submission['Survived'] = predictions['predict']
h2o.export_file(submission, os.getcwd() + "/submission.csv", force=True)
