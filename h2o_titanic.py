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
from h2o.grid import H2OGridSearch

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
predictor_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'TicketPrefix',
                   'TicketPostfix']

train.impute()
test.impute()

ss = train.split_frame(ratios=[0.80], seed=42)
train_split = ss[0]
valid_split = ss[1]

ntree_range = range(3, 100, 5)
hyper_params = {'max_depth': [i for i in range(2, 11, 2)],
                'ntrees': [i for i in range(1, 100, 5)],
                }

search_criteria = {'strategy': "RandomDiscrete", 'max_runtime_secs': 120}

grid = H2OGridSearch(model=H2ORandomForestEstimator,
                     hyper_params=hyper_params,
                     search_criteria=search_criteria)

grid.train(predictor_names, response_name_fact, training_frame=train_split, validation_frame=valid_split, seed=42)
# print(grid.summary())
# print(grid.auc(valid=True))
models = grid.get_grid(sort_by='accuracy', decreasing=True)
model = models[0]
print(model.auc(xval=True))

predictions = model.predict(test)

submission = test['PassengerId']
submission['Survived'] = predictions['predict']
h2o.export_file(submission, os.getcwd() + "/submission.csv", force=True)
