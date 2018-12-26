# Key: A key is an entry in the DKV that maps to an object in H2O.
# Frame: A Frame is a collection of Vec objects. It is a 2D array of elements.
# Vec: A Vec is a collection of Chunk objects. It is a 1D array of elements.
# Chunk: A Chunk holds a fraction of the BigData. It is a 1D array of elements.
# ModelMetrics: A collection of metrics for a given category of model.
# Model: A model is an immutable object having predict and metrics methods.
# Job: A Job is a non-blocking task that performs a finite amount of work.

import h2o
import os
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
from h2o.grid import H2OGridSearch

h2o.init()
h2o.remove_all()
train = h2o.import_file('train.csv', destination_frame='titanic_train')
test = h2o.import_file('test.csv', destination_frame='titanic_test')
# train = h2o.get_frame('titanic_test')
# test = h2o.get_frame("test")

# predictor_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
predictor_names = ['Pclass', 'Sex', 'Age']
response_name = 'Survived'
response_name_fact = 'Survived_factor'
train[response_name_fact] = train[response_name].asfactor()

gb_params = {'learn_rate': [0.01, 0.05, 0.1, .5],
             'max_depth': [2, 3, 5, 8],
             'ntrees': [5, 10, 20, 50, 100],
             'balance_classes': [True, False],
             'seed': 42}

gb_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,
                        grid_id='gb_grid',
                        hyper_params=gb_params)

gb_grid.train(predictor_names,
              response_name_fact,
              training_frame=train,
             )

gb_grid_results = gb_grid.get_grid(sort_by='accuracy', decreasing=True)

best_gb_model = gb_grid_results[0]

best_gb_model.model_performance(valid=True)

gb_predictions = best_gb_model.predict(test)
gb_submission = test['PassengerId']
gb_submission['Survived'] =  gb_predictions['predict']
h2o.export_file(gb_submission, os.getcwd() + "/gb_submission.csv", force=True)