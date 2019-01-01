import os

import h2o
from h2o import H2OFrame
from h2o.estimators import H2ORandomForestEstimator

import helpers

# Connext to h2o server and cleanup objects
h2o.init()
h2o.remove_all()

# Import data
col_types = {'Ticket': 'string', 'Cabin': 'string'}
test = h2o.import_file('test.csv', destination_frame='titanic_test', col_types=col_types)
train = h2o.import_file('train.csv', destination_frame='titanic_train', col_types=col_types)

# NOTE: rbind() does not work unless the columns are in the same order
response_name = 'Survived'
response_name_fact = 'Survived_factor'

# Convert training data to an enumeration for classification instead of regression
train[response_name_fact] = train[response_name].asfactor()
train.pop(response_name)

# Set dummy response var in test data
test[response_name_fact] = None
test[response_name_fact] = test[response_name_fact].asfactor()

# Combine data into one set for simpler processing
all_data = train.rbind(test)
train = None
test = None

# Get train and test indexes
test_idx = all_data[response_name_fact].isna()
train_idx = test_idx.logical_negation()

# Process the data to create additional features
all_data = H2OFrame(helpers.pre_pipeline_process(all_data.as_data_frame()))

# Predict Age
rows_missing_ages = all_data['Age'].isna()
unknown_ages = all_data[rows_missing_ages]
unknown_ages.pop('Age')
known_ages = all_data[rows_missing_ages.logical_negation()]
age_model = H2ORandomForestEstimator(seed=42)
age_model.train(['Title', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch', 'Fare'], 'Age', training_frame=known_ages)
age_prediction = age_model.predict(unknown_ages)
age_join_frame = age_prediction.cbind(unknown_ages['PassengerId'])
all_data = helpers.merge_ages(all_data, age_join_frame)


# Convert training data to an enumeration for classification instead of regression
# Impute will convert he response column from int to real and that would
# pevent H2O from converting the response column into a factor.
all_data[response_name_fact] = all_data[response_name_fact].asfactor()


# Fill in gaps
all_data.impute()


# Split the training data to create some validation data
ss = all_data[train_idx].split_frame(ratios=[0.9], seed=42)
train_split = ss[0]
valid_split = ss[1]

# Select a subset of features for training and prediction
predictor_names = ['Pclass', 'Sex', 'Age', 'Fare', 'SocialPosition', 'Embarked', 'Title', 'LastName']

model = H2ORandomForestEstimator(binomial_double_trees=True, max_depth=7, ntrees=60, balance_classes=True, seed=42,
                                 nfolds=5)
model.train(predictor_names, response_name_fact, training_frame=train_split, validation_frame=valid_split)
predictions = model.predict(all_data[test_idx])

# Create file for Kaggle
submission = all_data[test_idx, 'PassengerId']
submission['Survived'] = predictions['predict']
h2o.export_file(submission, os.getcwd() + "/submission.csv", force=True)
print(model.summary)
print(model.auc())
