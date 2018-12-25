import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

# Set seed to make experiments repeatable
seed = 42
random.seed(seed)
np.random.seed(seed)

# Load intermediate_files generated by load_data script
target = pd.read_pickle('intermediate_files/target.pkl')
training = pd.read_pickle('intermediate_files/training.pkl')
testing = pd.read_pickle('intermediate_files/testing.pkl')

# Create model and predictions
model = RandomForestClassifier(n_estimators=20, n_jobs=8)
# rfe = RFE(model, n_features_to_select=4, verbose=1)
# rfe.fit(training, target)
# training_best_features = training.iloc[:, rfe.support_]
# testing_best_features = testing.iloc[:, rfe.support_]

model.fit(training, target)
predication = model.predict(testing)

# model.fit(training_best_features, target)
# predication = model.predict(testing_best_features)

# Save predictions in to uploadable file
predication = pd.DataFrame(predication, index=testing.index.values)
predication.to_csv('predictions.csv', index_label='PassengerId', header=['Survived'])

scores = cross_val_score(model, training, target, cv=5)
print(sum(scores) / len(scores))