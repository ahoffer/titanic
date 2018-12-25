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

# Load files generated by load_data script
target = pd.read_pickle('target.pkl')
training = pd.read_pickle('pre_pipe_training.pkl')
testing = pd.read_pickle('pre_pipe_testing.pkl')


# Separate the instances with age values into a training set
x = training.Age.isnull()
# and put the instances with missing ages into another set
v = training.loc[x, :]
z = training.loc[~x, :]
# Restore passenger ID so everything can be joined
# all_training = pd.concat(passenger_id,training)

##### WORK IN PROGRESS ######
exit(0)