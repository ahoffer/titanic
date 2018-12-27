import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

import helpers

# Load from intermediate_files
training = pd.read_csv('train.csv')
testing = pd.read_csv('test.csv')

helpers.get_title('Crosby, Capt. Edward Gifford')

# Extract target and ID
target = training.pop('Survived')
target.to_pickle('intermediate_files/target.pkl')
trn_psg_id = training.pop('PassengerId')
tst_psg_id = testing.pop('PassengerId')

# Munge data
helpers.pre_pipeline_process(training)
helpers.pre_pipeline_process(testing)

# Save data
training.to_pickle('intermediate_files/pre_pipe_training.pkl')
training.to_pickle('intermediate_files/pre_pipe_testing.pkl')

# Pipeline transforms
numeric_pipeline = Pipeline([
    ('fill_median', SimpleImputer(strategy='median')),
    ('scale', RobustScaler())
])
categorical_pipeline = Pipeline([
    ('missing', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

transformer = ColumnTransformer([
    ('num', numeric_pipeline, ['Pclass', 'SibSp', 'Parch', 'Fare']),
    ('cat', categorical_pipeline, ['Sex', 'Cabin', 'Embarked', 'Title'])
])

# Create encoded/scaled spare matrices
csr_training = transformer.fit_transform(training)
csr_testing = transformer.transform(testing)

# Convert sparse matrices to dataframes with passenger ID for index
df_training = pd.DataFrame(csr_training.toarray(), index=trn_psg_id)
df_testing = pd.DataFrame(csr_testing.toarray(), index=tst_psg_id)

# Save as intermediate_files
df_training.to_pickle('intermediate_files/training.pkl')
df_testing.to_pickle('intermediate_files/testing.pkl')
