from scipy import sparse
import re
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler


def lex_ticket(ticket):
    # Get everything before the slash
    # Get everything after the slash but before the first number
    # Get all the remaining digits
    stripped = ticket.replace('.', '').upper().lstrip().rstrip()

    # {'prefix': 'STON', 'postfix': 'O2', 'number': '3101290'}
    return re.search('(?P<TicketPrefix>[A-Z]*)/?(?P<TicketPostfix>[A-Z0-9]*)\s*(?P<TicketNumber>[0-9]*)()', stripped)


def get_lastname(name):
    return name.split(',')[0].lstrip().rstrip()


# Define pre-pipeline transformations. Mutates input.
def prePipelineProcess(df):
    df.pop('PassengerId')
    match_objects = [lex_ticket(value[0]) for value in pd.DataFrame(df['Ticket']).values]
    for group_name in ('TicketPrefix', 'TicketPostfix', 'TicketNumber'):
        df[group_name] = [object.group(group_name) for object in match_objects]
    df.pop('Ticket')
    df['Name'] = df['Name'].map(get_lastname)


# Load from files
training = pd.read_csv('train.csv')
testing = pd.read_csv('test.csv')

# Extract target
target = training.pop('Survived')
pd.DataFrame(target).to_csv('target.csv', index=False)
passenger_id = testing['PassengerId']
pd.DataFrame(passenger_id).to_csv('passenger_id.csv', index=False)

# Munge data
prePipelineProcess(training)
prePipelineProcess(testing)

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
    ('cat', categorical_pipeline, ['Sex', 'Cabin', 'Embarked'])
])



sparse.save_npz("training", transformer.fit_transform(training))
sparse.save_npz("testing", transformer.transform(testing))
