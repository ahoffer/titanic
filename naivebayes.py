import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def lex_ticket(ticket):
    # Get everything before the slash
    # Get everything after the slash but before the first number
    # Get all the remaining digits
    stripped = ticket.replace('.', '').upper().lstrip().rstrip()
    print(stripped)
    x = re.search('(?P<prefix>[A-Z]*)/?(?P<postfix>[A-Z]*)\s*(?P<number>[0-9]*)()', stripped)
    print(x.groupdict())


path = 'train.csv'
df = pd.read_csv(path, index_col=None
                 , converters={
        'Age': lambda age: None if not age else pd.to_numeric(age)
    }
                 , dtype={
        'PassengerId': 'int'
        , 'Survived': 'category'
        , 'Pclass': 'int'
        , 'Name': 'str'
        , 'Sex': 'category'
        , 'SibSp': 'int'
        , 'Parch': 'int'
        , 'Ticket': 'str'
        , 'Fare': 'float'
        , 'Cabin': 'category'
        , 'Embarked': 'category'
    }
                 )
print(df.dtypes)
#     ,
# 'Pclass' : object , 'Age', 'Parch', 'Fare', 'Name', 'Sex', 'SibSp', 'Ticket', 'Cabin', 'Embarked'
#
#
#                      }) #: ,
#
#                          PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked})
# target = df['Survived']
# predictors = df.drop(columns=['Survived'])
#
# numeric = df[['PassengerId', 'Pclass', 'Age', 'Parch', 'Fare']]
# categoric = df[['Name', 'Sex', 'SibSp', 'Ticket', 'Cabin', 'Embarked']]
#
# # fill in NaNs
# imp_freq = SimpleImputer(strategy='most_frequent')
# imp_zero = SimpleImputer(strategy='constant', fill_value=0)
# # encode labels
# label_encoder = LabelEncoder()
# standardize_values = StandardScaler(with_mean=True)
#
# # handle numeric data
# for column in numeric.columns:
#     tempCol = numeric[column].values.reshape(-1, 1)
#     if numeric[column].isnull().any():
#         tempCol = imp_zero.fit_transform(tempCol)
#     tempCol = standardize_values.fit_transform(tempCol)
#     numeric[column] = tempCol.reshape(-1)
#
# # handle catagoric data
# for column in categoric.columns:
#     tempCol = categoric[column].values.reshape(-1, 1)
#     if categoric[column].isnull().any():
#         tempCol = imp_freq.fit_transform(tempCol)
#     tempCol = label_encoder.fit_transform(categoric[column].astype(str))
#     categoric[column] = tempCol.reshape(-1)
#
# df = pandas.concat([numeric, categoric], axis=1, sort=False)
#
# X_train, X_valid, y_train, y_valid = train_test_split(df, target, test_size=0.2)
#
# nb = BernoulliNB()
# nb.fit(X_train, y_train)
# print(nb.score(X_valid, y_valid))