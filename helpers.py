import re

import pandas as pd
from h2o import H2OFrame, h2o

social_position = {
    'MASTER': 'YOUTH',
    'MISS': 'YOUTH',
    'MS': 'YOUTH',
    'MLLE': 'YOUTH',
    'CAPT': 'OFFICER',
    'MAJOR': 'OFFICER',
    'COL': 'OFFICER',
    'SIR': 'NOBILITY',
    'DONA': 'NOBILITY',
    'DON': 'NOBILITY',
    'LADY': 'NOBILITY',
    'DR': 'PROFESSIONAL',
    'REV': 'PROFESSIONAL',
    'MR': 'NORMAL',
    'MRS': 'NORMAL',
    'JONKHEER': 'NORMAL',
    'MME': 'NORMAL',
    'THE': 'NORMAL'}


def get_title(fullname):
    # Assume everyone has a title for now.
    matcher = re.search('.*,\s+(?P<Title>\w+)', fullname.upper())
    val = matcher.group("Title")
    if val.isnumeric():
        print(val)
    return val


def get_social_position(title):
    return social_position.get(title)


def lex_ticket(ticket):
    # Get everything before the slash
    # Get everything after the slash but before the first number
    # Get all the remaining digits
    # Return the matcher group object
    stripped = ticket.replace('.', '').upper().lstrip().rstrip()

    # {'prefix': 'STON', 'postfix': 'O2', 'number': '3101290'}
    return re.search('(?P<TicketPrefix>[A-Z]*)/?(?P<TicketPostfix>[A-Z0-9]*)\s*(?P<TicketNumber>[0-9]*)()',
                     stripped)


def get_lastname(fullname):
    return fullname.split(',')[0].lstrip().rstrip().upper()


def setCabinInformation(df):
    # letters = [re.search('[A-Z]', value[0]) for value in pd.DataFrame(df['Cabin'].values)]
    na = df['Cabin'].notna()
    cabins = df.loc[na, 'Cabin']
    letters = [value[0] for value in cabins]
    df.loc[na, 'CabinLetter'] = letters


# Define pre-pipeline transformations. Mutates input.
def pre_pipeline_process(df):
    match_objects = [lex_ticket(value[0]) for value in pd.DataFrame(df['Ticket']).values]
    for group_name in ('TicketPrefix', 'TicketPostfix', 'TicketNumber'):
        df[group_name] = [object.group(group_name) for object in match_objects]
    df.pop('Ticket')
    df['LastName'] = df.Name.map(get_lastname)
    df['Title'] = df.Name.map(get_title)
    df['SocialPosition'] = df.Title.map(get_social_position)
    df.pop('Name')
    setCabinInformation(df)
    return df


def merge_ages(frame, ages):
    df = frame.merge(ages, all_x=True).sort('PassengerId').as_data_frame()
    missing_rows = df['Age'].isna()
    df.loc[missing_rows, 'Age'] = df.loc[missing_rows, 'predict']
    merged_frame = H2OFrame(df)
    # Somehow, the columns, some columns get corrupted in by the merge
    copy_df = h2o.deep_copy(merged_frame, 'copy_df')
    copy_df['Age'] = merged_frame.pop('Age')
    return copy_df.drop('predict')
