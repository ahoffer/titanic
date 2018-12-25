import re

import pandas as pd

def get_title(fullname):
    # Assume everyone has a title for now.
    matcher = re.search('.*,\s+(?P<Title>\w+)', fullname.upper())
    return matcher.group("Title")

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

# Define pre-pipeline transformations. Mutates input.
def pre_pipeline_process(df):
    match_objects = [lex_ticket(value[0]) for value in pd.DataFrame(df['Ticket']).values]
    for group_name in ('TicketPrefix', 'TicketPostfix', 'TicketNumber'):
        df[group_name] = [object.group(group_name) for object in match_objects]
    df.pop('Ticket')
    df['LastName'] = df.Name.map(get_lastname)
    df['Title'] = df.Name.map(get_title)
    df.pop('Name')
