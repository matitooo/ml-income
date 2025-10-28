import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data_path="data/training_data.data"
columns = [ "age",
    "employment_type",
    "fwf",
    "education_level",
    "schooling_period",
    "marital_status",
    "employment_area",
    "partnership",
    "ethnicity",
    "gender",
    "financial_gains",
    "financial_losses",
    "weekly_working_time",
    "country_of_birth",
    "income"
]


def preprocess_complete(data_path,names=columns,drop_unk=False):
    df=pd.read_csv(data_path,names=columns)

    #Remove initial spaces in string entries
    df = df.map(lambda x: x.lstrip() if isinstance(x, str) else x)

    #Drop duplicates
    df = df.drop_duplicates()

    if drop_unk:
        df=df[df['income']!="?"]
    
    else:
        df=df[df['income']=="?"]

    categorical_cols = ['employment_type','employment_area','education_level','marital_status','partnership','ethnicity','country_of_birth','gender']
    numerical_cols = ['age','fwf','schooling_period','financial_gains','financial_losses','weekly_working_time']
    df = df.drop(columns=['schooling_period'])
    numerical_cols.remove('schooling_period')
    #Map defining
    married_map = {
        'Husband': 'Married',
        'Wife': 'Married',
        'Not-in-family': 'Not married',
        'Own-child': 'Not married',
        'Unmarried': 'Not married',
        'Other-relative': 'Not married'
    }
    df['partnership'] = df['partnership'].map(married_map)
    y = df['income']
    if drop_unk:
        y = y.map({'<=50K': 0, '>50K': 1}).astype(int)
    X = df.drop(columns=['income'])
    #One hot encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    std = StandardScaler()
    X[numerical_cols] = std.fit_transform(X[numerical_cols]) 
    if drop_unk:
        return X,y 
    else:
        return X

