import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split



def get_data():

    #Read data
    data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
    data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

    data = pd.concat([data9, data10])
    data.dropna(inplace=True)

    #Preprocessing data
    conditions = [  
        (data['value'] == 0),
        (data['value'] > 0) & (data['value'] <= 2.08),
        (data['value'] > 2.08)
    ]
    values = [0, 1, 2]

    data['rain_group'] = np.select(conditions, values)

    return data

def get_tuning_data(classifier:bool, strong:bool):
    data = get_data()

    if not classifier and strong:
        train_valid_data = data[(data['datetime'] < datetime(2019, 10, 24)) & (data['rain_group'] == 2)]
    elif not classifier and not strong:
        train_valid_data = data[(data['datetime'] < datetime(2019, 10, 24)) & (data['rain_group'] == 1)]
    elif classifier:
        train_valid_data = data[data['datetime'] < datetime(2019, 10, 24)]

    x_train_valid = train_valid_data[['B09B','B10B','B12B','B14B','B16B','I2B','IRB','WVB','CAPE','TCC','TCW','TCWV']]

    if classifier:
        y_train_valid = train_valid_data['rain_group']
    elif not classifier:
        y_train_valid = train_valid_data['value']

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.2)

    return x_train, x_valid, y_train, y_valid