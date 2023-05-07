import pandas as pd
import numpy as np
from datetime import datetime
from imblearn.combine import SMOTETomek




test_date = datetime(2019, 10, 24)
valid_date = datetime(2019, 10, 12)
x_param = ['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']





# Use to read data from xlsx and merge them
# Use to classify rain group for training/tunning
def get_data():

    #Read data
    data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
    data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

    data = pd.concat([data9, data10])
    data.dropna(inplace=True)

    conditions = [(data['value'] == 0), 
                  ((data['value'] > 0) & (data['value'] <= 2.08)), 
                  (data['value'] > 2.08)]
    
    values = [0, 1, 2]
    data['rain_group'] = np.select(conditions, values)

    return data





#Resampling train data for better classification
def resampling(x_train, y_train):
    smote_tomek = SMOTETomek(random_state=0)
    x_train, y_train = smote_tomek.fit_resample(x_train, y_train)

    return x_train, y_train





# Get data for tunning classification
def get_tuning_class_data():
    data = get_data()

    train_data = data[data['datetime'] < valid_date]

    valid_data = data[(data['datetime'] >= valid_date) &
                      (data['datetime'] < test_date)]

    x_train = train_data[x_param]
    y_train = train_data['rain_group']

    x_train, y_train = resampling(x_train, y_train)

    x_valid = valid_data[x_param]
    y_valid = valid_data['rain_group']

    return x_train, y_train, x_valid, y_valid





# Get data for tunning classification 2
def tuning_class_data():
    data = get_data()

    train_data = data[data['datetime'] < test_date]

    x_train = train_data[x_param]
    y_train = train_data['rain_group']

    x_train, y_train = resampling(x_train, y_train)

    return x_train, y_train





# Get data for tunning regression
def get_tuning_reg_data(strong:bool):
    data = get_data()

    if strong:
        train_data = data[(data['datetime'] < valid_date) &
                          (data['rain_group'] == 2)]
        
        valid_data = data[(data['datetime'] >= valid_date) &
                          (data['datetime'] < test_date) &
                          (data['rain_group'] == 2)]
    elif not strong:
        train_data = data[(data['datetime'] < valid_date) &
                          (data['rain_group'] == 1)]
        
        valid_data = data[(data['datetime'] >= valid_date) &
                          (data['datetime'] < test_date) &
                          (data['rain_group'] == 1)]

    x_train = train_data[x_param]
    y_train = train_data['value']

    x_valid = valid_data[x_param]
    y_valid = valid_data['value']

    return x_train, y_train, x_valid, y_valid




# Get data for tunning regression
def tuning_reg_data(strong:bool):
    data = get_data()

    if strong:
        train_data = data[(data['datetime'] < test_date) &
                          (data['rain_group'] == 2)]
    elif not strong:
        train_data = data[(data['datetime'] < test_date) &
                          (data['rain_group'] == 1)]

    x_train = train_data[x_param]
    y_train = train_data['value']


    return x_train, y_train





def get_est_data():
    data = get_data()

    train_data = data[data['datetime'] < valid_date]
    test_data = data[data['datetime'] >= test_date]

    return train_data, test_data