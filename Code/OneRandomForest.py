import pandas as pd
import matplotlib.pyplot as pt
from ImportData import get_est_data, resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import r2_score as r2
from sklearn.feature_selection import r_regression

train_data, test_data = get_est_data()
x_param = ['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']


def display(res):
    print('R2' + '\t' + str(r2(res['value'], res['RF'])) + '\t' + str(r2(res['value'], res['IMERG'])))
    print('RMSE' + '\t' + str(rmse(res['value'], res['RF'], squared=False)) + '\t' + str(rmse(res['value'], res['IMERG'], squared=False)))
    print('MAE' + '\t' + str(mae(res['value'], res['RF'])) + '\t' + str(mae(res['value'], res['IMERG'])))


#strong Regression
x_strong_train = train_data[x_param]
y_strong_train = train_data['value']

x_strong_test = test_data[x_param]
y_strong_test = test_data['value']

strong_model = RandomForestRegressor(n_estimators=2000,
                                   max_features=0.75,
                                   min_samples_split=0.2,
                                   min_samples_leaf=0.175,
                                   min_weight_fraction_leaf=0.05,
                                   max_samples=0.9,
                                   random_state=1)
strong_model.fit(x_strong_train, y_strong_train)

strong_predict = strong_model.predict(x_strong_test)

test_data['RF'] = strong_predict



#Join & Export
result = pd.concat([test_data])
result.to_excel('result.xlsx')



#Statistics
print('Statistic' + '\t' + 'Random Forest' + '\t' + "IMERG")
print('All:')
display(result)
print('Pearson:')
print(r_regression(x_strong_test, strong_predict))

#Visualization
#Plot 1
pt.subplot(1, 2, 1)
pt.plot([-5000, 3000], [-5000, 3000],  c='black')
pt.scatter(result['RF'], result['value'], c='blue', s=10)
pt.axhline(y=0, c='red')
pt.axvline(x=0, c='red')
pt.xlim(-1, 100)
pt.ylim(-1, 100)
pt.xlabel("Rain Gauge Value")
pt.ylabel("Random Forest Value")

#Plot 2
pt.subplot(1, 2, 2)
pt.plot([-5000, 3000], [-5000, 3000],  c='black')
pt.scatter(result['IMERG'], result['value'], c='blue', s=10)
pt.axhline(y=0, c='red')
pt.axvline(x=0, c='red')
pt.xlim(-1, 100)
pt.ylim(-1, 100)
pt.xlabel("Rain Gauge Value")
pt.ylabel("IMERG Value")

pt.show()