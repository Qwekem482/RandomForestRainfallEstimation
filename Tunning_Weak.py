import pandas as pd
import numpy as np
from ImportData import tuning_reg_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def display(param, score, index):
    print('Best Param:', param)
    print('Best Score:', score)
    print('Best Index', index)



#Get data
x_train, y_train = tuning_reg_data(strong=False)



#Train and valid
# n_estimator = [50, 100, 200, 300,..., 3000]
# max_feature, min_samples_split = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# min_samples_leaf, min_weight_fraction_leaf = [0.025 ,0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]
# max_depth = [100, 200,..., 1000, 1200, 1400,..., 3000,3500, 4000,..., 8000, None]
# max_depth2 = [500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 900, 950]
tunning_param = {'n_estimators': [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500],                
                 'max_features': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                 'min_samples_split': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                 'max_samples': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                 'min_samples_leaf': [0.025 ,0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5],
                 'min_weight_fraction_leaf': [0.025 ,0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5],
                 'random_state': [1]}

scores = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']

model = RandomForestRegressor()
print('Start')
search = RandomizedSearchCV(estimator=model,
                            param_distributions = tunning_param,
                            n_iter = 100,
                            scoring=scores,
                            refit='r2',
                            cv=10,
                            random_state=10)
search.fit(x_train, y_train)

display(search.best_params_, search.best_score_, search.best_index_)


out = pd.DataFrame(data=search.cv_results_)
out.to_excel('Tunning/regression/weak/random_search.xlsx')