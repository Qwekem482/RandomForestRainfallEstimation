import pandas as pd
import numpy as np
from ImportData import get_tuning_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

#Get data
x_train, x_valid, y_train, y_valid = get_tuning_data(classifier=False, strong=True)



#Train and valid
param_dist = {'n_estimators': [498],
              'max_depth': [282],
              'min_samples_split': range(2, 1000),
              'max_leaf_nodes': [1000],
              'min_samples_leaf': [1000],
              'max_features': ['sqrt']}

model = RandomForestRegressor()
random_search = RandomizedSearchCV(estimator=model,
                            param_distributions=param_dist,
                            scoring='r2',
                            error_score='raise',
                            n_iter=100,
                            cv=10)
random_search.fit(x_train, y_train)



#Score
print('Best index:', random_search.best_index_)
print('Best scores:', random_search.best_score_)
print('Best hyperparameters:',  random_search.best_params_)

out_score = pd.DataFrame(data=random_search.cv_results_)
out_score.to_excel('HPT_reg_strong_nest.xlsx')