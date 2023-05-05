import pandas as pd
import numpy as np
from ImportData import get_tuning_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#Get data
x_train, x_valid, y_train, y_valid = get_tuning_data(classifier=True, strong=False)

#Train and valid
"""'n_estimators': (50, 500),
'max_depth': (1, 10),
'min_samples_split': (2, 1000),
'max_leaf_nodes': (1, 50),
'min_samples_leaf': (1, 500),
'max_features': (1, 5),
'max_features': ['sqrt', 'log2', None],"""

param_dist = {'n_estimators': range(1, 1000)}
"""'max_depth': range(1, 1000),
              'min_samples_split': range(2, 1000),
              'max_leaf_nodes': range(1, 1000),
              'min_samples_leaf': range(1, 1000),
              'max_features': ['sqrt', 'log2', None]}"""

scores = {'accuracy', 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'}

model = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=model,
                            param_distributions=param_dist,
                            scoring=scores,
                            refit='accuracy',
                            error_score='raise',
                            n_iter=100,
                            cv=10)
random_search.fit(x_train, y_train)



#Score
print('Best index:', random_search.best_index_)
print('Best scores:', random_search.best_score_)
print('Best hyperparameters:',  random_search.best_params_)

out_score = pd.DataFrame(data=random_search.cv_results_)
out_score.to_excel('HPT_class_nest.xlsx')
