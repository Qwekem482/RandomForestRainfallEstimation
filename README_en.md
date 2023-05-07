# Model

![RandomForest drawio](https://user-images.githubusercontent.com/80797630/236696148-2991574d-8454-4e86-a1c3-74a5396e757e.png)

# Train and Validate Data
Train and Validate data ("train data" for short) is divided into 3 group according to rain amount to train classification model (RF1):

•	Group 1: No rain (0 mm/h)

•	Group 2: Weak rain (under 2.8 mm/h)

•	Group 3: Strong/Heavy Rain (over 2.8 mm/h)

After that, these data is copied into 2. The first one is for training classification model (RF1). The second one will be divided into 3 part corresponding to 3 group of rain. Part 2 and 3 (Group 2 and 3) will be used for training RF2 and RF3

Because of the data is imbalanced, train data will be re-balanced by SMOTETomek-links technique

# Tunning model
These hyperparemeter will be adjust for tunning process
•	n_estimators: range(100, 3000, 100)
•	max_features: range(0.05, 1.0, 0.05)
•	min_samples_split: range(0.025, 0.5, 0.025)
•	min_samples_leaf: range(0.05, 1.0, 0.05)
•	max_samples: range(0.05, 1.0, 0.05)
•	min_weight_fraction_leaf: range(0.025, 0.5, 0.025)

# Run
Run RandomForest.py to Estimate Rainfall
Modify dataset location in ImportData.py
