import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('Datasets/Football_features_25-26 (V2).csv')
# print(df.head(15))

X = df.drop(columns = ["Result", "Home_team", "Away_team"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = RandomForestClassifier(min_samples_leaf = 8, n_estimators = 200, max_depth = 5) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

def metrics(y_test, y_pred, y_train, y_pred_train):
    score = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    score_train = accuracy_score(y_train, y_pred_train)

    print(f' Score (Test) : {score}, Score (Train) {score_train}')
    print(cr)
    
# metrics(y_test, y_pred, y_train, y_pred_train)


def hyperparameter_tuning(X_train, y_train):
    param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid = param_grid,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1,
        verbose = 2
    )

    grid_search.fit(X_train, y_train)
    print(f'Best Parameter : {grid_search.best_estimator_}')
    print(f'Best CV Score : {grid_search.best_score_}')

# hyperparameter_tuning(X_train, y_train)

