import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/Football_features_25-26.csv")
X = df.drop(columns = ["Result", "Home_team", "Away_team"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, 
                               min_samples_split = 2, min_samples_leaf = 5)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
train_pred = model.predict(X_train)

def metrics(y_test, prediction, train_pred, y_train):
    score = accuracy_score(y_test, prediction)
    cr = classification_report(y_test, prediction)
    train_score = accuracy_score(y_train, train_pred)

    print(f'The score is : {score}')
    print(f'The classification report is : {cr}')
    print(f'The score is : {train_score}')

def hyperparameter_tunning(X_train, y_train):
    param_grid = {
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_samples_leaf': [1, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=param_grid,
        cv = 10, 
        scoring = "accuracy",
        n_jobs = -1)
    
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters : {grid_search.best_params_}')
    print(f'Best CV Score : {grid_search.best_score_}')

hyperparameter_tunning(X_train, y_train)
metrics(y_test, prediction, train_pred, y_train) 

cd = y.value_counts()
def class_distribution_graph(cd):
    cd.plot(kind='bar')

    plt.title("Match Result Distribution")
    plt.xlabel("Result")
    plt.ylabel("Count")

    plt.show()

def home_vs_away_wins_graph(cd):
    plt.figure(figsize=(8,5))
    plt.bar(cd.index, cd.values)

    plt.title("Premier League Match Outcomes")
    plt.xlabel("Result")
    plt.ylabel("Number of Matches")

    plt.show()

class_distribution_graph(cd)
home_vs_away_wins_graph(cd)
