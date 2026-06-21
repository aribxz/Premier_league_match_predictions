# Premier League Match Outcome Prediction

![Match Outcome Distribution](Images/PL%20Match%20Outcome.png)

A machine learning project focused on predicting Premier League match outcomes using historical football data, feature engineering, and ensemble learning techniques.

The project was built alongside learning machine learning algorithms from first principles. Rather than treating models as black boxes, the emphasis was placed on understanding the complete ML workflow: data collection, preprocessing, feature engineering, model training, hyperparameter tuning, overfitting analysis, and evaluation.

---

## Project Objective

Predict the outcome of Premier League matches before kickoff.

Target classes:

| Label | Meaning  |
| ----- | -------- |
| H     | Home Win |
| D     | Draw     |
| A     | Away Win |

---

## Project Philosophy

This repository follows a simple learning cycle:

```text
Learn Algorithm
        ↓
Apply Algorithm
        ↓
Analyze Results
        ↓
Improve Dataset
        ↓
Learn Next Algorithm
        ↓
Repeat
```

As new machine learning algorithms are learned, the same football dataset is continuously improved and reused.

---

## Dataset

The original data was obtained from Football Data UK and contained over 100 match-related variables including:

* Teams
* Goals scored
* Match results
* Shots
* Corners
* Cards
* Betting odds
* Referee information
* Additional match statistics

A custom preprocessing pipeline was developed to convert raw match data into machine learning features that would be available before a match is played.

---

## Feature Engineering

A major focus of the project was feature engineering and preventing data leakage.

All features are generated chronologically using only information available before the target match.

### Initial Features

* Home Form
* Away Form
* Home Average Goals Scored
* Away Average Goals Scored
* Home Average Goals Conceded
* Away Average Goals Conceded

These statistics are calculated using each team's previous five matches.

### Advanced Features

The second version of the dataset introduced more than 30 engineered features, including:

* Form Difference
* Goal Difference
* Attack Difference
* Defence Difference
* Win Percentage
* Draw Percentage
* Clean Sheet Percentage
* Failed To Score Percentage
* Season Points
* League Position
* Position Difference
* Venue-Specific Form
* Points Ratio
* Multiple relative-strength features

These additions were motivated by football-specific insights and significantly improved model performance.

---

## Models Implemented

### Decision Tree

The first model used to understand:

* Tree-based learning
* Overfitting
* Model evaluation
* Hyperparameter tuning

Parameters explored:

* max_depth
* min_samples_split
* min_samples_leaf
* criterion (gini / entropy)

---

### Random Forest

Implemented to reduce variance and improve generalization.

Experiments included:

* Number of estimators
* Maximum depth
* Minimum samples per split
* Minimum samples per leaf

Feature importance analysis was also used to identify the most influential predictors.

---

### AdaBoost

Implemented using decision tree stumps as weak learners.

Experiments included:

* Learning rate
* Number of estimators
* Weak learner depth

This stage provided practical experience with boosting methods and bias-variance tradeoffs.

---

## Key Findings

### Data Leakage Matters

One of the most important lessons from this project was learning how easily future information can leak into training data and artificially inflate performance.

The preprocessing pipeline was intentionally designed to generate features before updating team histories to avoid leakage.

### Feature Engineering > Algorithm Switching

The largest performance improvements came from improving the dataset rather than changing models.

Adding richer football-specific features produced greater gains than extensive hyperparameter tuning.

### Overfitting Can Be Misleading

Early models achieved very high training accuracy while performing poorly on unseen matches.

Comparing training and testing performance became a critical part of model evaluation.

---

## Results

### Baseline

Class distribution:

* Home Win ≈ 42.6%
* Away Win ≈ 30.0%
* Draw ≈ 27.4%

A naive model predicting only home wins would achieve approximately 42.6% accuracy.

### Best Models

The strongest models achieved approximately 50% test accuracy, demonstrating that the models were learning meaningful patterns beyond the baseline.

Although draws remain difficult to predict, the project showed measurable improvements through feature engineering and richer historical information.

---

## Project Structure

```text
PL 25-26/
├── Datasets/
├── Documentation/
├── Files/
├── Images/
└── Notebooks/
```

### Datasets

Contains engineered feature datasets used for training and evaluation.

### Documentation

Detailed explanations of:

* Decision Trees
* Random Forests
* AdaBoost
* Feature engineering decisions
* Experimental results

### Files

Python implementations of:

* Decision Tree
* Random Forest
* AdaBoost

### Images

Visualizations of match outcome distributions.

### Notebooks

* Data cleaning
* Preprocessing
* Feature engineering pipelines

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Google Colab

---

## Future Work

Planned improvements include:

* XGBoost
* LightGBM
* CatBoost
* ELO Ratings
* Expected Goals (xG)
* Injury Information
* Market Odds
* Multi-season datasets
* Chronological validation strategies

---

## Final Takeaway

This project began as an exercise in learning machine learning algorithms but evolved into a practical study of how real-world ML systems are built.

The most valuable lesson learned throughout the process was:

> Better data often matters more than better models.
