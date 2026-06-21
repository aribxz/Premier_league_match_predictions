
## From Raw Match Data → Feature Engineering → Random Forest → AdaBoost

> A practical machine learning project built alongside learning ML algorithms in theory.
>
> Goal: Predict match outcomes (**H = Home Win, D = Draw, A = Away Win**) using historical football match data and progressively improve the dataset as new ML algorithms are learned.

---

# 📖 Project Philosophy

Instead of learning machine learning only through theory, the goal of this project was:

```text
Learn Algorithm
        ↓
Apply Algorithm
        ↓
Improve Dataset
        ↓
Analyze Results
        ↓
Learn Next Algorithm
        ↓
Repeat
```

This turns the project into a living ML laboratory where the dataset evolves alongside ML knowledge.

---

# Phase 1: Initial Dataset

The original dataset was built from historical Premier League match results.

For each match, information available before kickoff was generated using the previous 5 matches of each team.

The first version contained features such as:

| Feature |
|----------|
| Home_form |
| Away_form |
| Home_avg_scored |
| Away_avg_scored |
| Home_avg_conceded |
| Away_avg_conceded |
| Result |

---

## Initial Idea

The intuition was:

```text
Recent Form
+
Recent Attack Strength
+
Recent Defensive Strength
=
Enough Information To Predict Matches
```

Which sounds reasonable.

However...

Football is not that simple.

---

# Phase 2: Random Forest

After learning Random Forests theoretically, the dataset was used to train a Random Forest Classifier.

---

## First Results

The model achieved roughly:

```text
Train Accuracy ≈ 99%
Test Accuracy ≈ 35%
```

---

## What This Told Us

This was a classic example of:

# OVERFITTING

The model had memorized the training data.

Visually:

```text
Training Data

● ● ● ● ● ● ● ● ●
Random Forest memorizes everything

Train Accuracy = 99%

New Unseen Matches

✗ ✗ ✗ ✗ ✗

Test Accuracy = 35%
```

The model was not learning football.

It was learning the dataset.

---

# Phase 3: First Feature Engineering Pass

To reduce overfitting and provide more meaningful information, new difference-based features were added.

---

## Form Difference

### Formula

```text
Form Difference
=
Home Form - Away Form
```

### Example

```text
Liverpool Form = 12
Everton Form = 6

Form Difference = +6
```

### Why?

Because the model doesn't care about the absolute value as much as:

```text
Who is stronger right now?
```

---

## Attack Difference

### Formula

```text
Attack Difference
=
Home Avg Scored - Away Avg Scored
```

### Example

```text
Liverpool = 2.4 goals/game
Everton = 1.1 goals/game

Attack Difference = 1.3
```

---

## Defense Difference

### Formula

```text
Defense Difference
=
Home Avg Conceded - Away Avg Conceded
```

### Example

```text
Liverpool = 0.8 conceded/game
Everton = 1.6 conceded/game

Defense Difference = -0.8
```

---

# Results After First Feature Pass

Accuracy improved significantly.

Approximately:

```text
Train ≈ 63%
Test ≈ 50%
```

This was the first major breakthrough.

---

# Key Insight

At this point something important became clear.

The issue was not necessarily:

```text
Random Forest is bad
```

The issue was:

```text
The dataset is missing information
```

---

# Phase 4: Going Back To The Original CSV

Instead of continuing to tune models endlessly, the project went back to the raw football data.

This was one of the most important decisions of the entire project.

---

## Why?

Because:

```text
Garbage In
=
Garbage Out
```

No amount of tuning can save a dataset that lacks useful information.

---

# Dataset Rebuild

The dataset generation notebook was redesigned.

A complete team history system was introduced.

---

# Team History Structure

For every team:

```python
team_history[team].append({
    "Points": ...,
    "goals_scored": ...,
    "goals_conceded": ...,
    "venue": ...
})
```

This allowed much richer features to be calculated.

---

# Phase 5: Advanced Feature Engineering

---

# Goal Difference

### Formula

```text
Goal Difference
=
Goals Scored
-
Goals Conceded
```

Calculated over the last 5 matches.

---

## Why?

Goal difference is often more predictive than raw goals scored.

Example:

```text
Team A

Scored = 10
Conceded = 2

Goal Difference = +8
```

vs

```text
Team B

Scored = 10
Conceded = 10

Goal Difference = 0
```

Same attack.

Very different teams.

---

# Win Percentage

### Formula

```text
Wins / Last 5 Matches
```

---

## Why?

A team winning 4 out of 5 games carries important information.

---

# Draw Percentage

### Formula

```text
Draws / Last 5 Matches
```

---

## Why?

The project was struggling badly with draws.

This feature was introduced to help identify teams that frequently draw matches.

---

# Clean Sheet Percentage

### Formula

```text
Matches Without Conceding
/
Last 5 Matches
```

---

## Why?

Defensive reliability is important.

Example:

```text
Arsenal
0 goals conceded
```

is a strong signal.

---

# Failed To Score Percentage

### Formula

```text
Matches Without Scoring
/
Last 5 Matches
```

---

## Why?

Teams that struggle to score tend to lose more often.

---

# Season Points

For every team:

```text
3 points = Win
1 point = Draw
0 points = Loss
```

Accumulated throughout the season.

---

## Features Added

```text
Home_season_points
Away_season_points
Points_difference
```

---

# Why?

League table strength matters.

Example:

```text
Liverpool: 45 Points
Burnley: 15 Points
```

This information should influence predictions.

---

# League Position

Dynamic league positions were calculated before every match.

---

## Features

```text
home_position
away_position
Position_difference
Position_gap
```

---

## Why?

Humans naturally use league positions when predicting matches.

The model should have access to the same information.

---

# Venue-Specific Form

One of the strongest football-specific additions.

---

## Home Form

Only matches played at home.

```text
Home_home_form
```

---

## Away Form

Only matches played away.

```text
Away_away_form
```

---

## Difference

```text
Venue_form_difference
```

---

## Why?

Many teams behave differently:

```text
Home:
Strong

Away:
Weak
```

Ignoring venue can hide important patterns.

---

# Points Ratio

### Formula

```text
(Home Points + 1)
/
(Away Points + 1)
```

(+1 prevents division by zero)

---

## Why?

Ratios often capture relative strength better than differences.

---

# Final Engineered Dataset

Approximately:

```text
30+ Features
```

Including:

```text
Form
Attack
Defense
Goal Difference
Win %
Draw %
Clean Sheet %
Failed To Score %
Season Points
League Position
Venue Form
Ratios
Differences
```

---

# Random Forest Revisited

After rebuilding the dataset:

```text
Train Accuracy ≈ 60-63%
Test Accuracy ≈ 50%
```

Huge improvement over:

```text
Train = 99%
Test = 35%
```

---

# Feature Importance Analysis

Most important features included:

```text
Points Difference
Points Ratio
League Position
Goal Difference Difference
Season Points
```

---

## Interpretation

The model cared far more about:

```text
How strong is Team A compared to Team B?
```

than raw attack or defense numbers.

This validated much of the feature engineering work.

---

# Draw Prediction Problem

Throughout the project:

```text
Home Wins
Away Wins
```

were predicted reasonably.

Draws remained difficult.

---

## Why?

Football draws are inherently difficult.

They are:

```text
Less common
+
More random
+
Less distinguishable
```

than wins and losses.

---

# AdaBoost Phase

After learning AdaBoost theoretically, the same dataset was used.

---

## Initial Model

```python
AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=500,
    learning_rate=0.5
)
```

---

# Understanding AdaBoost

Unlike Random Forest:

```text
Random Forest

Tree 1
Tree 2
Tree 3

Vote
```

AdaBoost works as:

```text
Tree 1
↓
Focus On Mistakes
↓
Tree 2
↓
Focus On Mistakes
↓
Tree 3
```

Each learner tries to correct previous mistakes.

---

# Hyperparameter Experiments

Several combinations were tested.

---

## Learning Rate

```text
0.01
0.05
0.1
0.5
1
```

---

## Estimators

```text
50
100
200
500
1000
```

---

## Tree Depth

```text
max_depth = 1
max_depth = 2
max_depth = 3
```

---

# Interesting Discovery

### Depth = 1

```text
Train ≈ 54%
Test ≈ 41%
```

---

### Depth = 2

```text
Train ≈ 61%
Test ≈ 34%
```

---

## Interpretation

Depth 2 increased training performance but hurt generalization.

This was evidence of:

```text
Overfitting Beginning To Appear
```

---

# AdaBoost Conclusions

AdaBoost did not outperform Random Forest.

Best observed performance was roughly:

```text
≈ 50%
```

Similar to Random Forest.

---

# Dataset Baseline Analysis

Class distribution:

```text
H = 42.6%
A = 30.0%
D = 27.4%
```

Meaning:

A model predicting only:

```text
H
H
H
H
H
```

would already achieve:

```text
42.6% Accuracy
```

---

# Important Realization

The best models achieved:

```text
≈ 50%
```

while the baseline was:

```text
42.6%
```

So the models were genuinely learning.

However:

```text
The bottleneck is now the information available.
```

not the algorithm itself.

---

# Final Conclusions

## What Worked

✅ Difference-based features

✅ Goal Difference

✅ Season Points

✅ League Position

✅ Venue Form

✅ Richer historical information

---

## What Did Not Work

❌ Endless hyperparameter tuning

❌ Expecting AdaBoost to magically fix data limitations

❌ Relying only on recent form

---

## Biggest Lesson

The project demonstrated one of the most important truths in Machine Learning:

```text
Better Data
>
Better Models
```

The largest improvements came from:

```text
Feature Engineering
```

not from:

```text
Algorithm Switching
```

---

# Future Roadmap

As more algorithms are learned:

```text
Decision Tree ✅
Random Forest ✅
AdaBoost ✅
XGBoost ⏳
LightGBM ⏳
CatBoost ⏳
Neural Networks ⏳
```

the same dataset can continue evolving.

Potential future additions:

```text
Team Strength Ratings
ELO Ratings
Expected Goals (xG)
Transfer Activity
Injury Information
Market Odds
Multiple Seasons
```

---

# Final Takeaway

This project successfully achieved its original objective:

```text
Learn Theory
        ↓
Apply Theory
        ↓
Build Dataset
        ↓
Analyze Results
        ↓
Improve Dataset
        ↓
Learn Again
```

Rather than merely training models, the project became an exercise in learning how machine learning systems are actually built in the real world.

And the most valuable lesson learned was:

"Feature engineering and understanding the data often matter more than the choice of algorithm."