## Project Overview

The goal of this project was to implement decision trees (the theoretical part that I was learning). To make it more handy, I decided to take a real dataset of football matches for the Premier League season 25-26 from a UK database website. 

The prediction target was:

|Label|Meaning|
|---|---|
|H|Home Team Wins|
|D|Draw|
|A|Away Team Wins|

This project was intentionally approached from the perspective of learning machine learning fundamentals rather than simply maximizing accuracy. Throughout the process, emphasis was placed on understanding:

- Data preprocessing
    
- Feature engineering
    
- Data leakage
    
- Decision Trees
    
- Overfitting
    
- Hyperparameter Tuning
    
- Model Evaluation
    

---

# 1️. Acquiring the Dataset

The dataset was obtained from Football Data UK.

The original dataset contained match information such as:

- Teams
    
- Goals scored
    
- Match result
    
- Betting odds
    
- Referee information
    
- Cards
    
- Corners
    
- Shots
    
- Many other statistics
    

The raw dataset contained more than 100 columns.

Example:

|HomeTeam|AwayTeam|FTHG|FTAG|FTR|
|---|---|---|---|---|
|Liverpool|Bournemouth|4|2|H|

Where:

- FTHG = Full Time Home Goals
    
- FTAG = Full Time Away Goals
    
- FTR = Full Time Result
    

---

# 2️. Initial Problem with Raw Data

A Decision Tree cannot directly understand football teams.

For example:

```text
Liverpool
Arsenal
Chelsea
```

are simply strings.

Even if they were encoded numerically, the model would not understand:

- Current form
    
- Momentum
    
- Recent performance
    

A more meaningful representation was needed.

This led to the first major project decision:

> Instead of using team names directly, create features describing recent team performance.

---

# 3️. Feature Engineering Strategy

The question became:

> What information would a human football analyst use before a match?

Common answers include:

- Recent form
    
- Goals scored recently
    
- Goals conceded recently
    

Therefore, the following features were selected:

### Home Team Features

- Home_Form
    
- Home_Avg_Scored
    
- Home_Avg_Conceded
    

### Away Team Features

- Away_Form
    
- Away_Avg_Scored
    
- Away_Avg_Conceded
    

These features would be calculated using each team's previous five matches.

---

# 4️. The Most Important Concept: Data Leakage

A major issue emerged during feature engineering.

Suppose we want to predict:

```text
Liverpool vs Bournemouth
```

If we calculate Liverpool's form using a dataset that already contains the result of this match, we accidentally reveal the answer to the model.

Example:

```text
Liverpool wins 4-2
```

If those four goals are included in Liverpool's statistics before prediction, the model has effectively seen part of the answer.

This is called:

## Data Leakage

Data leakage occurs when information from the future is accidentally used during training.

The model appears more accurate than it actually is.

Avoiding leakage became one of the primary design goals of the preprocessing pipeline.

---

# 5️. Chronological Ordering

Before calculating any features, matches were sorted by date.

Conceptually:

```python
df = df.sort_values("Date")
```

Why?

Because football is a time series problem.

A team's history only makes sense when matches are processed in chronological order.

---

# 6️. Building Team Histories

A dictionary was used to store historical information for every team.

Conceptually:

```python
team_history = {}
```

Example structure:

```python
{
    "Liverpool": [
        {
            "Points": 3,
            "goals_scored": 4,
            "goals_conceded": 2
        }
    ]
}
```

Each team accumulated a growing list of previous performances.

---

# 7️. Generating Last 5 Match Statistics

A helper function was created to summarize a team's previous five matches.

```python
def get_last5_stats(history):

    if len(history) == 0:
        return [0, 0, 0]

    last5 = history[-5:]

    form = sum(match["Points"] for match in last5)

    avg_scored = (
        sum(match["goals_scored"] for match in last5)
        / len(last5)
    )

    avg_conceded = (
        sum(match["goals_conceded"] for match in last5)
        / len(last5)
    )

    return [form, avg_scored, avg_conceded]
```

### What this function does

For a given team:

1. Takes the last five matches.
    
2. Calculates total points.
    
3. Calculates average goals scored.
    
4. Calculates average goals conceded.
    

Returns:

```python
[
    form,
    avg_scored,
    avg_conceded
]
```

These values become model features.

---

# 8️. Building the Feature Dataset

The core preprocessing loop was:

```python
for _, row in df.iterrows():

    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]

    home_history = team_history.get(home_team, [])
    away_history = team_history.get(away_team, [])

    home_form, home_avg_scored, home_avg_conceded = get_last5_stats(home_history)

    away_form, away_avg_scored, away_avg_conceded = get_last5_stats(away_history)

    feature_rows.append({
        "Home_team": home_team,
        "Away_team": away_team,
        "Home_form": home_form,
        "Away_form": away_form,
        "Home_avg_scored": home_avg_scored,
        "Away_avg_scored": away_avg_scored,
        "Home_avg_conceded": home_avg_conceded,
        "Away_avg_conceded": away_avg_conceded,
        "Result": row["FTR"]
    })

    # Update team history AFTER features are created
```

---

# 9️. Why the Order Matters

The pipeline followed this sequence:

```text
Read Previous History
        ↓
Create Features
        ↓
Save Features
        ↓
Update Team History
```

This ordering is crucial.

If team history were updated before feature creation:

```text
Current Match
        ↓
Added to History
        ↓
Used as Feature
```

then leakage would occur.

The project intentionally avoided this mistake.

---

# 10. Final Feature Dataset

The final engineered dataset contained:

|Feature|
|---|
|Home_Form|
|Away_Form|
|Home_Avg_Scored|
|Away_Avg_Scored|
|Home_Avg_Conceded|
|Away_Avg_Conceded|
|Result|

Dataset size:

```text
380 Matches
```

which corresponds to a complete Premier League season.

---

# 1️1️. Exporting and Moving to VS Code

After preprocessing was completed in Google Colab:

```python
feature_df.to_csv(
    "football_features.csv",
    index=False
)
```

The engineered dataset was exported and loaded into VS Code for modeling.

---

# 1️2️. Building the First Decision Tree

Input features:

```python
X
```

Target:

```python
y = Result
```

The dataset was split:

```text
80% Training
20% Testing
```

using:

```python
train_test_split()
```

A baseline Decision Tree was trained.

---

# 1️3️. Initial Results

Initial performance:

```text
Test Accuracy ≈ 44.7%
```

Classification report showed:

- Home wins predicted reasonably well
    
- Away wins predicted reasonably well
    
- Draws were difficult
    

This was expected because football draws are notoriously difficult to predict.

---

# 1️4️. Discovering Overfitting

A critical observation was made.

Training accuracy:

```text
~99%
```

Testing accuracy:

```text
~44%
```

This indicates severe overfitting.

The model had memorized the training set rather than learning general football patterns.

---

# 1️5️. Hyperparameter Tuning

Grid Search was used to search multiple Decision Tree configurations.

Parameters explored:

```python
{
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "criterion"
}
```

Key ideas:

### max_depth:

Controls how deep the tree may grow.

### min_samples_leaf:

Prevents leaves from containing tiny numbers of matches.

### min_samples_split:

Prevents excessive splitting.

### criterion:

Controls how split quality is measured.

Options:

```text
gini
entropy
```

---

# 1️6️. Tuned Model Results

Best parameters were manually inserted into the model after Grid Search.

Result:

```text
Training Accuracy:
99% → 52%

Testing Accuracy:
44% → 44%
```

This was one of the most important discoveries in the project.

The model became dramatically less overfit.

However, test performance barely changed.

---

# 1️7️. Major Conclusion

The Decision Tree was no longer the primary limitation.

The real limitation became:

## Feature Quality

The model only knew:

```text
Form
Goals Scored
Goals Conceded
```

It did not know:

```text
Team Strength
Home/Away Tendencies
Squad Quality
Historical Strength
```

This means the next improvements should focus on feature engineering rather than more aggressive tuning.

---

# 1️8️. Lessons Learned

### Machine Learning is often a data problem.

Improving features can matter more than improving algorithms.

---

### Data Leakage is dangerous.

A model can appear highly accurate while secretly using future information.

---

### Overfitting can be diagnosed by comparing:

```text
Training Accuracy
vs
Testing Accuracy
```

---

### Hyperparameter Tuning cannot compensate for weak features.

Better trees cannot invent information that does not exist.

---

### Feature Engineering is often the most valuable part of the pipeline.

The custom rolling-history features contributed far more value than changing Decision Tree settings.

---

#  Future Improvements

Potential next steps:

### Additional Features

- Goal Difference Form
    
- Home-only Form
    
- Away-only Form
    
- Win Percentage
    
- Clean Sheet Percentage
    

### Alternative Models

- Random Forest
    
- XGBoost
    
- LightGBM
    

### Better Validation

Use chronological train-test splits instead of random splits to simulate real-world prediction.

---

# Final Project Outcome

A complete end-to-end football prediction pipeline was built:

```text
Raw Football Data
        ↓
Data Cleaning
        ↓
Chronological Ordering
        ↓
Leakage-Free Feature Engineering
        ↓
Feature Dataset Creation
        ↓
Decision Tree Training
        ↓
Overfitting Analysis
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
```

The project successfully demonstrated the full machine learning workflow while highlighting the importance of thoughtful feature engineering, careful preprocessing, and rigorous evaluation.