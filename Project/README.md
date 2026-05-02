# Titanic Survival Predictions

This project applies multiple machine learning classification models to the Titanic dataset to predict passenger survival based on engineered features and demographic data. Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset

## Overview
The goal of this project is to predict whether a passenger survived the Titanic disaster using structured, tabular data. The dataset contains information on 891 passengers, including demographic details, ticket class, fare, and family relationships. This is formulated as a binary classification problem, where the target variable Survived indicates whether a passenger lived (1) or died (0).

The approach taken in this project emphasizes feature engineering and model comparison. Instead of relying solely on raw variables, several new features were created, such as family size, title extraction from names, cabin assignment indicators, and ticket-sharing counts. These engineered features help capture underlying social and economic patterns that influenced survival.

Multiple machine learning models were trained and evaluated, including both simple models like Logistic Regression and more advanced ensemble methods such as Random Forest and Gradient Boosting.

### Best Result:

The best-performing model, Gradient Boosting, achieved a cross-validation accuracy of approximately 83.4%, slightly outperforming other ensemble models.

## Summary of Work Done
### Data

The dataset consists of a single CSV file containing passenger-level information. Each row represents one individual, and the columns include both numerical and categorical features.

The dataset contains:

- 891 total instances
- A mix of numerical features (Age, Fare, SibSp, Parch)
- Categorical features (Sex, Embarked, Cabin, Ticket)
- A binary target variable (Survived)

Missing values were present in several columns, particularly Age, Cabin, and Embarked, which required preprocessing before modeling.

### Preprocessing/Cleanup
Significant preprocessing was required to prepare the data for machine learning. Missing values in the Age column were filled using the mean, while missing cabin values were replaced with a placeholder to indicate unknown status.

A major focus of this project was feature engineering, which led to several new variables:

- Family Size: Combined number of siblings/spouses and parents/children
- Family Size Grouped: Categorized into small, medium, and large families
- Title: Extracted from passenger names to capture social status
- Cabin Assigned: Binary indicator for whether a passenger had a cabin
- Fare and Age Binning: Converted continuous values into categories
- Name Length: Proxy for social status
- Ticket Sharing Counts: Number of passengers sharing the same ticket

These features significantly improved the model’s ability to capture survival patterns.

Categorical variables were encoded using one-hot encoding and ordinal encoding, while numerical features were passed through without scaling where appropriate.

## Data Visualization

### Key Findings:

| Feature | Pattern | Survival Impact |
|------|------|------|
| Sex   | Female: 74%, Male: 19%   | Strongest predictor   |
| Pclass   | 1st: 63%, 2nd: 47%, 3rd: 24%   | Clear class effect   |
| Cabin_Assigned   | Yes: 67%, No: 30%   | Wealth indicator   |
| Family_Size_Grouped   | Small: 58%, Alone: 30%, Large: 16%   | Optimal group advantage   |
| Title   | Noble/Mrs: 78%, Mr: 16%   | Status/gender combined   |
| Age   | Youngest (0-19): 48%, Middle (19-25): 33%   | Children survived more   |
| Fare   | Highest bracket: 64%, Lowest: 22%   | Direct wealth correlation   |
