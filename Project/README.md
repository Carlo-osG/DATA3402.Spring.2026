# Titanic Survival Predictions

This repository attempts to predict the chances of survival and the factors that increase and/or decrease the chances of survival applying Logistic Regression, Decision Tree Classifier, Gaussian Naive Bayes and more from the Titanic Dataset. Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset

## Overview
The task is to predict whether a passenger survived the Titanic disaster using structured tabular data such as age, sex, ticket class, and family relationships. This is a binary classification problem where the target variable is Survived (0 = No, 1 = Yes).

My approach focuses on:

- Extensive feature engineering (family size, title extraction, cabin assignment, etc.)
- Handling missing values and categorical encoding
- Training and comparing multiple machine learning models

I evaluated a wide range of models including Decision Trees, Random Forests, Gradient Boosting, and more.

### Best Result:
Gradient Boosting Classifier achieved ~83.4% cross-validation accuracy, outperforming all other models.
