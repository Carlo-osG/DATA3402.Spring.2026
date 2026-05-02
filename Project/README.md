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

### Data Visualization

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

Passengers with longer names show higher survival rates, likely because longer names often include formal titles linked to higher social status. Since wealthier passengers had better access to lifeboats, name length serves as an indirect indicator of class and survival probability.

<img width="576" height="433" alt="image" src="https://github.com/user-attachments/assets/a43b9fc3-728f-4503-a986-7b8251a93e4b" />

### Correlation Matrix: 
Survived correlated with Name_Length (+0.33) and Pclass (-0.34). SibSp and Parch showed correlation, confirming they could be combined.

<img width="768" height="642" alt="image" src="https://github.com/user-attachments/assets/f2cac4df-e6b6-49f5-9036-a70199fb7500" />

Other important findings included:

- Higher fares were strongly associated with higher survival
- Passengers with assigned cabins were more likely to survive
- Smaller families had better survival outcomes than large families
- Titles such as “Mrs” and “Miss” were associated with higher survival rates

These insights directly guided the feature engineering process.

## Problem Formulation

The problem is defined as a supervised classification task:

- Input: Passenger features (demographics, ticket info, engineered features)
- Output: Binary survival prediction (0 or 1)

The dataset was split into training and validation sets using stratified sampling to preserve class balance.

## Models & Preprocessing

### Encoding:
- Ordinal: Family_Size_Grouped (has ranking)
- One-Hot: Sex, Embarked (no ranking)
- Numeric: Pclass, Age, Fare, Cabin_Assigned, Name_Size, TicketNumberCounts

Pipeline: ColumnTransformer → SimpleImputer → Encoders → Model

The models tested include:

- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Classifier
- Logistic Regression
- Gaussian Naive Bayes
- AdaBoost
- Extra Trees
- Gradient Boosting

All models were tuned using GridSearchCV with 5-fold stratified cross-validation to ensure fair comparison.

## Training

Training was performed using the Scikit-learn library. The dataset was split into 80% training and 20% validation data, ensuring that the class distribution remained consistent.

Hyperparameters for each model were optimized using grid search. Due to the relatively small size of the dataset, training times were short, allowing for extensive experimentation across models.

## Performance Comparison

The performance of each model was evaluated using cross-validation accuracy.

| Model | Accuracy |
|------|------|
| Gradient Boosting   | 0.8343   |
| Random Forest   | 0.8287   |
| Extra Trees   | 0.8132   |
| Logistic Regression   | 0.8048   |
| KNN   | 0.8048   |
| Decision Tree   | 0.8020   |
| SVC   | 0.8006   |
| AdaBoost   | 0.7964   |
| Naive Bayes   | 0.7655   |

Ensemble methods clearly performed the best, with Gradient Boosting slightly outperforming Random Forest.

## Conclusion

This project demonstrates that feature engineering is critical in tabular machine learning problems. By creating meaningful features from the raw data, model performance improved significantly.

The most influential features were:

- Sex
- Passenger Class (Pclass)
- Title
- Cabin Assignment
- Fare
 -Family Size

Additionally, ensemble models proved to be the most effective approach for this dataset.

## Future Work

There are several directions for improving this project:

- Implement advanced boosting algorithms such as XGBoost, LightGBM, or CatBoost
- Apply ensemble stacking techniques
- Perform deeper hyperparameter tuning
- Explore feature selection methods to reduce redundancy

## How to Reproduce Results

To reproduce the results, follow these steps:

- Clone the repository
  - git clone Kaggle_Tabular_Data.ipynb
  - cd Kaggle_Tabular_Data.ipynb

- Install required packages
  - pip install pandas numpy seaborn matplotlib scikit-learn

- Download the dataset from Kaggle and place it in the project directory

- Run the training notebook or script to train models and generate predictions

### Repository Structure

- preprocess.ipynb       |  Data cleaning and feature engineering
- visualization.ipynb    |  Exploratory data analysis
- models.py              |  Model definitions
- training.ipynb         |  Training and hyperparameter tuning
- performance.ipynb      |  Model comparison
- inference.ipynb        |  Generate predictions
- utils.py               |  Helper functions
- submission_*.csv       |  Kaggle submission files

### Software Setup

Required libraries:

pandas
numpy
seaborn
matplotlib
scikit-learn

## References
- Kaggle Titanic Competition: https://www.kaggle.com/c/titanic
- Scikit-learn Documentation: https://scikit-learn.org/
- Lecture Notes: Data 3402 (Spring 2025)
- Kaggle Community Notebooks
- Titanic Machine Learning Tutorial (YouTube)
