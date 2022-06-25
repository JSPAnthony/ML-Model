#!/usr/bin/env python
# coding: utf-8

# ### Heart Disease Prediction ML Model
# 
# Final Project - ADS January 2022 Cohort
# 
# Dataset source:  https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

# importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import skew
# import IPython
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import pickle as pk
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

import warnings
warnings.filterwarnings('ignore')

# Loading the dataset as heart_df

heart_df = pd.read_csv('heart.csv')


# ### Data Discovery

# Preview the first few rows of the dataset
heart_df.head()

# Preview the last rows of the dataset
heart_df.tail()

# Listing of the columns name
list(heart_df.columns)


# ### The dataset columns information
# 
# Age: age of the patient [years]
# 
# Sex: sex of the patient [M: Male, F: Female]
# 
# ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# 
# RestingBP: resting blood pressure [mm Hg]
# 
# Cholesterol: serum cholesterol [mm/dl]
# 
# FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# 
# RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality
# (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or
# definite left ventricular hypertrophy by Estes' criteria]
# 
# MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# 
# ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# 
# Oldpeak: oldpeak = ST [Numeric value measured in depression]
# 
# ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# 
# HeartDisease: output class [1: heart disease, 0: Normal]

# Data shape
heart_df.shape()

# Data type
heart_df.dtypes()

# The technical summary of the dataset
heart_df.info()

# Statistical summary of columns with numerical value
heart_df.describe()

# Checking for nas
heart_df.isna().sum()

# Checking for null value
heart_df.isnull().count()

# The unique values in the dataset
heart_df.nunique()

# The duplicate values in the dataset
heart_df.duplicated().sum()

# Skewness.

for col in heart_df.columns:
    if heart_df[col].dtype != 'object':
        print(f'{col}: {scipy.stats.skew(heart_df[col])}')


for col in heart_df.columns:
    if heart_df[col].dtype != 'object':
        print(f'{col}: {scipy.stats.skew(heart_df[col])}')
plt.figure(figsize=(15, 5))
sns.kdeplot(x=heart_df['Age'], label='Age')
sns.kdeplot(x=heart_df['RestingBP'], label='R_BP')
sns.kdeplot(x=heart_df['Cholesterol'], label='Chol')
sns.kdeplot(x=heart_df['FastingBS'], label='F_BS')
sns.kdeplot(x=heart_df['MaxHR'], label='M_HR')
sns.kdeplot(x=heart_df['Oldpeak'], label='O_Pk')
sns.kdeplot(x=heart_df['HeartDisease'], label='HD')
plt.legend()
plt.show()

# Kurtosis.
for col in heart_df.columns:
    if heart_df[col].dtype != 'object':
        print(f'{col}: {scipy.stats.kurtosis(heart_df[col])}')
plt.figure(figsize=(15, 5))
sns.kdeplot(x=heart_df['Age'], label='Age')
sns.kdeplot(x=heart_df['RestingBP'], label='R_BP')
sns.kdeplot(x=heart_df['Cholesterol'], label='Chol')
sns.kdeplot(x=heart_df['FastingBS'], label='F_BS')
sns.kdeplot(x=heart_df['MaxHR'], label='M_HR')
sns.kdeplot(x=heart_df['Oldpeak'], label='O_Pk')
sns.kdeplot(x=heart_df['HeartDisease'], label='HD')
plt.legend()
plt.show()

heart_df.value_counts()

heart_df['ChestPainType'].value_counts(ascending=False)

heart_df['Sex'].value_counts()

heart_df['FastingBS'].value_counts()

heart_df['RestingECG'].value_counts()

heart_df['ExerciseAngina'].value_counts()

heart_df['ST_Slope'].value_counts()

heart_df['HeartDisease'].value_counts()


# ### Data Visualization

heart_df.corr()

# Plot showing the corelation matrix
plt.figure(figsize=(15, 5))
sns.heatmap(heart_df.corr(), annot=True, cmap='YlGnBu')
plt.show()

# Scatter Plot Matrix
sns.pairplot(hue='HeartDisease', data=heart_df)

# Age distribition
sns.displot(heart_df['Age'], kde=True)

# ploting numerical features with target
Numerical = heart_df.select_dtypes(include=['int64', 'float64'])
for i in Numerical:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=i, data=heart_df, hue='HeartDisease')
    plt.legend(['Normal', 'Heart Disease'])
    plt.title(i)
    plt.show()

# ploting categorical features with target
Categorical = heart_df.select_dtypes(include=['object'])
for i in Categorical:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=i, data=heart_df, hue='HeartDisease', edgecolor='black')
    plt.legend(['Normal', 'Heart Disease'])
    plt.title(i)
    plt.show()

# distribution plot of Age for HeartDisease
plt.figure(figsize=(15, 5))
sns.distplot(heart_df['Age'][heart_df['HeartDisease'] == 1], kde=True, color='red', label='Heart Disease')
sns.distplot(heart_df['Age'][heart_df['HeartDisease'] == 0], kde=True, color='green', label='Normal')
plt.legend()


# ### Data Preprocessing & Features Engineering


heart_df.info()

Categorical = heart_df.select_dtypes(include=['object'])
Categorical.head()

Numerical = heart_df.select_dtypes(include=['int64', 'float64'])
Numerical.head()

# Using Label Encoder from Scikit-Learn to encode the categorical features

le = LabelEncoder()
Categorical['Sex_Encoded'] = le.fit_transform(Categorical['Sex'])
Categorical['ChestPainType_Encoded'] = le.fit_transform(Categorical['ChestPainType'])
Categorical['RestingECG_Encoded'] = le.fit_transform(Categorical['RestingECG'])
Categorical['ExerciseAngina_Encoded'] = le.fit_transform(Categorical['ExerciseAngina'])
Categorical['ST_Slope_Encoded'] = le.fit_transform(Categorical['ST_Slope'])
Categorical.head()

# Drop the uncoded columns from the categorical data
categorical_data = Categorical.drop(columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
                                    axis=1)
categorical_data.head()

# Rename the catecorical_data columns
categorical_data = categorical_data.rename(columns={'Sex_Encoded': 'Sex', 'ChestPainType_Encoded': 'ChestPainType',
                                                    'RestingECG_Encoded': 'RestingECG',
                                                    'ExerciseAngina_Encoded': 'ExerciseAngina',
                                                    'ST_Slope_Encoded': 'ST_Slope'})
categorical_data.head()

Numerical.head(10)

# Merging the categorical and numerical data
df = pd.concat([categorical_data, Numerical], axis=1)
df.head(10)

# Split the target column from the rest of the columns
X = df.drop(['HeartDisease'], axis=1)
y = df['HeartDisease']


# Building the model

y.value_counts()

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Feature Importance

fi = ExtraTreesClassifier()
fi_train = fi.fit(X_train, y_train)

df_fi_train = pd.DataFrame(fi_train, index=X_train.columns)
df_fi_train.nlargest(10, df_fi_train.columns).plot(kind='barh')
plt.show()

fi_test = fi.fit(X_test, y_test)

df_fi_test = pd.DataFrame(fi_test, index=X_test.columns)
df_fi_test.nlargest(10, df_fi_test.columns).plot(kind='barh')
plt.show()

X_train.head(10)

# Apply algorithm 
# Models to be tested: Logistic Regression, KNearest, SVM, Decision Tree, Random Forest, XGBoost, LightGBM

classifiers = {
    "LogisticRegression": LogisticRegression(),
    "KNeighbors": KNeighborsClassifier(),
    "SVC": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}


# Compute the training score of each models
train_scores = []
test_scores = []
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    train_score = round(classifier.score(X_train, y_train), 2)
    train_scores.append(train_score)
    test_score = round(classifier.score(X_test, y_test), 2)
    test_scores.append(test_score)
print(train_scores)
print(test_scores)

#
train_cross_scores = []
test_cross_scores = []
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    train_score = cross_val_score(classifier, X_train, y_train, cv=5)
    train_cross_scores.append(round(train_score.mean(), 2))
    test_score = cross_val_score(classifier, X_test, y_test, cv=5)
    test_cross_scores.append(round(test_score.mean(), 2))

print(train_cross_scores)
print(test_cross_scores)

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = cross_val_predict(rf, X_test, y_test, cv=5)
print(roc_auc_score(y_test, rf_pred))

#
fpr, tpr, _ = roc_curve(y_test, rf_pred)
plt.plot(fpr, tpr)
plt.show()


# Hyper parameter Tuning

# Logistic Regression
lr = LogisticRegression()
lr_params = {"penalty": ['l2'], "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_logistic = GridSearchCV(lr, lr_params)
grid_logistic.fit(X_train, y_train)
lr_best = grid_logistic.best_estimator_
print(lr_best)

# KNearest Neighbors
knear = KNeighborsClassifier()
knear_params = {"n_neighbors": list(range(2, 7, 1)), "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brutle']}
grid_knear = GridSearchCV(knear, knear_params)
grid_knear.fit(X_train, y_train)
knear_best = grid_knear.best_estimator_
print(knear_best)

# SVC

svc = SVC()
svc_params = {"C": [0.5, 0.7, 0.9, 1], "kernel": ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(svc, svc_params)
grid_svc.fit(X_train, y_train)
svc_best = grid_svc.best_estimator_
print(svc_best)

# Decision Tree

tree = DecisionTreeClassifier()
tree_params = {"criterion": ['gini', 'entropy'], "max_depth": list(range(2, 5, 1)),
               "min_samples_leaf": list(range(5, 7, 1))}
grid_tree = GridSearchCV(tree, tree_params)
grid_tree.fit(X_train, y_train)
tree_best = grid_tree.best_estimator_
print(tree_best)

# Using Xgboost to train the model

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_train_score = cross_val_score(xgb_model, X_train, y_train, cv=5)
xgb_test_score = cross_val_score(xgb_model, X_test, y_test, cv=5)

print(round(xgb_train_score.mean(), 2))
print(round(xgb_test_score.mean(), 2))

# Using lightgbm to train the model

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_train_score = cross_val_score(lgb_model, X_train, y_train, cv=5)
lgb_test_score = cross_val_score(lgb_model, X_test, y_test, cv=5)

print(round(lgb_train_score.mean(), 2))
print(round(lgb_test_score.mean(), 2))

# Save the model
pk.dump(lgb_model, open("heart_disease.pkl", "wb"))

# Load a saved model
loaded_pickle_model = pk.load(open("heart_disease.pkl", "rb"))

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2), 
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict

# Make predictions and evaluate the loaded model
h_disease_pred = loaded_pickle_model.predict(X_test)
evaluate_preds(y_test, h_disease_pred)

# Create the app object
app = FastAPI()

# Load trained Pipeline
model = load_model('heart_disease.pkl')

# Define predict function

@app.post('/predict')
def predict(sex, chestpaintype, restingecg,	exerciseangina,	st_slope, age,	restingbp,	cholesterol,
            fastingbs,	maxhr,	oldpeak):
    data = pd.DataFrame([[sex,	chestpaintype,	restingecg,	exerciseangina,	st_slope,	age,	restingbp,
                          cholesterol,	fastingbs,	maxhr,	oldpeak]])
    data.columns = ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'st_slope', 'age', 'restingbp',
                    'cholesterol', 'fastingbs', 'maxhr', 'oldpeak']

    predictions = predict_model(model, data=data)
    return {'prediction': int(predictions['Label'][0])}
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
