#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read the dataset
df1 = pd.read_csv('data/F1_benign.csv', low_memory=False)
df2 = pd.read_csv('data/F2_DOS.csv', low_memory=False)
df3 = pd.read_csv('data/F3_Replay.csv', low_memory=False)
df4 = pd.read_csv('data/F4_EvilTwin.csv', low_memory=False)
df5 = pd.read_csv('data/F5_FDI.csv', low_memory=False)

# Data exploration
p123 = pd.concat([df1, df2, df3], ignore_index=True)
p45 = pd.concat([df5, df4], ignore_index=True)

remove_columns_123 = [s for s in p123.columns if s not in p45.columns]
remove_columns_45 = [s for s in p45.columns if s not in p123.columns]
remove_columns = remove_columns_45 + remove_columns_123

for col in p123.columns:
    if col in remove_columns:
        p123.pop(col)


for col in p45.columns:
    if col in remove_columns:
        p45.pop(col)

df = pd.concat([p123, p45], ignore_index=True)

# Fill missing values
df = df.fillna(method="ffill")
print(df.isnull().sum())

# Data cleaning and transformation
df['class'] = df['class'].str.replace("class", "benign")
print(df['class'].unique())
print(df['class'].dtypes)
print(df.isna().any())
print(df.duplicated())

# Split data into train and test sets
X = df.drop('class', axis='columns')
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training and evaluation
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Predict on test dataset
y_pred = clf.predict(X_test)

# Model evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Test Result:", clf.score(X_test, y_test))
print("Train Result:", clf.score(X_train, y_train))
