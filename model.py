#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read the dataset
df = pd.read_csv('./Dataset_T-ITS.csv', low_memory=False)

# Data exploration
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.shape)
print(df.columns)
print(df.count())
print(df['class'].value_counts())
print(df.isnull().sum())

# Fill missing values
df = df.fillna(method="ffill")
print(df.isnull().sum())

# Data cleaning and transformation
df['class'] = df['class'].str.replace("class", "benign")
print(df['class'].unique())
print(df['class'].dtypes)
print(df.isna().any())
print(df.duplicated())

# Convert columns to numeric
numeric_cols = ['timestamp_c', 'frame.number', 'frame.len', 'frame.protocols', 'wlan.duration', 'wlan.ra', 'wlan.ta', 'wlan.da', 'wlan.sa', 'wlan.bssid', 'wlan.frag', 'wlan.seq', 'llc.type', 'ip.hdr_len', 'ip.len', 'ip.id', 'ip.flags', 'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'tcp.seq_raw', 'tcp.ack_raw', 'tcp.hdr_len', 'tcp.flags', 'tcp.window_size', 'tcp.options', 'udp.srcport', 'udp.dstport', 'udp.length', 'data.data', 'data.len', 'wlan.fc.type', 'wlan.fc.subtype', 'time_since_last_packet']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
print(df.count())

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
