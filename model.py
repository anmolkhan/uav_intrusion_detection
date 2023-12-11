#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics





df= pd.read_csv('./Dataset_T-ITS.csv')



df= pd.read_csv('./Dataset_T-ITS.csv', low_memory=False)



df.head()



df.tail()



df.info()



df.describe()



df.shape



df.columns



df.count()



df['class'].value_counts()



df.isnull().sum()



df=df.fillna(method="ffill")



df.isnull().sum()



df['class'].unique()



df['class']= df['class'].str.replace("class" , "benign")



df['class'].unique()



df['class'].dtypes



df.isna().any()



df.duplicated()



numeric_cols = ['timestamp_c','frame.number','frame.len','frame.protocols','wlan.duration','wlan.ra','wlan.ta','wlan.da','wlan.sa','wlan.bssid','wlan.frag','wlan.seq','llc.type','ip.hdr_len','ip.len','ip.id','ip.flags','ip.ttl','ip.proto','ip.src','ip.dst','tcp.srcport','tcp.dstport','tcp.seq_raw','tcp.ack_raw','tcp.hdr_len','tcp.flags','tcp.window_size','tcp.options','udp.srcport','udp.dstport','udp.length','data.data','data.len','wlan.fc.type','wlan.fc.subtype','time_since_last_packet']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')



df.count()



df.count()



X = df.drop('class', axis='columns')
y = df['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)



X_train



X_test



y_train



y_test



clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



print("Test Result :",clf.score(X_test,y_test))
print("Train Result :",clf.score(X_train,y_train))
