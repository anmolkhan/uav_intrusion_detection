#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import svm


# In[2]:


df= pd.read_csv('D:/DataScience/Statistice and Probablity/CW_1/CW1_Submission_4237096/Dataset_T-ITS.csv')


# In[3]:


df= pd.read_csv('D:/DataScience/Statistice and Probablity/CW_1/CW1_Submission_4237096/Dataset_T-ITS.csv', low_memory=False)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


df.columns


# In[10]:


df.count()


# In[11]:


df['class'].value_counts()


# In[12]:


df.isnull().sum()


# In[13]:


df=df.fillna(method="ffill")


# In[14]:


df.isnull().sum()


# In[15]:


df['class'].unique()


# In[16]:


df['class']= df['class'].str.replace("class" , "benign")


# In[17]:


df['class'].unique()


# In[18]:


df['class'].dtypes


# In[19]:


df.isna().any()


# In[20]:


df.duplicated()


# In[37]:


numeric_cols = ['timestamp_c','frame.number','frame.len','frame.protocols','wlan.duration','wlan.ra','wlan.ta','wlan.da','wlan.sa','wlan.bssid','wlan.frag','wlan.seq','llc.type','ip.hdr_len','ip.len','ip.id','ip.flags','ip.ttl','ip.proto','ip.src','ip.dst','tcp.srcport','tcp.dstport','tcp.seq_raw','tcp.ack_raw','tcp.hdr_len','tcp.flags','tcp.window_size','tcp.options','udp.srcport','udp.dstport','udp.length','data.data','data.len','wlan.fc.type','wlan.fc.subtype','time_since_last_packet']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')


# In[38]:


df.count()


# In[40]:


df.count()


# In[48]:


X = df.drop('class', axis='columns')
y = df['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[49]:


X_train


# In[50]:


X_test


# In[51]:


y_train


# In[52]:


y_test


# In[54]:


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[55]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[56]:


print("Test Result :",clf.score(X_test,y_test))
print("Train Result :",clf.score(X_train,y_train))


# In[58]:





# In[ ]:




