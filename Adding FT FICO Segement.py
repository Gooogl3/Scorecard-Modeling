#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
from datetime import datetime

# Files for upload
path1 = r'\\neptune\RAD\6 Audit\Scorecard 7.0 Score Audit\20220121\SCORECARD_01-21-22.csv'
#path2 = 


df1 = pd.read_csv(path1)


# In[10]:


df1.head()


# In[11]:


# Filling NaN with 0
df1['BYR1_FICO'] = df1['BYR1_FICO'].fillna(0)


# In[22]:


df1['B1_FACTORTRUST_HIT'].dtypes


# In[13]:


# Defining Conditions
conditions = [
    (df1['BYR1_FICO'] == 0),
    (df1['BYR1_FICO'] >= 9001) & (df1['BYR1_FICO'] <= 9003),
    (df1['BYR1_FICO'] >= 250) & (df1['BYR1_FICO'] <= 900) & (df1['B1_FACTORTRUST_HIT'] == 0),
    (df1['BYR1_FICO'] >= 250) & (df1['BYR1_FICO'] <= 900) & (df1['B1_FACTORTRUST_HIT'] == 1),
    ]
# Defining Values for the Conditions
values = ['Missing', '9000s', 'Valid FT NO Hit','Valid FT Hit']

# Creating a new column with the conditions and values
df1['Fico_Seg'] = np.select(conditions, values)


# In[14]:


df1.head()


# In[16]:


# Testing
df2 = df1[['BYR1_FICO', 'B1_FACTORTRUST_HIT']]
df2.head()


# In[17]:


conditions = [
    (df2['BYR1_FICO'] == 0),
    (df2['BYR1_FICO'] >= 9001) & (df2['BYR1_FICO'] <= 9003),
    (df2['BYR1_FICO'] >= 250) & (df2['BYR1_FICO'] <= 900) & (df2['B1_FACTORTRUST_HIT'] == 0),
    (df2['BYR1_FICO'] >= 250) & (df2['BYR1_FICO'] <= 900) & (df2['B1_FACTORTRUST_HIT'] == 1),
    ]
values = ['Missing', '9000s', 'Valid FT NO Hit','Valid FT Hit']
df2['Fico_Seg'] = np.select(conditions, values)


# In[18]:


df2.head()

