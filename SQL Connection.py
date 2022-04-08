#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import os
import pyodbc 


# In[7]:


# Setting up a Connection
con = pyodbc.connect('Trusted_Connection=yes',
                     driver = '{SQL Server}',
                     server = 'sqlrad', 
                     database = 'MainDatabase_Current')

cur = con.cursor()


# In[8]:


# Writing the SQL query
SC80_Final_Results_query = "SELECT [LN_Key2]      ,[TARGET_GB]      ,[SC80_Stage1]      ,[SC80_Segment]      ,[Buyer1_State]      ,[Accepts_Rejects]      ,[DataSource]  FROM [SC80_Modeling].[dbo].[SC80_Final_Results]"

Vantage_query = "SELECT [LN_Key2]      ,[FICCLAV8_SCORE]      ,[VANTAGE_V4_SCORE]  FROM [SC80_Modeling].[dbo].[SC80_Premier_AB_1819_Imputed]"


LN_2_DealNo_query = "SELECT [DL_MASTER]      ,[LN_Key2]  FROM [SC80_Modeling].[dbo].[mb_dev]"


# In[9]:


# Creating Tables based off of Query
cur.execute(SC80_Final_Results_query)
results_after = cur.fetchall()

# SQL Tables
SC80_Final_Results = pd.read_sql(SC80_Final_Results_query, con)
vantage = pd.read_sql(Vantage_query, con)


# In[10]:


SC80_Final_Results.head()

