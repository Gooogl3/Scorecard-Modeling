#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Function Definition. 

def roc_calc(df, target_name, *score_name):
    import pandas as pd
    import numpy as np
    
    df[target_name] = df[target_name].str.upper()
    df = df[(df[target_name] == 'GOOD') | (df[target_name] == 'BAD')]
    
    result = {}
    for score in score_name:
        piv = df.pivot_table(df,index=score,columns=target_name,aggfunc='size',fill_value=0)
        piv = piv.sort_index()
        if piv.index.max() <= 1:     # if it's second stage bad rate, sort from high to low
            piv = piv.sort_index(ascending=False)
        else:                        # if it's first stage score, sort from low to high
            piv = piv.sort_index()
        piv['Bad%'] = (piv.BAD.cumsum(axis=0))/(piv.BAD.sum())
        piv['Good%'] = (piv.GOOD.cumsum(axis=0))/(piv.GOOD.sum())
        roc = (piv['Good%'] - piv['Good%'].shift(1)) * (piv['Bad%'] + piv['Bad%'].shift(1))/2   
        roc = sum(roc.replace(np.nan,0))        
        result[score] = roc
    result = pd.Series(result)
    result.name = 'ROC'
    return result
    
    
def roc_plot(df, target_name,Title, *score_name):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    df[target_name] = df[target_name].str.upper()
    df = df[(df[target_name] == 'GOOD') | (df[target_name] == 'BAD')]
    
    fig = plt.figure(figsize=(6,6))
    
    roc_lst = []
    
    for score in score_name:
        piv = df.pivot_table(df,index=score,columns=target_name,aggfunc='size',fill_value=0)
        piv = piv.sort_index(ascending=False)
        if piv.index.max() <= 1:     # if it's second stage bad rate, sort from high to low
            piv = piv.sort_index(ascending=False)
        else:                        # if it's first stage score, sort from low to high
            piv = piv.sort_index()
        piv['Bad%'] = (piv.BAD.cumsum(axis=0))/(piv.BAD.sum())
        piv['Good%'] = (piv.GOOD.cumsum(axis=0))/(piv.GOOD.sum())
        roc = (piv['Good%'] - piv['Good%'].shift(1)) * (piv['Bad%'] + piv['Bad%'].shift(1))/2   
        roc = sum(roc.replace(np.nan,0)) *100  
        roc_lst.append(roc)
        
        if len(roc_lst) == 1:
            plt.plot(piv['Good%'], piv['Bad%'], lw=2, label = "{}: {:.2f}%, Base".format(score, roc))
        else:
            lift = 100* (roc - roc_lst[0])/roc_lst[0]
            plt.plot(piv['Good%'], piv['Bad%'], lw=2, label = "{}: {:.2f}%, Lift={:.2f}%".format(score, roc,lift))
        
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('Cumulative % of Goods')
    plt.ylabel('Cumulative % of Bads')
    plt.grid()
    plt.title(Title)
    plt.legend(prop={'size':13}, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    
def lift(base_roc, new_roc):
    return (new_roc-base_roc)/base_roc

# Set up
import pandas as pd
import numpy as np


# In[36]:


# Put in the path of the data file and column names inside of the apostrophe
path = r'\\neptune\RAD\4 Models\Scorecard 8.0_Redesign\Temp\JN\I. Limited Credit\Stage 2.xlsx'
target = 'TARGET_GB_NEW'
score1 = 'SCORECARD_POINTS'
score2 = 'Stage 2'
score3 = 'Stage 2 - Bad' 
#score4 = 'SC70_Stage2_Points'
#score5 = ...   # add more if you need

# Load the data file
df = pd.read_excel(path)
df.head()


# In[39]:


roc_plot(df, target,'Segment I - Stage 1', score1, score3)


# In[28]:


roc_plot(df, target,'Segment I - Stage 2', score2,)


# In[5]:


df_CA = df[df['Buyer 1#s State'] == 'CA']
roc_plot(df_CA, target, 'Missing 000 SSN NO LN - CA',score1)


# In[6]:


df_TX = df[df['Buyer 1#s State'] == 'TX']
roc_plot(df_TX, target,'Missing 000 SSN NO LN - TX', score1)


# In[10]:


#States = ['FL','CA','TX']
#df2 = df[ (df['Fee'] >= 22000) & (df['Discount'] == 2300)]

df_exclude_Major_States= df[(df['Buyer 1#s State'] != 'CA') & (df['Buyer 1#s State'] != 'FL') & (df['Buyer 1#s State'] != 'TX')]

roc_plot(df_exclude_Major_States, target,'Missing 000 SSN NO LN - Other States', score1)


# In[9]:


df_FL_TX = df[ (df['Buyer 1#s State'] == 'FL') | (df['Buyer 1#s State'] == 'TX')]

roc_plot(df_FL_TX, target,'Missing 000 SSN NO LN - FL & TX', score1)

