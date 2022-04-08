#!/usr/bin/env python
# coding: utf-8

# In[6]:


# import packages
import pandas as pd #this for importing/exporting data and creating dataframes
import numpy as np #this is for scientific computation. 
import copy #this allows to copy
import scipy.stats.stats as stats #this is probability distributions and a library of statistical functions
from sklearn.model_selection import train_test_split #this is to split the data into train and test (validation)
from sklearn.linear_model import LogisticRegression #this is logistic regression
from sklearn.metrics import roc_auc_score #to calculate the ROC
import matplotlib.pyplot as plt #this is for the graph
from sklearn.tree import DecisionTreeClassifier #this is for decision tree
from sklearn import tree #this is for decision tree too
pd.options.mode.chained_assignment = None  # default='warn' #this is for hide warm


# In[19]:


# import data
db = pd.read_csv(r"\\neptune\RAD\4 Models\Scorecard 8.0\Modeling Data\Missing\CA\Accepts_2600.csv") #imports the accepted file


# In[20]:


# frequency
Count_Bad = db.groupby("TARGET_GB").count().loc['BAD','LNKEY']
Count_Good_INDET = db.groupby("TARGET_GB").count().loc[['GOOD','INDET'],'LNKEY'].sum()
if Count_Good_INDET > Count_Bad:
    freq_dic = {'GOOD':Count_Good_INDET/Count_Bad,'BAD':1}
else:
    freq_dic = {'GOOD':1,'BAD':Count_Bad/Count_Good_INDET}
freq_dic


# In[21]:


def Add_Frequency(row,freq_dic):
    if row["TARGET_GB"] == 'BAD':
        return freq_dic['BAD']
    else:
        return freq_dic['GOOD']

db['Frequency'] = db.apply(lambda row : Add_Frequency(row,freq_dic), axis=1)
db[['TARGET_GB','Frequency']].head(5)


# In[22]:


# filter Good and Bad
Goods=db[db["TARGET_GB"]=="GOOD"]  #we use two db because one db will give the count of the whole dataset
Bads=db[db["TARGET_GB"]=="BAD"]
print("Good:",len(Goods), "Bad:",len(Bads))


# In[20]:


#sample down
if len(Goods) >= len(Bads):
    Bad = Bads
    Good = Goods.sample(len(Bad),random_state =2602)
    print("Good:",len(Good),"Bad:",len(Bad))
else:
    Good = Goods
    Bad = Bads.sample(len(Good),random_state=2602)
    print("Good:",len(Good),"Bad:",len(Bad))


# In[21]:


# append Good and Bad
GB=Good.append(Bad) #can do Good.append(Bad)
print("append Good and Bad:",len(GB)) #GB #will give the dataset that has the good and bad appended


# In[22]:


#not included in code
data_type = GB.dtypes #gives the datatype for each variable
#data_type #prints the list of the variable with the datatype


# In[23]:


# all categorical variable and unnecessary columns
a = GB.dtypes[GB.dtypes=='object'].index.tolist()
b = GB.dtypes[GB.dtypes=='datetime64[ns]'].index.tolist()
a.extend(b)
a.extend(['OPENINGBALANCE_PCT','Contract_Open_Date','SCOREX','BY2_SCORE','NET_AFPCT','BYRAGE','ANNUAL_RATE','NAMTF_LFC'          ,'LOT_ID','CHECK_DATE','X_BOOK','RECOURSE','DISC_PC' ,'BY1_SCORE','PTI','CASH_DOWNPCT'          ,'INVOICE_BOOK_VALUE','NADA_VALUE','VHCL_YEAR','ODO','GROSS_MONTH','DOWN','PAYMENT','Buyer_1_MOSATWORK'          ,'Buyer_2_MOSATWORK','CTRL_ZIP','Advance','BYR1_FICO','AMTF_LFC','DECRULESET','CAR_AGE','AMTF_LFC_PCT_Risk'          ,'AUTOMOBILE_PRICE','ACV_VALUE','Valid_SSN_Indicator','CLOSE_DATE','CLOSE_DATE','GAIN','DR_ORIGBAL'          ,'DAYS_LATE','PAIDMO','TERM','FIRST_PMT_PAYMENT','LTVPCT_XBOOK','APR_ANNUAL_RATE','AMOUNT_FINANCED'          ,'BAD_CLOSECODE_Indicator','OPEN_YEARMONTH','CLOSE_YEARMONTH','LNKey','Frequency','SQL_LTV'         ,'SC60_Stage1','SC60_Stage2','SQL_CASHDOWNPCT','SQL_OPENINGBALANCE_PCT','SQL_PTI','Client Data','VANTAGE_V4_SCORE'         ,'IQF9416','IQF9417', 'IQF9410', 'IQF9415','IQF9510','IQA9510','IQF9540','TRADE_DATE'         ,'RSS2005_0_2','RVS2005_0_4','pd4_ad','telecommunications_score','bankcard_score','auto_score','short_term_lending_score'         ,'LL_BAL_TO_CREDIT','Days_Since_SF_LAST_APPL_RCVD','LL_SATISFACTORY','SL_DIFF_HOMEPHONE_EVER','ins5_online_ft_fis','RECORD_NB'])#extends the above "a" list to include other columns that we do not need 
a #a #will print the list "a" that includes all the columns that are categorical and that are not needed for the model


# In[24]:


# drop unnecessary columns
GB_for_group = GB.drop(a,axis=1) #drops the list from GB which is defined in the append of good and bad. axis =1 is the columns
#GB_for_group #should give the new number of columns after dropping the variables in the list "a"


# In[25]:


#create the frequency table by grabbing the frequency column from the table GB. This should be the same number as the append
frequency = GB['Frequency']
#frequency #will give the freq for each row in the GB table


# In[26]:


#data partition
y = pd.get_dummies(GB["TARGET_GB"]) #converts categorical variable (target) into dummy/indicator variables such as zero and one. The G_B becomes two columns (Good and Bad)
y_Good = y["GOOD"] #target. Good is 1 and bad is 0
X_train, X_test, y_train, y_test = train_test_split(GB_for_group, y_Good, random_state = 12345, test_size=0.2) #2609
#X_train, X_test, y_train, y_test = train_test_split(all columns except target G_B, target column, random_state (random seed), test_size (or can use train_size))
#X_train #(training data)
#y_train #(training data target variable only)
#X_test #(test data)
#y_test #(test data target variable only)


# In[27]:


# create quartile_cut function
def quartile_cut(temp_not_missing, frequency, m):
    t_name = temp_not_missing.name #column name
    a = temp_not_missing.copy() #copy the dataframe. Note this "a" is not the same "a" from above
    a = pd.concat([a,frequency],join='inner',axis=1) #combine the column with the frequency column
    a = a.sort_values(by = t_name) #sorts the column name from smallest to largest
    a['cumulative sum'] = a[frequency.name].cumsum() #calculates the cumulative sum of the frequency
    interval = (a['cumulative sum'].max()+0.05)/m #find the max of the cumulative sum and adds 0.5 to it and then divides it by m which is the number of buckets
    b = {'Bucket_#':[],'Bucket_max':[]} #emtpy lists
    for i in range(1,m+1): #the list will only start at 1 and end at m. It will not include m+1
        b['Bucket_#'].append(i) #b will keep getting appended with just the values of i. 
        b['Bucket_max'].append(a.loc[a['cumulative sum'] <= i*interval, t_name].max()) #this will check the max for each interval. Example, interval 1 for FICO can only go up to FICO score 418
    c = pd.DataFrame(b) #creates the data fram with bucket number and the bucket max for each bucket number
    c['Bucket_max'] = c['Bucket_max'].fillna(a[t_name].min()) #if the row is blank for a column then just include it in the minimum bin
    temp_not_missing = pd.concat([temp_not_missing,frequency],join='inner',axis=1) #combines column name with frequency
    temp_not_missing = temp_not_missing.reset_index() #reset the index of the dataframe 
    temp_not_missing = temp_not_missing.sort_values(by = t_name) #sorts it by small to large by column name
    temp_not_missing[t_name] = temp_not_missing[t_name].astype('float64') #they all have the same data type
    c['Bucket_max'] = c['Bucket_max'].astype('float64') #they all have the same data type
    if len(temp_not_missing) == 0: #if temp_not_missing includes nothing print variable name includes Nan only
        print(t_name + ' are all Nan')
    else: 
        d = pd.merge_asof(temp_not_missing,         c.sort_values('Bucket_max'),         left_on = t_name, right_on = 'Bucket_max',direction = 'forward').set_index('index')
    return d['Bucket_#']


# In[28]:


# interactive grouping with decision tree(*means new)
def interactive_grouping2(X_train, y_train, frequency,n=5, reject_level=0.02): #(n is the maximum number of bins)
    X_train['useless_column'] = 1 #add a useless_column that will be used in decision tree*
    columns_name = X_train.columns  #gives the column name for the training data only
    y_name = y_train.name #gives the column name "GOOD". This variable is found in the data partition code
    freq_name = frequency.name #gives the column name "Frequency".
    group_dictionary={} #creates an empty dictionary
    IV_dictionary = {} #creates an empty dictionary
    for i in range(len(columns_name)-1):
        temp_not_missing = X_train.loc[X_train[columns_name[i]].notnull(), [columns_name[i],'useless_column']] #makes the columns no longer null
        not_missing_with_y = pd.concat([temp_not_missing,y_train],join='inner',axis=1) #add a new column (the train target variable) and if good then 1 or 0 if bad
        not_missing_with_y_fre = pd.concat([not_missing_with_y,frequency],join='inner',axis=1) #add a new column (frequency)
        not_missing_with_y_fre["freq*Good"] = not_missing_with_y_fre[freq_name]*not_missing_with_y_fre[y_name] #multiplies  frequency by the value of good (either 1 or 0). for calculating Good event rate 
        temp_missing = X_train.loc[X_train[columns_name[i]].isnull(), columns_name[i]] #split rows with Nan value out
        missing_with_y = pd.concat([temp_missing,y_train],join='inner',axis=1)
        missing_with_y_fre = pd.concat([missing_with_y,frequency],join='inner',axis=1)
        missing_with_y_fre["freq*Good"] = missing_with_y_fre[freq_name]*missing_with_y_fre[y_name]
        r = 0 #this is to reset the while loop when starting a variable i.e it will do FICO then in order to do cc2_ad we want everything to reset
        p = 0 #this is to reset the while loop when starting a variable i.e it will do FICO then in order to do cc2_ad we want everything to reset
        u = 0 #this is to reset the while loop when starting a variable i.e it will do FICO then in order to do cc2_ad we want everything to reset
        m = n # this is to reset m
    # X = temp_missing, monotonic event rate grouping
        if len(not_missing_with_y_fre) > 0: #if not_missing_with_y_fre has record
            while (np.abs(r) < 0.9 or u <= 100) and m >= 1: #stop until (spearman correlation great than 0.9 and smallest bin has records more than 150) or m = 0
                not_missing_with_y_fre["Bucket1"] = quartile_cut(not_missing_with_y_fre[columns_name[i]],frequency, 20) #quartile function from above. This will create a new variable called Bucket which is the bucket interval from quartile cut
                if m > 1:
                    clf = tree.DecisionTreeClassifier(random_state = 0, max_leaf_nodes = m, min_weight_fraction_leaf = 0.045) # *set decision tree parameters (max_leaf_nodes: max number of bins,min_weight_fraction_leaf: min sample % for each bins)*
                    clf = clf.fit(not_missing_with_y_fre[["Bucket1",'useless_column']], not_missing_with_y_fre[y_name], not_missing_with_y_fre[freq_name].values) # *fit not_missing_with_y_fre*
                    not_missing_with_y_fre["Bucket"] = clf.apply(not_missing_with_y_fre[["Bucket1",'useless_column']]) # *find bin number for each records*
                    d2 = not_missing_with_y_fre.groupby('Bucket', as_index = True) #group by function. group by bucket from above
                    r, p = stats.spearmanr(d2[columns_name[i]].mean(), d2["freq*Good"].sum()/d2[freq_name].sum()) #find the average for the column name in the training data, sum of the good event rate and divides it by the sum of the frequency then calculate spearman correlation
                    u = d2[y_name].count().min() #returns the count and minimum of the column that gives the output good.
                    m = m - 1
                else: #if not_missing_with_y_fre doesn't have record
                    not_missing_with_y_fre["Bucket"] = 1
                    d2 = not_missing_with_y_fre.groupby('Bucket', as_index = True)
                    m = 0
            d3 = pd.DataFrame(d2[columns_name[i]].min()) #creates a table
            d3 = d3.rename(columns={columns_name[i]:'min_' + columns_name[i]}) #renames the column in above to mini_column name
            d3['max_' + columns_name[i]] = d2[columns_name[i]].max() #creates the "max_variable name"
            d3[y_name] = d2["freq*Good"].sum() #sum of the good event rate
            d3['total'] = d2[freq_name].sum() #create column called total which is the sum of the frequency 
            d3[y_name + '_rate'] = d3[y_name]/d3['total'] #creates a column for the good rate which is the count of good divided by the total for each interval
            d4 = (d3.sort_values(by = 'min_' + columns_name[i])).reset_index(drop = True) #sorts the "min_variable name" column
            d4 = d4.dropna(subset=['min_' + columns_name[i]]) #drop row that have Nan for min_columns
            d4 = d4.append({y_name:missing_with_y_fre["freq*Good"].sum(), 'total':missing_with_y_fre[freq_name].sum()                 , y_name + '_rate':missing_with_y_fre["freq*Good"].sum()/missing_with_y_fre[freq_name].sum()},ignore_index=True)
            d4 = d4.dropna(subset=[y_name + '_rate'])
        else:
            print(columns_name[i] + ' includes Nan only')
            d4 = pd.DataFrame({y_name:[missing_with_y_fre["freq*Good"].sum()], 'total':[missing_with_y_fre[freq_name].sum()]                         , y_name + '_rate':[missing_with_y_fre["freq*Good"].sum()/missing_with_y_fre[freq_name].sum()]})
        
    # WOE, varified in excel
        Total_Event = d4[y_name].sum() #sum of the event rate
        Total_Nonevent = d4['total'].sum() - Total_Event #sum of the nonevent rate
        d4['WOE'] = d4.apply(lambda x: np.log(((x[y_name]+0.5)/Total_Event)/((x['total']-x[y_name]+0.5)/Total_Nonevent)),axis=1) #WOE calculated
        d4 = d4.dropna(subset=['WOE']) #drop row that have Nan for WOE
        group_dictionary[columns_name[i]] = d4
    # Information Value, varified in excel
        IV_dictionary[columns_name[i]] = d4.apply(lambda x: ((x[y_name]/Total_Event)-((x['total']-x[y_name])/Total_Nonevent))*x['WOE'],axis=1).sum() #information value calculated
    IV_table = pd.DataFrame(list(IV_dictionary.items()),columns=['Variable_Name','Information_Value']).sort_values('Information_Value', ascending=False) #creates table with variable name and the information value. It is ordered by large to small information value
    # reject weak variables
    IV_table["Reject_or_Accept"] = IV_table.apply(lambda x: "Accept" if x['Information_Value']>=reject_level else "reject",axis=1) #mark all the reject variables if the information value is less than reject else mark accept
    # remove rejected variables from train
    Accept_List = IV_table.loc[IV_table["Reject_or_Accept"] == "Accept", 'Variable_Name'].values.tolist() #creates a list of accepted variables
    X = X_train[Accept_List] #training data for the accepted variables
    
    # convert real vaule to WOE
    #X1 = X.fillna(X.min()-1).reset_index()
    #columns = X.columns
    #converted_X = X.copy()
    #for i in range(len(columns)):   
        #X1[columns[i]] = X1[columns[i]].astype('float64')
        #X2 = pd.merge_asof(X1.sort_values(columns[i]), \
                        #group_dictionary[columns[i]].fillna(group_dictionary[columns[i]].min()-1).sort_values('min_' + columns[i]), \
                        #left_on = columns[i], right_on = 'min_' + columns[i])[['index','WOE']].set_index('index')
        #converted_X[columns[i]] = X2['WOE']
    return group_dictionary, Accept_List, IV_table


# In[29]:


#convert X_test

def convert_X(X_test, Accept_List, group_dictionary):
    X = X_test[Accept_List] #select variables with information value greater than 0.02
    X1 = X.fillna(X.min()-200000).reset_index() # impute Nan with minimum value-100000
    columns = X.columns #object"columns" incluses all column name
    converted_X = X.copy() # copy X to converted_X
    group_dictionary1 = copy.deepcopy(group_dictionary) # copy group_dictionary to group_dictionary1
    for i in range(len(columns)): #convert original value to WOE based on group_dictionary1
                X1[columns[i]] = X1[columns[i]].astype('float64')
                group_dictionary1[columns[i]].loc[0,['min_' + columns[i]]] = group_dictionary1[columns[i]]['min_' + columns[i]].min()-100000
                X2 = pd.merge_asof(X1.sort_values(columns[i]),                                     group_dictionary1[columns[i]].fillna(group_dictionary1[columns[i]].min()-300000).sort_values('min_' + columns[i]),                                     left_on = columns[i], right_on = 'min_' + columns[i])[['index','WOE']].set_index('index')
                converted_X[columns[i]] = X2['WOE'] #save result to converted_X
    return X, converted_X


# In[30]:


# run the interactive grouping
group_dictionary, Accept_List, IV_table = interactive_grouping2(X_train, y_train,frequency, n=5, reject_level=0.018)


# In[66]:


# regroup function
def regroup( group_dictionary, group_number=[0, 1, 2, 3, 4, 5], variable_name = 'crossindustry_score'):
    pd_Series = pd.Series(group_number) #convert list to series
    t = group_dictionary[variable_name].set_index(pd_Series) # reset the index
    t1 = t.reset_index().groupby('index') # group by index
    t2 = pd.DataFrame(t1['GOOD'].sum()) # sum 'GOOD' by index
    t2['total'] = t1['total'].sum() # sum total by index
    t2['GOOD_rate'] = t2['GOOD']/t2['total'] # recalculate good rate
    Total_Event = group_dictionary[variable_name]['GOOD'].sum() # count total event
    Total_Nonevent = group_dictionary[variable_name]['total'].sum() - Total_Event # count total nonevent
    t2['WOE'] = t2.apply(lambda x: np.log(((x['GOOD']+0.5)/Total_Event)/((x['total']-x['GOOD']+0.5)/Total_Nonevent)),axis=1) #recalculate WOE
    group_dictionary[variable_name] = t.drop(['GOOD_rate','WOE'],axis = 1).join(t2[['GOOD_rate','WOE']], how = 'left').reset_index(drop = True).sort_values(t.columns[0],ascending=True) # cover original GOOD_rate and WOE


# In[67]:


#group example: regroup(group_dictionary, group_number=[0, 1, 2, 3, 4, 0], variable_name = 'crossindustry_score')


# In[68]:


group_dictionary['ALL7350']


# In[69]:


#show information value table
IV_table


# In[70]:


# the count number for Accepting variable
IV_table[IV_table['Reject_or_Accept'] == 'Accept'].count()


# In[35]:


# run the convert X
Original_X_train, Converted_X_train = convert_X(X_train, Accept_List, group_dictionary)
Original_X_test, Converted_X_test = convert_X(X_test, Accept_List, group_dictionary)


# In[36]:


# find the frequency for train data and test data
y_train_freq = pd.concat([y_train,frequency],join='inner',axis=1)['Frequency']
y_test_freq = pd.concat([y_test,frequency],join='inner',axis=1)['Frequency']


# In[37]:


# variable selection by using lasso regression
drop_list = []
N = 15 #number of variables
Test_roc = [] #create a empty list for storing Test ROC
Train_roc = [] #create a empty list for storing Train ROC
Cs = np.logspace(-2, 0, num=30) #(select 20 numbers between 10^-2 and 10^4 )
# for each C pick the N most important variables and use these vairbles to rerun logistic regression. finally, calculate ROC for train and test.
for C in Cs:
    lr = LogisticRegression(penalty='l1',C = C, solver='saga',random_state = 1).fit(Converted_X_train.drop(drop_list,axis = 1), y_train,y_train_freq)
    feature_importances = pd.DataFrame(lr.coef_.T,
                                       index = Converted_X_train.drop(drop_list,axis = 1).columns,
                                        columns=['importance']).sort_values('importance',ascending=False)
    new_X_train=Converted_X_train[feature_importances.iloc[0:N,:].index.tolist()] #select the best N variables and create a new train dataset
    new_X_test=Converted_X_test[feature_importances.iloc[0:N,:].index.tolist()] #select the best N variables and create a new test dataset
    lr = LogisticRegression().fit(new_X_train, y_train,y_train_freq) #run logistic regression
    y_decision_fn_scores_auc = lr.decision_function(new_X_train) # calculate score (just like when we calculate Taprezoid ROC we need socre)
    Train_roc.append(roc_auc_score(y_train, y_decision_fn_scores_auc, sample_weight = y_train_freq)) #calculate ROC and print it out
    y_decision_fn_scores_auc = lr.decision_function(new_X_test)
    Test_roc.append(roc_auc_score(y_test, y_decision_fn_scores_auc, sample_weight = y_test_freq))

# plot train and test ROC (X: Cs the regulation varible, Y: ROC)
ax = plt.gca()
ax.set_xscale('log')
ax.plot(Cs, Train_roc,'g', label = 'Train')
ax.plot(Cs, Test_roc,'b', label = 'Test')
#Set limits and titles
plt.ylim([0.68,0.745]) # Y range
plt.xlabel('Cs')
plt.ylabel('AUC or ROC')
plt.legend()
plt.title('Tune regularization')
 
plt.savefig('Tuning.png')
plt.show()

df = pd.DataFrame([Cs,Train_roc,Test_roc]).T.rename(columns={0:"C",1:"Train", 2:"Test"}) # save result in dataframe
print(df[df["Test"] == df["Test"].max()]) #print out the C with highest Test ROC
print(df) #print out all C, Train ROC score and Test ROC score


# In[38]:


# run logistic regression and print aout ROC and variables we use
lr = LogisticRegression(penalty='l1',C =  0.035622, solver='saga',random_state = 1).fit(Converted_X_train.drop(drop_list,axis = 1), y_train,y_train_freq) #run lasso logistic regression using best C
feature_importances = pd.DataFrame(lr.coef_.T,
                                       index = Converted_X_train.drop(drop_list,axis = 1).columns,
                                        columns=['importance']).sort_values('importance',ascending=False) # create a datafrme to save feature importance
new_X_train=Converted_X_train[feature_importances.iloc[0:N,:].index.tolist()] #select the best N variables and create a new train dataset
new_X_test=Converted_X_test[feature_importances.iloc[0:N,:].index.tolist()] #select the best N variables and create a new test dataset
lr = LogisticRegression().fit(new_X_train, y_train,y_train_freq) #run the logistic
y_decision_fn_scores_auc = lr.decision_function(new_X_train) # calculate decision score (when we calculate Taprezoid ROC we need socre) 
print('Train set AUC: ',roc_auc_score(y_train, y_decision_fn_scores_auc, sample_weight = y_train_freq)) #calculate ROC and print it out
y_decision_fn_scores_auc = lr.decision_function(new_X_test)
print('Test set AUC: ',roc_auc_score(y_test, y_decision_fn_scores_auc, sample_weight = y_test_freq))
selected_feature = feature_importances.iloc[0:N,:].index.tolist() # select the best N variable's name and  and save them in list 
lr_coef = pd.DataFrame(lr.coef_.T, index = new_X_train.columns, columns=['coefficient']).sort_values('coefficient',ascending=False) # create a new dataframe to save variable name and coefficient
print(lr_coef) #print variable coefficient
lr.intercept_[0] #print out intercept


# In[39]:


# calculate score
scorecard_point ={}
for i in lr_coef.index.tolist():
    scorecard_point[i] = group_dictionary[i].copy()
    scorecard_point[i]['score_point'] =  round((scorecard_point[i]['WOE']*lr_coef.loc[i][0]+lr.intercept_[0]/N)*28.8539008+200/N)
scorecard_point['crossindustry_score']


# In[40]:


# create scorecard table
for i,j in zip(lr_coef.index.tolist(),range(len(lr_coef))):
    if j == 0:
        scorecard = scorecard_point[i].copy()
        scorecard = scorecard.rename(columns={'min_' + i:'min','max_'+ i:'max'})
        scorecard['variable'] = i
        scorecard = scorecard.set_index('variable')
        
    else:
        scorecard2 = scorecard_point[i].copy()
        scorecard2 = scorecard2.rename(columns={'min_' + i:'min','max_'+ i:'max'})
        scorecard2['variable'] = i
        scorecard2 = scorecard2.set_index('variable')
        scorecard = pd.concat([scorecard,scorecard2], ignore_index=False)
scorecard_before = scorecard.copy()


# In[41]:


# import rejected dataset
rejected_db = pd.read_csv(r"\\neptune\RAD\4 Models\Scorecard 8.0\Preliminary\Clarity_and_FT\ValidFICO_Clarity_Hit_CA\Rejects_43742.csv")
len(rejected_db)


# In[42]:


# sample rejected dataset down
rejected_sampledown = rejected_db.sample(n=len(GB),random_state=1000)
print("Sample to :",len(rejected_sampledown))


# In[43]:


# prepare data for reject inference
#a.remove('Frequency')
#a.remove('G_B')
for i in ['Frequency','TARGET_GB']:
    a.remove(i)
rejected_droped = rejected_sampledown.drop(a,axis=1)


# In[44]:


rejected_droped['useless_column'] = 1


# In[45]:


# prepare data for reject inference
GB_accept = GB_for_group.copy()
GB_accept['GOOD'] = y['GOOD']
GB_accept['Frequency'] = GB['Frequency']
GB_accept.head(100)


# In[46]:


# createreject_inference_function

def reject_inference(GB_accept, rejected_droped, model,selected_feature,group_dictionary,rejection_rate =0.7):
    # Convert data
    r_db, converted_r_db = convert_X(rejected_droped, selected_feature, group_dictionary)
    
    Assume_Good = rejected_droped.copy()
    Assume_Good['Good_prob'] = rejection_rate/(1 - rejection_rate)/(len(rejected_droped)/GB_accept['Frequency'].sum())
    Assume_Good['GOOD'] = 1 
    Assume_Good['Frequency'] = Assume_Good['Good_prob']*model.predict_proba(converted_r_db)[:,1]
    
    Assume_Bad = rejected_droped.copy()
    Assume_Bad['Good_prob'] = rejection_rate/(1 - rejection_rate)/(len(rejected_droped)/GB_accept['Frequency'].sum())
    Assume_Bad['GOOD'] = 0
    Assume_Bad['Frequency'] = Assume_Good['Good_prob']*model.predict_proba(converted_r_db)[:,0]
    
    Assume_GB = Assume_Good.append(Assume_Bad, ignore_index=True)
    
    Accept_Reject = GB_accept.reset_index().append(Assume_GB[GB_accept.columns.tolist()], ignore_index=True,sort=False)
    return Accept_Reject


# In[47]:


# run reject inference function
Accept_Reject= reject_inference(GB_accept, rejected_droped, lr,selected_feature, group_dictionary, rejection_rate =0.7)
print('Accept_Reject:', len(Accept_Reject))


# In[48]:


# prepare data for training
y_Good2=Accept_Reject["GOOD"]
Accept_Reject_index = Accept_Reject['index']
Accept_Reject_freq = Accept_Reject['Frequency']
Accept_Reject_noindex = Accept_Reject.drop(['index','GOOD'],axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(Accept_Reject_noindex, y_Good2, random_state = 1234567, test_size=0.3)
y_train_freq2 = X_train2['Frequency']
y_test_freq2 = X_test2['Frequency']
X_train2 = X_train2.drop(['Frequency'],axis = 1)
X_test2 = X_test2.drop(['Frequency'],axis = 1)


# In[49]:


# run interactive grouping again
group_dictionary2, Accept_List2, IV_table2 = interactive_grouping2(X_train2, y_train2, Accept_Reject_freq, n=5, reject_level=0.02)


# In[50]:


group_dictionary2['crossindustry_score']


# In[51]:


# show information value table
IV_table2


# In[52]:


# convert test value to WOE
Original_X_train2, Converted_X_train2 = convert_X(X_train2, Accept_List2, group_dictionary2)
Original_X_test2, Converted_X_test2 = convert_X(X_test2, Accept_List2, group_dictionary2)


# In[53]:


# variable selection
drop_list = []
N2 = 15 # number of variables
Test_roc = []
Train_roc = []
Cs = np.logspace(-3, 1, num=20)
for C in Cs:
    lr2 = LogisticRegression(penalty='l1', solver='saga',C = C).fit(Converted_X_train2.drop(drop_list,axis = 1), y_train2,y_train_freq2)
    feature_importances2 = pd.DataFrame(lr2.coef_.T,
                                       index = Converted_X_train2.drop(drop_list,axis = 1).columns,
                                        columns=['importance']).sort_values('importance',ascending=False)
    new_X_train2=Converted_X_train2[feature_importances2.iloc[0:N2,:].index.tolist()]
    new_X_test2=Converted_X_test2[feature_importances2.iloc[0:N2,:].index.tolist()]
    lr2 = LogisticRegression().fit(new_X_train2, y_train2,y_train_freq2)
    y_decision_fn_scores_auc2 = lr2.decision_function(new_X_train2)
    Train_roc.append(roc_auc_score(y_train2, y_decision_fn_scores_auc2, sample_weight = y_train_freq2))
    y_decision_fn_scores_auc2 = lr2.decision_function(new_X_test2)
    Test_roc.append(roc_auc_score(y_test2, y_decision_fn_scores_auc2, sample_weight = y_test_freq2))
    
ax = plt.gca()
ax.set_xscale('log')
ax.plot(Cs, Train_roc,'g', label = 'Train')
ax.plot(Cs, Test_roc,'b', label = 'Test')
#Set limits and titles
plt.ylim([0.69,0.75])
plt.xlabel('Cs')
plt.ylabel('AUC or ROC')
plt.legend()
plt.title('Tune regularization')
 
plt.savefig('Tuning.png')
plt.show()

df2 = pd.DataFrame([Cs,Train_roc,Test_roc]).T.rename(columns={0:"C",1:"Train", 2:"Test"})
print(df2[df2["Test"] == df2["Test"].max()])
print(df2)


# In[54]:


# run logistic regression
lr2 = LogisticRegression(penalty='l1', solver='saga',C = 0.029764).fit(Converted_X_train2.drop(drop_list,axis = 1), y_train2,y_train_freq2)
feature_importances2 = pd.DataFrame(lr2.coef_.T,
                                       index = Converted_X_train2.drop(drop_list,axis = 1).columns,
                                        columns=['importance']).sort_values('importance',ascending=False)
new_X_train2=Converted_X_train2[feature_importances2.iloc[0:N2,:].index.tolist()]
new_X_test2=Converted_X_test2[feature_importances2.iloc[0:N2,:].index.tolist()]
lr2 = LogisticRegression().fit(new_X_train2, y_train2,y_train_freq2)
y_decision_fn_scores_auc2 = lr2.decision_function(new_X_train2)
print('Train set AUC: ',roc_auc_score(y_train2, y_decision_fn_scores_auc2, sample_weight = y_train_freq2))
y_decision_fn_scores_auc2 = lr2.decision_function(new_X_test2)
print('Test set AUC: ',roc_auc_score(y_test2, y_decision_fn_scores_auc2, sample_weight = y_test_freq2))
lr_coef2 = pd.DataFrame(lr2.coef_.T, index = new_X_train2.columns, columns=['coefficient']).sort_values('coefficient',ascending=False)
print(lr_coef2)
lr2.intercept_[0]


# In[55]:


# calculate score
scorecard_point ={}
for i in lr_coef2.index.tolist():
    scorecard_point[i] = group_dictionary2[i].copy()
    scorecard_point[i]['score_point'] =  round((scorecard_point[i]['WOE']*lr_coef2.loc[i][0]+lr2.intercept_[0]/N2)*28.8539008+200/N2)
scorecard_point['cc2_ad']


# In[56]:


# score function
# the same as Convert_X function
def score(db_original, Feature_List, scorecard_point):
    db = db_original[Feature_List]
    db1 = db.fillna(db.min()-200000).reset_index()
    columns = db.columns
    converted_db = db.copy()
    scorecard_point1 = copy.deepcopy(scorecard_point)
    for i in range(len(columns)):
                db1[columns[i]] = db1[columns[i]].astype('float64')
                scorecard_point1[columns[i]].loc[0,['min_' + columns[i]]] = scorecard_point[columns[i]]['min_' + columns[i]].min()-100000
                db2 = pd.merge_asof(db1.sort_values(columns[i]),                                     scorecard_point1[columns[i]].fillna(scorecard_point1[columns[i]].min()-300000).sort_values('min_' + columns[i]),                                     left_on = columns[i], right_on = 'min_' + columns[i])[['index','score_point']].set_index('index')
                converted_db[columns[i]] = db2['score_point']
                converted_db['sum'] = converted_db[Feature_List].sum(axis = 1)
    return converted_db


# In[57]:


# run the score function to calculate score
scored_db = score(db,lr_coef2.index.tolist(),scorecard_point)


# In[58]:


scored_db


# In[59]:


# add a new column 'score' to original dataset
db['score'] = scored_db['sum']


# In[60]:


# create scorecard table
for i,j in zip(lr_coef2.index.tolist(),range(len(lr_coef2))):
    if j == 0:
        scorecard = scorecard_point[i].copy()
        scorecard = scorecard.rename(columns={'min_' + i:'min','max_'+ i:'max'})
        scorecard['variable'] = i
        scorecard = scorecard.set_index('variable')
        
    else:
        scorecard2 = scorecard_point[i].copy()
        scorecard2 = scorecard2.rename(columns={'min_' + i:'min','max_'+ i:'max'})
        scorecard2['variable'] = i
        scorecard2 = scorecard2.set_index('variable')
        scorecard = pd.concat([scorecard,scorecard2], ignore_index=False)
scorecard


# In[61]:


#Stage 2
#Filter out Good and Bad

Stage2_Goods=db[db["TARGET_GB"]=="GOOD"]  #we use two db because one db will give the count of the whole dataset
Stage2_Bads=db[db["TARGET_GB"]=="BAD"]
Stage2_GB=Stage2_Goods.append(Stage2_Bads) #can do Good.append(Bad)
print("append Good and Bad:",len(Stage2_GB)) #GB #will give the dataset that has the good and bad appended


# In[62]:


# Convert Good and Bad to dummy variable
Stage2_GB_y = pd.get_dummies(Stage2_GB["TARGET_GB"]) #converts categorical variable (target) into dummy/indicator variables such as zero and one. The G_B becomes two columns (Good and Bad)
Stage2_GB_y_Bad = Stage2_GB_y["BAD"]


# In[63]:


# Train and Test split
Stage2_X_train, Stage2_X_test, Stage2_y_train, Stage2_y_test = train_test_split(Stage2_GB[['score','SQL_CASHDOWNPCT', 'SQL_PTI', 'SQL_OPENINGBALANCE_PCT']], Stage2_GB_y_Bad, random_state = 34, test_size=0.25) #2609


# In[64]:


# Run logistic Regression
lr3 = LogisticRegression().fit(Stage2_X_train, Stage2_y_train)


# In[65]:


y_decision_fn_scores_auc3 = lr3.decision_function(Stage2_X_train)
print('Train set AUC: ',roc_auc_score(Stage2_y_train, y_decision_fn_scores_auc3))
y_decision_fn_scores_auc3 = lr3.decision_function(Stage2_X_test)
print('Test set AUC: ',roc_auc_score(Stage2_y_test, y_decision_fn_scores_auc3))
lr_coef3 = pd.DataFrame(lr3.coef_.T, index = Stage2_X_train.columns, columns=['coefficient']).sort_values('coefficient',ascending=False)
print(lr_coef3)
lr3.intercept_[0]


# In[ ]:


# calculate Bad Proba
db['Bad_Proba'] = lr3.predict_proba(db[['score','SQL_CASHDOWNPCT', 'SQL_PTI', 'SQL_OPENINGBALANCE_PCT']])[:,1]


# In[ ]:


abc = IV_table2.set_index('Variable_Name').loc[new_X_train2.columns.tolist(),:].sort_values('Information_Value', ascending=False)
abc


# In[ ]:


# export result to excel
writer = pd.ExcelWriter(r'\\neptune\RAD\4 Models\Scorecard 8.0\Temp\JM\FT_Clarity\Clarity hit vs FT hit\CA\Clarity\8.0 Valid FICO C_Hit CA(Clarity, FT, no spectrum).xlsx')
db.to_excel(writer,'Sheet1')
scorecard.to_excel(writer,'Variables_after')
scorecard_before.to_excel(writer,'Variables_before')
abc.to_excel(writer,'Information_Values')
writer.save()


# In[ ]:




