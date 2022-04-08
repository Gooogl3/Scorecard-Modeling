#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import csv

##-----------------------------------Importing the Data-----------------------------------------------------------------##
data_dictionary = pd.read_excel(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\202119527A Match PremAB_Modified.xls')
Start = (data_dictionary['BEGIN'] - 1).tolist()
End = data_dictionary['END'].tolist()
Variable_name =[data_dictionary['FIELD NAME'].tolist()]

Slices =[]
for a in zip(Start,End):
    Slices.append(a)


# In[2]:


##-----------------------------------Appending the Data-----------------------------------------------------------------##
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q12018.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q12019.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q12021.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q22018.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q22019.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q32018.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q32019.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q42018.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\Raw Data\202119527A.ArcoutAB.Q42019.txt','r') as b:
    for line in b:
        c = [line[slice(*slc)] for slc in Slices]
        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
        Variable_name.append(d)
b.close()
##-----------------------------------Copy as needed -----------------------------------------------------------------##
# with open(r'\\neptune\RAD\9 Temp Hold\JN\DataParsingPractice\201919307A.ArcoutAB.Q22016.txt','r') as b:
#    for line in b:
#        c = [line[slice(*slc)] for slc in Slices]
#        d = [cell.strip() for cell in c ]
        ##e = [cell.strip('0') for cell in d ]
#        Variable_name.append(d)
# b.close()


# In[3]:


##-------------------Creating New list for every 255 Rows for SQL upload-------------------------------------------------##
Clarity_1 =[]
Clarity_2 =[]
Clarity_3 =[]
Clarity_4 =[]
for row in Variable_name:
    Clarity_1.append(row[:255])
    Clarity_2.append([row[0]] + row[255:500])
    Clarity_3.append([row[0]] + row[500:750])
    Clarity_4.append([row[0]] + row[750:])


# In[4]:


##----------------------Creating Seprate Excel Files for separated Data ------------------------------------------------##
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\AB\Clarity_1.5.csv','w', newline='') as resultFile:
    w = csv.writer(resultFile)
    w.writerows(Clarity_1)
resultFile.close()  
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\AB\Clarity_2.5.csv','w', newline='') as resultFile:
    w = csv.writer(resultFile)
    w.writerows(Clarity_2)
resultFile.close()  
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\AB\Clarity_3.5.csv','w', newline='') as resultFile:
    w = csv.writer(resultFile)
    w.writerows(Clarity_3)
resultFile.close()  
with open(r'\\neptune\RAD\9 Temp Hold\JN\Sep 2021 Experian Data Parsing\AB\Clarity_4.5.csv','w', newline='') as resultFile:
    w = csv.writer(resultFile)
    w.writerows(Clarity_4)
resultFile.close()  

##-----------------------------------Copy as needed -----------------------------------------------------------------##
#with open(r'\\neptune\RAD\9 Temp Hold\JN\DataParsingPractice\Clarity_2.csv','w', newline='') as resultFile:
#    w = csv.writer(resultFile)
#    w.writerows(Clarity_2)
#resultFile.close()


# In[ ]:




