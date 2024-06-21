#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[3]:


df = pd.read_csv("C:/Users/Ijaz khan/Downloads/kidney_disease.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[9]:


df.nunique()


# In[10]:


df.shape


# In[11]:


df.drop('id', inplace=True,axis = 1)


# In[12]:


df.head()


# In[118]:


text_column = ['pcv','wc','rc','htn']
for i in text_column:
    print(f"{i} : {df[i].dtypes}")


# In[119]:


def covert_to_numaric(df,col):
    df[col]= pd.to_numeric(df[col], errors = 'coerce')
for c in text_column:
    covert_to_numaric(df,c)
    print(f"{c} : {df[c].dtypes}")
    


# In[125]:


df['htn'].unique()


# In[120]:


def mean_entry(df,col):
    mean_df = df[col].mean()
    df[col].fillna(value = mean_df, inplace = True)


# In[54]:


def mode_entry(df,col):
    mode_df = df[col].mode()[0]
    df[col].fillna(value = mode_df, inplace = True)


# In[55]:


col_cor = [col for col in df if df[col].dtypes != 'object']
print(col_cor)


# In[56]:


cat_cor = [cat for cat in df if df[cat].dtype == 'object']
print(cat_cor)


# In[57]:


for c in col_cor:
    mean_entry(df,c)
    print(c)


# In[58]:


for s in cat_cor:
    mode_entry(df,s)
    print(s)


# In[59]:


df.isnull().sum()


# In[65]:


for d in df:
    print(f"{d} : {df[d].unique()}")


# In[66]:


df['dm'] = df['dm'].replace(to_replace = {' yes':'yes', '\tno':'no', '\tyes':'yes'})
df['cad'] = df['cad'].replace(to_replace = {'\tno':'no'})
df['classification'] = df['classification'].replace (to_replace = {'ckd\t':'ckd', 'notckd':'not ckd'})


# In[67]:


for d in df:
    print(f"{d} : {df[d].unique()}")


# In[68]:


cat_cor = [cat for cat in df if df[cat].dtype == 'object']
print(cat_cor)


# In[69]:


df['rbc'] = df['rbc'].map ({'normal': 1, 'abnormal': 0 })
df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0 })
df['pcc'] = df['pcc'].map({'notpresent' : 1 , 'present' : 0 })
df['ba'] = df['ba'].map({'notpresent' : 1 , 'present' : 0 })
df['htn'] = df['htn'].map({'yes' : 1 , 'no' : 0 })
df['dm'] = df['dm'].map({'yes' : 1 , 'no' : 0 })
df['cad'] = df['cad'].map({'yes' : 1 , 'no' : 0 })
df['appet'] = df['appet'].map({'good' : 1 , 'poor' : 0 })
df['pe'] = df['pe'].map({'yes' : 1 , 'no' : 0 })
df['ane'] = df['ane'].map({'yes' : 1 , 'no' : 0 })
df['classification'] = df['classification'].map({'ckd' : 1 , 'not ckd' : 0 })


# In[70]:


df.head()


# In[71]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,linewidth = 0.5)
plt.show()
           


# In[75]:


best_cor = df.corr()['classification'].abs().sort_values(ascending = False)[1:]
best_cor


# In[78]:


df['classification'].value_counts()


# In[79]:


x = df.drop('classification', axis=1)
y = df['classification']


# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
print(f"'x' shape : {x_train.shape}")
print(f"'y' shape : {x_test.shape}")


# In[82]:


from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier()


# In[83]:


dct.fit(x_train,y_train)


# In[84]:


y_pred = dct.predict(x_test)
y_pred


# In[91]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[99]:


mod = []
mod.append(('Naive Bayes :', GaussianNB()))
mod.append(('Neighbors :',KNeighborsClassifier(n_neighbors = 8)))
mod.append(('Ensemble :',RandomForestClassifier()))
mod.append(("svm",SVC(kernel = 'linear')))


# In[1]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix


# In[108]:


for name,mod1 in mod:
    print(name,mod1)
    print()
    mod1.fit(x_train,y_train)
    y_pred = mod1.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print('\n')
    print('Accuracy : ', accuracy_score(y_test,y_pred))
    print('\n')
    print('F1 :',f1_score(y_test,y_pred))
    print('\n')
    print('Recall :', recall_score(y_test,y_pred))
    print('\n')
    print('precision :', precision_score(y_test,y_pred))
    print('\n')


# In[ ]:





# In[ ]:




