#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# In[1]:

# In[2]:


import pandas as pd
import numpy as np


# In[2]:

# In[3]:


df=pd.read_csv(r"D:\DS\Assignement\assignment_module_19_Decision_Tree_&_Random_Forest\Fraudcheck\Fraud_check.csv")


# In[3]:

# In[4]:


df.head(3)


# In[4]:

# In[5]:


df.columns=df.columns.str.replace(".","_") # changing the label value.


# In[5]:

# In[6]:


df.head()


# In[6]:

# In[48]:


df.isnull().sum()


# In[7]:


df.shape


# In[7]:

# In[8]:
#classified the income amount based on taxabel income amount on the basis of good and risky.

df["taxincome"]="<=30000"
df.loc[df["Taxable_Income"]>=30000,"taxincome"]="Good"
df.loc[df["Taxable_Income"]<=30000,"taxincome"]="Risky"


# In[10]:

# In[9]:


df.drop(["Taxable_Income"],inplace=True,axis=1)


# In[11]:

# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[12]:

# In[11]:


lb=LabelEncoder()


# In[13]:

# In[12]:


df.columns


# In[14]:

# In[13]:


df['Undergrad']=lb.fit_transform(df['Undergrad'])
df['Marital_Status']=lb.fit_transform(df['Marital_Status'])
df['Urban']=lb.fit_transform(df['Urban'])


# In[15]:

# In[14]:


df.head()


# In[16]:

# In[15]:


df['taxincome'].unique()


# In[17]:

# In[16]:


df.taxincome.value_counts()


# In[18]:

# In[17]:


import seaborn as sns


# In[19]:

# In[18]:


sns.boxplot(df['City_Population']) #check for outliers in population feature


# In[22]:

# In[19]:


colnames=list(df.columns)
predictors=colnames[:5]
target=colnames[5]


# In[21]:

# In[20]:


from sklearn.model_selection import train_test_split


# In[23]:

# In[21]:


train,test=train_test_split(df,test_size=0.3)


# In[24]:

# In[22]:


from sklearn.tree import DecisionTreeClassifier as dt


# In[25]:

# In[23]:


model=dt(criterion='entropy')


# In[26]:

# In[24]:


model.fit(train[predictors],train[target])


# In[27]:

# In[25]:


pred_train=model.predict(train[predictors])


# In[29]:

# In[26]:


pred_test=model.predict(test[predictors])


# In[34]:

# In[27]:


from sklearn.metrics import confusion_matrix,classification_report


# In[35]:

# In[28]:


print(confusion_matrix(train[target],pred_train))
print("\n")
print(classification_report(train[target],pred_train))


# In[36]:

# In[29]:


print(confusion_matrix(test[target],pred_test))
print("\n")
print(classification_report(test[target],pred_test))


# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


rf=RandomForestClassifier(n_estimators=15)


# In[32]:


rf.fit(train[predictors],train[target])


# In[33]:


pred_train_rf=rf.predict(train[predictors])


# In[34]:


print(confusion_matrix(train[target],pred_train))
print("\n")
print(classification_report(train[target],pred_train))


# In[35]:


pred_test_rf=rf.predict(test[predictors])


# In[36]:


print(confusion_matrix(test[target],pred_test))
print("\n")
print(classification_report(test[target],pred_test))


# #model is overfittied

# In[39]:


rf = RandomForestClassifier(oob_score=True) 


# In[75]:


rf.fit(train[predictors],train[target])


# In[76]:


pred_train_rf=rf.predict(train[predictors])


# In[77]:


print(confusion_matrix(train[target],pred_train_rf))
print("\n")
print(classification_report(train[target],pred_train_rf))


# In[78]:


pred_test_rf=rf.predict(test[predictors])


# In[79]:


print(confusion_matrix(test[target],pred_test_rf))
print("\n")
print(classification_report(test[target],pred_test_rf))


# #above model is overfitted. random forest classifier is itself a good classifier thats why avoiding hyperparameter tuning here 
# #but it is not working and so going for multinomial naive bayes

# In[61]:


from sklearn.naive_bayes import MultinomialNB


# In[67]:


mnb=MultinomialNB()


# In[69]:


mul_nb=mnb.fit(train[predictors],train[target])


# In[70]:


pred_train_mul=mul_nb.predict(train[predictors])


# In[71]:


print(confusion_matrix(train[target],pred_train_mul))
print("\n")
print(classification_report(train[target],pred_train_mul))


# In[72]:


pred_test_mul=mul_nb.predict(test[predictors])


# In[73]:


print(confusion_matrix(test[target],pred_test_mul))
print("\n")
print(classification_report(test[target],pred_test_mul))


# #best fit model is multinomial navie bayes model.
