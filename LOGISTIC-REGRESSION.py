#!/usr/bin/env python
# coding: utf-8

# ### =========================LOGISTIC-REGRESSION==================

# ### =====================BANK -FULL DATA=============

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings. filterwarnings('ignore')


# In[4]:


bank=pd.read_csv("bank-full.csv",sep=";")
bank


# In[5]:


bank.shape


# In[6]:


bank.isna().sum()


# In[7]:


bank.dtypes


# In[8]:


bank.describe()


# In[9]:


bank.info()


# In[10]:


bank1=pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome'])
bank1


# In[11]:


pd.set_option("display.max.columns",None)


# In[12]:


bank1


# In[13]:


bank1['default']=np.where(bank1['default'].str.contains('yes'),1,0)


# In[14]:


bank1['housing']=np.where(bank1['housing'].str.contains('yes'),1,0)


# In[15]:


bank1['loan']=np.where(bank1['loan'].str.contains('yes'),1,0)


# In[16]:


bank1['y']=np.where(bank1['y'].str.contains('yes'),1,0)


# In[17]:


bank1


# In[18]:


bank1['month'].value_counts()


# In[19]:


order={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}


# In[20]:


bank1=bank1.replace(order)


# In[21]:


bank


# In[22]:


bank1.info()


# In[25]:


x=pd.concat([bank1.iloc[:,0:11],bank1.iloc[:,12:]],axis=1)
x


# In[26]:


y=bank1.iloc[:,11]
y


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=12,shuffle=True)


# In[33]:


x_train.shape,y_train.shape


# In[35]:


x_train


# In[36]:


y_train


# In[40]:


from sklearn.linear_model import LogisticRegression
Logistic_model=LogisticRegression()
Logistic_model.fit(x_train,y_train)


# In[43]:


Logistic_model.coef_
Logistic_model.intercept_


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[45]:


classifier=LogisticRegression()
classifier.fit(x,y)


# In[46]:


y_pred=classifier.predict(x)


# In[47]:


y_predictDataFrame=pd.DataFrame({'actual':y,'predict_prob':y_pred})


# In[48]:


y_predictDataFrame


# In[49]:


confusion_matrix=confusion_matrix(y,y_pred)
print(confusion_matrix)


# In[50]:


(39107+1274)/(37109+4015+815+1274)


# In[51]:


from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


# In[52]:


fpr,tpr,thresholds =roc_curve(y,classifier.predict_proba(x)[:,1])
auc=roc_auc_score(y,y_pred)
plt.plot(fpr,tpr,color='blue',label='logit model(area=%0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')


# In[53]:


auc


# In[ ]:




