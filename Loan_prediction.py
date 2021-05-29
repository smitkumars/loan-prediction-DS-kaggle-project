#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings

warnings.filterwarnings("ignore")


# In[4]:


train= pd.read_csv("train_loan.csv")
test= pd.read_csv("test_loan.csv")


# In[5]:


train_original=train.copy()
test_original=test.copy()


# In[6]:


train.columns


# In[7]:


test.columns


# In[8]:


train.describe()


# In[9]:


train.info()


# In[10]:


test.info()


# In[11]:


train.dtypes


# In[13]:


train.shape


# In[14]:


test.shape


# In[15]:


train['Loan_Status'].value_counts()


# In[16]:


train['Loan_Status'].value_counts(normalize=True)


# In[17]:


train['Loan_Status'].value_counts().plot.bar()


# In[21]:


plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Gender')
plt.subplot(222)

train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Married')
plt.subplot(223)


train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Self_Employed')
plt.subplot(224)

train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Credit_History')
plt.show()


# In[22]:


plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents')
plt.subplot(132)

train['Education'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Education')
plt.subplot(133)

train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Property_Area')
plt.show()


# In[23]:


plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)

train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[152]:


train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")
Text(0.5,0.98,'')


# In[27]:


plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[30]:


plt.figure(1)
plt.subplot(121)

df=train.dropna()

sns.distplot(df['LoanAmount']);
plt.subplot(122)
df['LoanAmount'].plot.box(figsize=(16,5))
plt.show()


# In[31]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status'])

Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))


# In[34]:


Married=pd.crosstab(train['Married'],train['Loan_Status'])

dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])

Education=pd.crosstab(train['Education'],train['Loan_Status'])

Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])


Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

dependents.div(dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()


# In[39]:


Credit_History= pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=  pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

plt.show()

Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.show()


# In[41]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[45]:


bins=[0,2500,4000,6000,81000]

group=['Low','Average','High','Very high']

train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])

Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.xlabel('ApplicantIncome')

P=plt.ylabel('Percentage')


# In[46]:


bins=[0,1000,3000,42000]

group=['Low','Average','High']

train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.xlabel('CoapplicantIncome')

P=plt.ylabel('Percentage')


# In[48]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']


train['Total_Income']


# In[49]:


bins=[0,2500,4000,6000,81000]

group=['Low','Average','High','Very high']

train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

train['Total_Income_bin']


# In[50]:


Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.xlabel('Total_Income')

P=plt.ylabel('Percentage')


# In[51]:


bins=[0,100,200,700]

group=['Low','Average','High']

train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])

LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.xlabel('LoanAmount')

P=plt.ylabel('Percentage')


# In[58]:


#train=train.drop(['Income_bin','Coapplicant_Income_bin','LoanAmount_bin','Total_Income_bin','Total_Income'],axis=1)

train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N',0,inplace=True)
train['Loan_Status'].replace('Y',1,inplace=True)
train


# In[64]:


matrix=train.corr()

f,ax= plt.subplots(figsize=(9,6))

sns.heatmap(matrix,vmax=8,square=True,cmap="BuPu")


# In[65]:


train.isnull().sum()


# In[66]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)

train['Married'].fillna(train['Married'].mode()[0],inplace=True)

train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)

train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)


# In[67]:


train.isnull().sum()


# In[68]:


train['Loan_Amount_Term'].value_counts()


# In[69]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


# In[70]:


train['Loan_Amount_Term'].value_counts()


# In[71]:


train.isnull().sum()


# In[73]:


train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)


# In[74]:


train.isnull().sum()


# In[75]:


test['Gender'].fillna(test['Gender'].mode()[0],inplace=True)

test['Married'].fillna(test['Married'].mode()[0],inplace=True)

test['Dependents'].fillna(test['Dependents'].mode()[0],inplace=True)

test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace=True)

test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace=True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace=True)


test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace=True)


# In[77]:


test.isnull().sum()


# In[80]:


train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)

test['LoanAmount_log']=np.log(test['LoanAmount'])
#test['LoanAmount_log'].hist(bins=20)


# Evaluation Metrics for Classification Problems
# 
# Accuracy- it using confusion matrix which is tabular fromat of actual vs predicted values. 
# 
# True postitive- targets which are actually true (Y) and we have predicted them true 
# 
# True negative- targets which are actually false (N and we have predicted them false (N)
# 
# False positive - targets whsich are actually false N and we have predicted them true Y
# 
# False negative- targets which are actually True Y but we have predicted false N
# 
# 
# Precision- measure of correctness in true prediction 
#         = TP/(TP+FP)
#         
# Recall= TP/(TP+FN)
# 
# Specificity= TN/(TN+FP)
# 
# ROC curve= receiver operating characterstic

# In[81]:


#Model Building

train=train.drop('Loan_ID',axis=1)

test=test.drop('Loan_ID',axis=1)


# In[82]:


X=train.drop('Loan_Status',1)
y=train.Loan_Status


# In[83]:


X=pd.get_dummies(X)

train=pd.get_dummies(train)

test=pd.get_dummies(test)


# In[84]:


from sklearn.model_selection import train_test_split
x_train, x_cv,y_train,y_cv= train_test_split(X,y,test_size=0.3)


# In[85]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[86]:


model= LogisticRegression()
model.fit(x_train,y_train)

LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,max_iter=100,multi_class='ovr',n_jobs=1,penalty='12',random_state=1,solver='liblinear',tol=0.0001,verbose=0,warm_start=False)


# In[87]:


pred_cv= model.predict(x_cv)


# In[88]:


accuracy_score(y_cv,pred_cv) # 77% accurate,identified 77% of loan status correctly


# In[89]:


pred_test=model.predict(test)


# In[90]:


submission=pd.read_csv("sample_submission.csv")


# In[91]:


submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']


# In[92]:


submission['Loan_Status'].replace(0,'N',inplace=True)

submission['Loan_Status'].replace(1,'Y',inplace=True)


# In[94]:


pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[95]:


194*20


# In[96]:


628/194.75


# In[99]:


# Logistic regression using stratified k-folds cross validation

from sklearn.model_selection import StratifiedKFold


# In[107]:


i=1
kf= StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X.loc[train_index],X.loc[test_index]
    ytr,yvl=y[train_index],y[test_index]
    model=LogisticRegression(random_state=1)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+=1
    pred_test= model.predict(test)
    pred=model.predict_proba(xvl)[:,1]
        


# In[108]:


from sklearn import metrics 

fpr,tpr,_=metrics.roc_curve(yvl,pred)
auc=metrics.roc_auc_score(yvl,pred)
plt.figure(figsize=(12,8))

plt.plot(fpr,tpr,label="validation, auc="+str(auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True positive Rate')

plt.legend(loc=4)

plt.show()


# In[109]:


submission['Loan_Status']=pred_test

submission['Loan_ID']=test_original['Loan_ID']


# In[110]:


pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('Logisic.csv')


# In[111]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

test['Total_Income']=test['ApplicantIncome']+train['CoapplicantIncome']


# In[ ]:





# In[112]:


sns.distplot(train['Total_Income'])


# In[113]:


train['Total_Income_log']=np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log'])

test['Total_Income_log']=np.log(test['Total_Income'])


# In[114]:


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']

test['EMI']=test['LoanAmount']/train['Loan_Amount_Term']

sns.distplot(train['EMI'])


# In[115]:


train['Balance Income']=train['Total_Income']-(train['EMI']*1000)
#multiply with 1000 to make units equal test

test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

sns.distplot(train['Balance Income'])


# In[116]:


train=train.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1)

test=test.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1)


# In[145]:


test


# In[ ]:





# In[147]:


##Model building

X=train.drop('Loan_Status',1)
y=train.Loan_Status


# In[148]:


X=pd.get_dummies(X)

train=pd.get_dummies(train)

test=pd.get_dummies(test)


# In[149]:


##Logistic Regression

i=1
kf= StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X.loc[train_index],X.loc[test_index]
    ytr,yvl=y[train_index],y[test_index]
    
    model=LogisticRegression(random_state=1)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+=1
   # pred_test= model.predict(test)
    #pred=model.predict_proba(xvl)[:,1]


# In[140]:


#pred_test= model.predict(test)
#pred=model.predict_proba(xvl)[:,1]

test= test.dropna()
xvl=xvl.dropna()


# In[150]:


pred_test= model.predict(test)
pred= model.predict_proba(xvl)[:,1]


# In[ ]:





# In[ ]:




