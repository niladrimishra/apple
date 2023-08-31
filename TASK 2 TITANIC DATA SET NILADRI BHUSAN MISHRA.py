#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
     


# In[2]:


df=pd.read_csv('Titanic_Dataset.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[9]:


df.shape


# In[13]:


df.columns


# In[17]:


df.dtypes


# In[18]:


df.duplicated().sum()


# In[19]:



null=df.isna().sum().sort_values(ascending=False)
null = null[null>0]
null


# In[20]:


df.isnull().sum().sort_values(ascending=False)*100/len(df)


# In[21]:



df.drop(columns = 'Cabin', axis = 1, inplace = True)
df.columns


# In[23]:


df['Age'].fillna(df['Age'].mean(),inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)


# In[24]:


df.isna().sum()


# In[25]:



df[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()


# In[26]:


df['Survived'].unique()


# In[27]:


df['Sex'].unique()


# In[28]:



df['Pclass'].unique()


# In[29]:


df['SibSp'].unique()


# In[30]:



df['Parch'].unique()


# In[31]:



df['Embarked'].unique()


# In[32]:


df.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
df.columns


# In[33]:


df.info


# In[34]:


df.describe()


# In[35]:


df.describe(include='O')


# In[36]:


d1 = df['Sex'].value_counts()
d1


# In[37]:


sns.countplot(x=df['Sex'])
plt.show()


# In[38]:



plt.figure(figsize=(5,5))
plt.pie(d1.values,labels=d1.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[39]:


sns.countplot(x=df['Sex'],hue=df['Survived'])
plt.show()


# In[40]:


sns.countplot(x=df['Embarked'],hue=df['Sex'])
plt.show()


# In[41]:


sns.countplot(x=df['Pclass'])
plt.show()


# In[42]:



sns.countplot(x=df['Pclass'],hue=df['Sex'])
plt.show()
     


# In[43]:


sns.kdeplot(x=df['Age'])
plt.show()
     


# In[44]:


print(df['Survived'].value_counts())
sns.countplot(x=df['Survived'])
plt.show()


# In[45]:



sns.countplot(x=df['Parch'],hue=df['Survived'])
plt.show()


# In[46]:


sns.countplot(x=df['SibSp'],hue=df['Survived'])
plt.show()


# In[47]:


sns.countplot(x=df['Embarked'],hue=df['Survived'])
plt.show()


# In[48]:



sns.kdeplot(x=df['Age'],hue=df['Survived'])
plt.show()


# In[49]:


df.hist(figsize=(10,10))
plt.show()


# In[ ]:


sns.boxplot(df)
plt.show()


# In[51]:


df.corr()


# In[92]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[93]:



sns.pairplot(df)
plt.show()


# In[94]:


df['Survived'].value_counts()


# In[95]:


sns.countplot(x=df['Survived'])
plt.show()


# In[96]:



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in ['Sex','Embarked']:
    df[column] = le.fit_transform(df[column])

df.head()


# In[97]:



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[98]:


cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = df[cols]
y = df['Survived']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[99]:


x.head()
y.head()


# In[100]:


print(891*0.10)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[101]:



def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))
    print('Testing Score',model.score(x_test,y_test))


# In[102]:



#logistic regression
lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In[103]:


mscore(lr)
ypred_lr = lr.predict(x_test)
print(ypred_lr)
cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# In[104]:


#KNN classifier
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)


# In[105]:


mscore(knn)
ypred_knn = knn.predict(x_test)
print(ypred_knn)
cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)
     


# In[106]:


#SVC
svc = SVC(C=1.0)
svc.fit(x_train, y_train)


# In[107]:



mscore(svc)
ypred_svc = svc.predict(x_test)
print(ypred_svc)
cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# In[108]:


#RandomForest
rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)


# In[109]:


mscore(rfc)
ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)
cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# In[110]:


#Decision Tree
dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)


# In[111]:



mscore(dt)
ypred_dt = dt.predict(x_test)
print(ypred_dt)
cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# In[112]:


#Adaboost
ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)


# In[114]:


mscore(ada_boost)
ypred_ada_boost = ada_boost.predict(x_test)
cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[116]:


models = pd.DataFrame({ 'Model':['Logistic Regression','knn','SVC','Random Forest Classifier','Decision Tree Classifier','Ada Boost Classifier'],
    'Score': [acc_lr,acc_knn,acc_svc,acc_rfc,acc_dt,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[117]:


colors = ["blue", "green", "red", "yellow","orange","purple"]

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )


# In[119]:


#Decision tree model emerged as highest accurate model

