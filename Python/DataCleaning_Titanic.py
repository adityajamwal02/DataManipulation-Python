#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Reading the dataset (csv file)

df=pd.read_csv('train.csv')
df


# In[3]:


# Dimensions of dataset

df.shape


# In[4]:


# Statistical measures of numerical columns

df.describe()


# In[5]:


# Information about titanic.csv

df.info()


# In[6]:


# Random Sample

df.sample()


# In[7]:


# All columns present 

df.columns


# In[8]:


# Cross relation between survived people and their sex

pd.crosstab(df["Sex"], df["Survived"])


# In[9]:


# Countplot for the same as above

ax=sns.countplot(x="Sex", hue="Survived", palette="Set1", data=df)
ax.set(title="Survivors a/c to Sex", xlabel="Sex", ylabel="Total")
plt.show()


# In[10]:


# Factorplot 1

sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=df, aspect=0.9, size=3.5)


# In[11]:


# Factorplot 2

sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=df, aspect=0.9, size=3.5)


# In[12]:


# First 5 entries of dataset

df.head()


# In[13]:


# Facetgird of survived people w.r.t fare charge

graph=sns.FacetGrid(df, col="Survived")
graph.map(plt.hist, "Fare", bins=20)


# In[14]:


# Handling anomalies/ outliers

df.loc[df["Fare"]>400, "Fare"]=df["Fare"].median()


# In[15]:


graph=sns.FacetGrid(df, col="Survived")
graph.map(plt.hist, "Fare", bins=20)


# In[16]:


# Checking for null value count

for column in df:
    print(column, ": ", df[column].isnull().sum())


# In[17]:


# Null age values filled with median of age

df["Age"].fillna(df["Age"].median(), inplace=True)


# In[18]:


for column in df:
    print(column, ": ", df[column].isnull().sum())


# In[19]:


print(df["Embarked"].value_counts())


# In[20]:


df["Embarked"].fillna("S", inplace=True)
del df["Cabin"]


# In[21]:


for column in df:
    print(column, ": ", df[column].isnull().sum())


# In[22]:


df["Name"].sample(10)


# In[23]:


# Feature Engg

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "No title in name"
    

# Functional programming

titles = set([x for x in df.Name.map(lambda x: get_title(x))])
print(titles)


def shorter_title(x):
    title=x["Title"]
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'the Countess', 'Dona', 'Lady', 'Sir']:
        return 'Royalty'
    elif title in ['Mlle', 'Ms', 'Miss', 'Mrs', 'Mme']:
        return 'Female'
    else: 
        return 'Male'


# In[24]:


# Creating new column with Title rather than different names

df["Title"]=df["Name"].map(lambda x: get_title(x))

df['Title']=df.apply(shorter_title, axis=1)
print(df.Title.value_counts())


# In[25]:


# Deleting Name column as Title have been added with scaling feature engg

df.drop("Name", axis=1, inplace=True)
df.sample(20)


# In[26]:


del df["Ticket"]


# In[27]:


df.Sex.replace(('male','female'),(0,1), inplace=True)
df.Embarked.replace(('S','C','Q'),(0,1,2), inplace=True)
df.Title.replace(('Male','Female','Royalty','Officer'),(0,1,2,3), inplace=True)


# In[30]:


df.sample(25)




