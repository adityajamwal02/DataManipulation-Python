#!/usr/bin/env python
# coding: utf-8

# ## BIVARIATE ANALYSIS 

# CONTINUOUS AND CATEGORICAL

# In[ ]:


import seaborn as sns
sns.boxplot(x=df['Survived'], y=df['Age'])


# In[ ]:


sns.barplot(x=df['Sex'], y=df['Age'])


# In[ ]:




