#!/usr/bin/env python
# coding: utf-8

# In[47]:


'''Problem Statement- 'you have a multidimensional set of data (such as a set of hidden unit activations) 
and you want to see which points are closest to others.'''


# In[48]:


'''Solution-"We can clearly see that three flowers' three species of iris correspond to 3 charactastic. 
When plotted by 2 PCA technique was effective of finding the hidden structure underlying our flower info " '''


# In[49]:


'''Result-PCA allows you to identify the dimensions of greatest variance, to the dimensions of least variance. 
PCA1 has greatest variance.'''


# In[15]:


from sklearn import datasets
import matplotlib.pyplot as plt
iris=datasets.load_iris()


# In[16]:


iris.data.shape


# In[17]:


iris.get('feature_names')


# Extract 6 Rows
# 

# In[18]:


iris.data[0:6,:]


# In[19]:


from sklearn.decomposition import PCA   


# In[20]:


pca=PCA(n_components=2)  # as we want two principal componets that account for most of the structure


# In[21]:


x=pca.fit_transform(iris.data)  # fit our model


# In[22]:


x.shape   # two principal componets rather than 4


# In[ ]:


x.tar


# In[23]:


plot= plt.scatter(x[:,0], x[:,1])


# In[24]:


iris.target.shape


# In[29]:


iris.target[0:134] #output of target


# In[32]:


import numpy as np
unique_elements,count_elements = np.unique(iris.target,return_counts=True)
np.asarray((unique_elements,count_elements))


# In[33]:


list(iris.target_names)


# In[41]:


plot1=plt.scatter(x[:,0], x[:,1], c=iris.target)
plt.legend()


# In[ ]:




