
# coding: utf-8

# # Table of Contents
#  <p>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')


# In[2]:


x_rand = np.random.rand(100)
y_rand = np.random.rand(100)


# In[3]:


fig, ax = plt.subplots()
ax.scatter(x_rand, y_rand)


# In[4]:


x_rand = np.random.rand(500)
y_rand = np.random.rand(500)


# In[5]:


fig, ax = plt.subplots()
ax.hist2d(x_rand, y_rand);


# In[ ]:




