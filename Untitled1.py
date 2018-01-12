
# coding: utf-8

# # Table of Contents
#  <p>

# In[2]:


# %load Untitled.py


# # Table of Contents
#  <p>

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')


# In[3]:


x_rand = np.random.rand(100)


# In[4]:


y_rand = np.random.rand(100)


# In[5]:


fig, ax = plt.subplots()
ax.scatter(x_rand, y_rand)


# In[6]:


x_rand = np.random.rand(500)
y_rand = np.random.rand(500)


# In[9]:


fig, ax = plt.subplots()
ax.hist2d(x_rand, y_rand);


# In[ ]:






# In[ ]:




