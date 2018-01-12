
# coding: utf-8

# # Table of Contents
#  <p>

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')


# In[13]:


x_rand = np.random.rand(100)


# In[14]:


y_rand = np.random.rand(100)


# In[15]:


fig, ax = plt.subplots()
ax.scatter(x_rand, y_rand)

