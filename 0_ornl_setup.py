
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Load-data" data-toc-modified-id="Load-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load data</a></div><div class="lev1 toc-item"><a href="#Set-up-data" data-toc-modified-id="Set-up-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Set up data</a></div><div class="lev2 toc-item"><a href="#PVLib-clearsky" data-toc-modified-id="PVLib-clearsky-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>PVLib clearsky</a></div><div class="lev2 toc-item"><a href="#Statistical-clearsky" data-toc-modified-id="Statistical-clearsky-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Statistical clearsky</a></div><div class="lev2 toc-item"><a href="#PVLib-clearsky-detection" data-toc-modified-id="PVLib-clearsky-detection-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>PVLib clearsky detection</a></div><div class="lev1 toc-item"><a href="#Dump-to-file" data-toc-modified-id="Dump-to-file-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dump to file</a></div>

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pytz
import itertools

import pvlib
import cs_detection

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Load data

# In[2]:


NSRDB_PATH = os.path.expanduser('/Users/benellis/data_sets/nsrdb/ornl_area/')
GROUND_PATH = os.path.expanduser('/Users/benellis/data_sets/ornl_midc/z5475326.txt')


# In[3]:


nsrdb = cs_detection.ClearskyDetection.read_nsrdb_dir(NSRDB_PATH, 'EST')


# In[4]:


ground = cs_detection.ClearskyDetection.read_ornl_file(GROUND_PATH, 'EST')


# # Set up data

# In[5]:


nsrdb.df[nsrdb.df['GHI'] < 0] = 0


# In[6]:


ground.df[ground.df['GHI'] < 0] = 0


# In[7]:


nsrdb.df['sky_status'] = (nsrdb.df['Cloud Type'] == 0) & (nsrdb.df['GHI'] > 0)


# ## PVLib clearsky

# In[9]:


params = {'altitude': 245, 'latitude': 35.93, 'longitude': -84.31}


# In[10]:


nsrdb.generate_pvlib_clearsky(**params)


# In[11]:


ground.generate_pvlib_clearsky(**params)


# ## Statistical clearsky
nsrdb.generate_statistical_clearsky()ground.generate_statistical_clearsky()
# ## PVLib clearsky detection

# In[12]:


ground.pvlib_clearsky_detect()


# # Dump to file

# In[15]:


nsrdb.to_pickle('ornl_nsrdb.pkl.gz', overwrite=True)


# In[16]:


ground.to_pickle('ornl_ground.pkl.gz', overwrite=True)


# In[ ]:





# In[ ]:




