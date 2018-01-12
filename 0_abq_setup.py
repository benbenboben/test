
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Load-data" data-toc-modified-id="Load-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load data</a></div><div class="lev1 toc-item"><a href="#Set-up-data" data-toc-modified-id="Set-up-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Set up data</a></div><div class="lev2 toc-item"><a href="#PVLib-clearsky" data-toc-modified-id="PVLib-clearsky-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>PVLib clearsky</a></div><div class="lev2 toc-item"><a href="#Statistical-clearsky" data-toc-modified-id="Statistical-clearsky-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Statistical clearsky</a></div><div class="lev2 toc-item"><a href="#PVLib-clearsky-detection" data-toc-modified-id="PVLib-clearsky-detection-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>PVLib clearsky detection</a></div><div class="lev1 toc-item"><a href="#Dump-to-file" data-toc-modified-id="Dump-to-file-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dump to file</a></div>

# In[16]:


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

# In[17]:


NSRDB_PATH = os.path.expanduser('~/data_sets/nsrdb/abq_area/')
GROUND_PATH = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')


# In[18]:


nsrdb = cs_detection.ClearskyDetection.read_nsrdb_dir(NSRDB_PATH, 'MST')


# In[4]:


ground = cs_detection.ClearskyDetection.read_snl_rtc(GROUND_PATH, 'MST')


# # Set up data

# In[5]:


nsrdb.df[nsrdb.df['GHI'] < 0] = 0


# In[6]:


ground.df[ground.df['GHI'] < 0] = 0


# In[7]:


nsrdb.df['sky_status'] = (nsrdb.df['Cloud Type'] == 0) & (nsrdb.df['GHI'] > 0)


# ## PVLib clearsky

# In[8]:


params = {'altitude': 1658, 'latitude': 35.0549, 'longitude': -106.5433}


# In[9]:


nsrdb.generate_pvlib_clearsky(**params)


# In[10]:


ground.generate_pvlib_clearsky(**params)


# ## Statistical clearsky

# In[11]:


nsrdb.generate_statistical_clearsky()


# In[12]:


ground.generate_statistical_clearsky()


# ## PVLib clearsky detection

# In[13]:


ground.pvlib_clearsky_detect()


# # Dump to file

# In[14]:


nsrdb.to_pickle('abq_nsrdb.pkl', overwrite=True)


# In[15]:


ground.to_pickle('abq_ground.pkl', overwrite=True)


# In[ ]:




