
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Load-data" data-toc-modified-id="Load-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load data</a></div><div class="lev1 toc-item"><a href="#Set-up-data" data-toc-modified-id="Set-up-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Set up data</a></div><div class="lev2 toc-item"><a href="#PVLib-clearsky" data-toc-modified-id="PVLib-clearsky-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>PVLib clearsky</a></div><div class="lev2 toc-item"><a href="#Statistical-clearsky" data-toc-modified-id="Statistical-clearsky-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Statistical clearsky</a></div><div class="lev2 toc-item"><a href="#PVLib-clearsky-detection" data-toc-modified-id="PVLib-clearsky-detection-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>PVLib clearsky detection</a></div><div class="lev1 toc-item"><a href="#Dump-to-file" data-toc-modified-id="Dump-to-file-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dump to file</a></div><div class="lev1 toc-item"><a href="#Science" data-toc-modified-id="Science-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Science</a></div>

# In[15]:


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
from visualize_plotly import Visualizer

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Load data

# In[2]:


NSRDB_PATH = os.path.expanduser('~/data_sets/nsrdb/srrl_area/')
GROUND_PATH = os.path.expanduser('./srrl_cloud')


# In[3]:


nsrdb = cs_detection.ClearskyDetection.read_nsrdb_dir(NSRDB_PATH, 'MST')


# In[4]:


ground = cs_detection.ClearskyDetection.read_srrl_dir(GROUND_PATH, 'MST', keepers=['GHI', 'Total Cloud Cover [%]', 'Opaque Cloud Cover [%]'])
ground.df.index = ground.df.index.tz_convert('MST')
ground.df['Total Cloud Cover [%]'] = ground.df['Total Cloud Cover [%]'].apply(lambda x: (x >= 1) * x)
ground.df['Opaque Cloud Cover [%]'] = ground.df['Opaque Cloud Cover [%]'].apply(lambda x: (x >= 1) * x)


# In[5]:


nsrdb.intersection(ground.df.index)


# # Set up data

# In[6]:


nsrdb.df[nsrdb.df['GHI'] < 0] = 0


# In[7]:


ground.df[ground.df['GHI'] < 0] = 0


# In[8]:


nsrdb.df['sky_status'] = (nsrdb.df['Cloud Type'] == 0) & (nsrdb.df['GHI'] > 0)


# ## PVLib clearsky

# In[9]:


params = {'altitude': 1829, 'latitude': 39.74, 'longitude': -105.18}


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

# In[13]:


nsrdb.to_pickle('srrl_nsrdb_cloudy.pkl', overwrite=True)


# In[14]:


ground.to_pickle('srrl_ground_cloudy.pkl', overwrite=True)


# # Science

# In[16]:


ground_small = cs_detection.ClearskyDetection(ground.df)


# In[17]:


ground_small.trim_dates('07-01-2006', '07-08-2006')


# In[18]:


vis = Visualizer()
vis.add_line_ser(ground_small.df['GHI'], 'GHI')
vis.add_line_ser(ground_small.df['Clearsky GHI pvlib'], 'GHIcs')
vis.add_line_ser(ground_small.df['Total Cloud Cover [%]'], 'TCC')
vis.add_line_ser(ground_small.df['Opaque Cloud Cover [%]'], 'OCC')
vis.show()


# In[ ]:




