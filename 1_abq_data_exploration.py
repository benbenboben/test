
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Load-data" data-toc-modified-id="Load-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load data</a></div><div class="lev1 toc-item"><a href="#Measured-vs-modeled-values" data-toc-modified-id="Measured-vs-modeled-values-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Measured vs modeled values</a></div><div class="lev2 toc-item"><a href="#NSRDB" data-toc-modified-id="NSRDB-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>NSRDB</a></div><div class="lev2 toc-item"><a href="#Ground" data-toc-modified-id="Ground-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Ground</a></div><div class="lev2 toc-item"><a href="#Ground-and-NSRDB" data-toc-modified-id="Ground-and-NSRDB-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Ground and NSRDB</a></div><div class="lev1 toc-item"><a href="#Wrap-up" data-toc-modified-id="Wrap-up-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Wrap up</a></div>

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
import visualize
import utils

import pvlib
import cs_detection
import visualize_plotly as visualize
# import visualize
# from bokeh.plotting import output_notebook
# output_notebook()

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Load data

# Read pickeld data from setup notebook.

# In[2]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb.pkl.gz')


# In[3]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground.pkl.gz')


# In[4]:


nsrdb.trim_dates('01-01-2015', '01-01-2016')


# In[5]:


ground.trim_dates('10-01-2015', '01-01-2016')


# # Measured vs modeled values

# ## NSRDB

# In[6]:


vis = visualize.Visualizer()


# In[7]:


vis.add_line_ser(nsrdb.df['GHI'], 'GHI')
vis.add_line_ser(nsrdb.df['Clearsky GHI'], 'Clearsky GHI')
vis.add_line_ser(nsrdb.df['Clearsky GHI pvlib'], 'Clearsky GHI pvlib')
vis.add_line_ser(nsrdb.df['Clearsky GHI stat'], 'Clearsky GHI stat')
vis.add_circle_ser(nsrdb.df[nsrdb.df['sky_status'] == 1]['GHI'], 'NSRDB clear')


# In[8]:


vis.show()


# PVLib is systematically higher than the NSRDB and statistical clearsky mdoels.  It will also provide consistent behavior between different data sets, so that probably shouldn't be a large concern.  All of the modeled GHI's look about the same despite the PVLib peaks being high.  In general, I would agree with the NSRDB clearness metric.  It looks like it misses some obvious points (based on GHI alone).  It also picks points that are in 'noisy' periods that probably shouldn't be picked.  Clearsky GHI stat has some issues where the curve is not smooth.  This might have to be smooth/interpolated in a different way.

# In[ ]:


utils.mean_abs_diff(nsrdb.df['Clearsky GHI pvlib'], nsrdb.df['Clearsky GHI'])


# In[ ]:


utils.mean_abs_diff(nsrdb.df['Clearsky GHI pvlib'], nsrdb.df['Clearsky GHI stat'])


# In[ ]:


nsrdb.robust_rolling_smooth('Clearsky GHI stat', 3)


# In[ ]:


vis = visualize.Visualizer()


# In[ ]:


vis.add_line_ser(nsrdb.df['GHI'], 'GHI')
vis.add_line_ser(nsrdb.df['Clearsky GHI'], 'Clearsky GHI')
vis.add_line_ser(nsrdb.df['Clearsky GHI pvlib'], 'Clearsky GHI pvlib')
vis.add_line_ser(nsrdb.df['Clearsky GHI stat'], 'Clearsky GHI stat')
vis.add_line_ser(nsrdb.df['Clearsky GHI stat smooth'], 'Clearsky GHI stat smooth')


# In[ ]:


vis.show()


# ## Ground

# In[ ]:


vis = visualize.Visualizer()


# In[ ]:


vis.add_line_ser(ground.df['GHI'], 'GHI')
vis.add_line_ser(ground.df['Clearsky GHI pvlib'], 'Clearsky GHI pvlib')
vis.add_line_ser(ground.df['Clearsky GHI stat'], 'Clearsky GHI stat')
vis.add_circle_ser(ground.df[ground.df['sky_status pvlib'] == 1]['GHI'], 'pvlib clear')


# In[ ]:


vis.show()


# The statistical clearksy trend looks ok, but it's terribly noisy.  We will try some smoothing techniques to provide a more reliable solution.

# In[ ]:


utils.mean_abs_diff(ground.df['Clearsky GHI pvlib'], ground.df['Clearsky GHI stat'])


# Mean absolute difference is quite good.  The noise is relatively small so this should be expected.  The main worry with the noise is that window based metrics for determing sky clarity might be affected.

# In[ ]:


ground.robust_rolling_smooth('Clearsky GHI stat', 60)


# In[ ]:


vis = visualize.Visualizer()


# In[ ]:


vis.add_line_ser(ground.df['GHI'], 'GHI')
vis.add_line_ser(ground.df['Clearsky GHI pvlib'], 'Clearsky GHI pvlib')
vis.add_line_ser(ground.df['Clearsky GHI stat'], 'Clearsky GHI stat')
vis.add_line_ser(ground.df['Clearsky GHI stat smooth'], 'Clearsky GHI stat smooth')
vis.add_circle_ser(ground.df[ground.df['sky_status pvlib'] == 1]['GHI'], 'pvlib clear')


# In[ ]:


vis.show()


# In[ ]:


utils.mean_abs_diff(ground.df['Clearsky GHI pvlib'], ground.df['Clearsky GHI stat smooth'])


# Smoothing did not make the statistical curve fit the pvlib curve dramatically better.  In fact, it's almost negligible.  The smoothness is what we desired though, and it looks to much more closely resemble the statistical curve in that respect.

# ## Ground and NSRDB

# In[ ]:


ground.intersection(nsrdb.df.index)


# In[ ]:


nsrdb.intersection(ground.df.index)


# In[ ]:


vis = visualize.Visualizer()


# In[ ]:


vis.add_line_ser(ground.df['GHI'], 'Ground GHI')
vis.add_line_ser(nsrdb.df['GHI'], 'NSRDB GHI')

vis.add_line_ser(ground.df['Clearsky GHI pvlib'], 'PVLib GHI_cs')  # PVLib clearsky will be the same for both (used same location)
vis.add_line_ser(nsrdb.df['Clearsky GHI'], 'NSRDB GHI_cs')

vis.add_line_ser(ground.df['Clearsky GHI stat smooth'], 'Ground GHI_cs smooth')
vis.add_line_ser(nsrdb.df['Clearsky GHI stat smooth'], 'NSRDB GHI_cs smooth')


# In[ ]:


vis.show()


# All measurements seem to match well.  Smoothing 1min and 30min data agrees well here.  Care should be taken when selecting window sizes in future for different data frequencies.

# In[ ]:


utils.mean_abs_diff(ground.df['GHI'], nsrdb.df['GHI'])


# In[ ]:


utils.mean_abs_diff(ground.df['Clearsky GHI pvlib'], nsrdb.df['Clearsky GHI'])


# In[ ]:


utils.mean_abs_diff(ground.df['Clearsky GHI stat smooth'], nsrdb.df['Clearsky GHI stat smooth'])


# # Wrap up

# Add the smoothed statistical clearsky to the original data frames and dump to file.

# In[ ]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb.pkl')


# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground.pkl')


# In[ ]:


nsrdb.robust_rolling_smooth('Clearsky GHI stat', 3)


# In[ ]:


ground.robust_rolling_smooth('Clearsky GHI stat', 60)


# In[ ]:


nsrdb.to_pickle('abq_nsrdb_1.pkl', overwrite=True)


# In[ ]:


ground.to_pickle('abq_ground_1.pkl', overwrite=True)


# In[ ]:




