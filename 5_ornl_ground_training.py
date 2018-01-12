
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Read-and-process-data" data-toc-modified-id="Read-and-process-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read and process data</a></div><div class="lev1 toc-item"><a href="#Align-date-ranges" data-toc-modified-id="Align-date-ranges-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Align date ranges</a></div>

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree

import pytz
import itertools
import visualize
import utils
import pydotplus
import xgboost as xgb

from sklearn import metrics

import pvlib
import cs_detection

import visualize_plotly as visualize

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Read and process data

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[2]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('EST')


# In[3]:


nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[4]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('EST')


# In[5]:


feature_cols = [
'tfn',
'abs_ideal_ratio_diff',
'abs_ideal_ratio_diff mean',
'abs_ideal_ratio_diff std',
'abs_ideal_ratio_diff max',
'abs_ideal_ratio_diff min',
'GHI Clearsky GHI pvlib gradient ratio', 
'GHI Clearsky GHI pvlib gradient ratio mean', 
'GHI Clearsky GHI pvlib gradient ratio std', 
'GHI Clearsky GHI pvlib gradient ratio min', 
'GHI Clearsky GHI pvlib gradient ratio max', 
'GHI Clearsky GHI pvlib gradient second ratio', 
'GHI Clearsky GHI pvlib gradient second ratio mean', 
'GHI Clearsky GHI pvlib gradient second ratio std', 
'GHI Clearsky GHI pvlib gradient second ratio min', 
'GHI Clearsky GHI pvlib gradient second ratio max', 
'GHI Clearsky GHI pvlib line length ratio',
'GHI Clearsky GHI pvlib line length ratio gradient',
'GHI Clearsky GHI pvlib line length ratio gradient second'
]

target_cols = ['sky_status']


# # Align date ranges

# In[6]:


ground.df.index[0], ground.df.index[-1]


# In[7]:


nsrdb.df.index[0], nsrdb.df.index[-1]


# In[8]:


ground2 = cs_detection.ClearskyDetection(ground.df)


# In[9]:


ground2.trim_dates('01-01-2008', '08-01-2010')
ground2.df = ground2.df[ground2.df.index.minute % 30 == 0]


# In[10]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df)


# In[11]:


nsrdb2.trim_dates('01-01-2008', '08-01-2010')
nsrdb2.df = nsrdb2.df[nsrdb2.df.index.minute % 30 == 0]


# In[12]:


vis = visualize.Visualizer()
vis.add_line_ser(ground2.df['GHI'], 'Grnd GHI')
vis.add_line_ser(ground2.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
vis.add_line_ser(nsrdb2.df['GHI'], 'NSRDB GHI')
vis.add_line_ser(nsrdb2.df['Clearsky GHI pvlib'], 'NSRDB GHIcs')
vis.show()


# In[13]:


print(np.sqrt(np.mean((nsrdb2.df['GHI'] - ground2.df['GHI'])**2)))


# In[14]:


print(np.sqrt(np.mean((nsrdb2.df['GHI'] - nsrdb2.df['Clearsky GHI pvlib'])**2)))


# In[15]:


print(np.sqrt(np.mean((ground2.df['GHI'] - ground2.df['Clearsky GHI pvlib'])**2)))


# In[16]:


ground2 = cs_detection.ClearskyDetection(ground.df)


# In[17]:


ground2.trim_dates('01-01-2008', '08-01-2010')


# In[18]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df)


# In[19]:


nsrdb2.trim_dates('01-01-2008', '08-01-2010')


# In[20]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[21]:


ground2.df['sky_status'].head()


# In[22]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[23]:


ground2.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[24]:


ground3 = cs_detection.ClearskyDetection(ground2.df)
ground3.trim_dates('07-01-2010', '07-15-2010')


# In[25]:


vis = visualize.Visualizer()
vis.add_line_ser(ground3.df['GHI'], 'Grnd GHI')
vis.add_line_ser(ground3.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
vis.add_circle_ser(ground3.df[ground3.df['sky_status']]['GHI'], 'Clear')
vis.show()


# In[27]:


utils.calc_all_window_metrics(ground2.df, 11, 'GHI', 'Clearsky GHI pvlib', overwrite=True)


# In[28]:


ground2.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[29]:


ground2_train = ground2.df[ground2.df.index.minute % 30 == 0]


# In[30]:


ground2_train = ground2_train[ground2_train.index < '07-01-2010']


# In[31]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
# clf = ensemble.RandomForestClassifier(n_estimators=100)
# clf = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=.005, reg_lambda=.01)
clf = ensemble.RandomForestClassifier(max_depth=6, n_estimators=128, n_jobs=-1)


# In[32]:


from sklearn import model_selection


# In[33]:


scores = model_selection.cross_val_score(clf, ground2_train[feature_cols].values, ground2_train[target_cols].values.flatten())


# In[34]:


np.mean(scores), np.std(scores)


# In[35]:


clf.fit(ground2_train[feature_cols].values, ground2_train[target_cols].values.flatten())


# In[36]:


ground2_test = cs_detection.ClearskyDetection(ground2.df)


# In[37]:


ground2_test.trim_dates('07-01-2010', '07-15-2010')


# In[38]:


pred = ground2_test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 11)


# In[39]:


pred2 = pred[pred.index.minute % 30 == 0]


# In[40]:


sky_stat = ground2_test.df[ground2_test.df.index.minute % 30 == 0]['sky_status']


# In[41]:


cm = metrics.confusion_matrix(sky_stat, pred2)


# In[42]:


vis = visualize.Visualizer()


# In[43]:


vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[44]:


ghi = ground2_test.df['GHI']
ghi_cs = ground2_test.df['Clearsky GHI pvlib']
is_clear = ground2_test.df['sky_status iter']
sky = ground2_test.df['sky_status']


# In[45]:


ghi = ghi[(ghi.index < '07-15-2010') & (ghi.index >= '07-01-2010')]
ghi_cs = ghi_cs[(ghi_cs.index < '07-15-2010') & (ghi_cs.index >= '07-01-2010')]
sky = sky[(sky.index < '07-15-2010') & (sky.index >= '07-01-2010')]


# In[46]:


vis = visualize.Visualizer()
vis.add_line_ser(ghi, 'ghi')
vis.add_line_ser(ghi_cs, 'ghi_cs')
vis.add_circle_ser(ghi[is_clear], 'clear')
vis.add_circle_ser(ghi[sky], 'nsrdb')
# vis.add_circle_ser(ghi[ghi.index.isin(sky.index)], 'clear nsrdb')
vis.show()


# In[ ]:





# In[ ]:




