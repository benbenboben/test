
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Import-and-setup-data" data-toc-modified-id="Import-and-setup-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import and setup data</a></div><div class="lev1 toc-item"><a href="#Train-model" data-toc-modified-id="Train-model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train model</a></div><div class="lev1 toc-item"><a href="#Test-on-ground-data" data-toc-modified-id="Test-on-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Test on ground data</a></div><div class="lev2 toc-item"><a href="#SRRL" data-toc-modified-id="SRRL-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>SRRL</a></div><div class="lev2 toc-item"><a href="#Sandia-RTC" data-toc-modified-id="Sandia-RTC-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Sandia RTC</a></div><div class="lev2 toc-item"><a href="#ORNL" data-toc-modified-id="ORNL-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>ORNL</a></div>

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

from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model

import pvlib
import cs_detection

import visualize_plotly as visualize

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')

import warnings
warnings.filterwarnings('ignore')


# # Import and setup data

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[2]:


nsrdb_srrl = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
nsrdb_srrl.df.index = nsrdb_srrl.df.index.tz_convert('MST')
nsrdb_srrl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb_abq = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb_abq.df.index = nsrdb_abq.df.index.tz_convert('MST')
nsrdb_abq.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb_ornl = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb_ornl.df.index = nsrdb_ornl.df.index.tz_convert('EST')
nsrdb_ornl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# # Train model

# * Train model on all available NSRBD data
#     * ORNL
#     * Sandia RTC
#     * SRRL
# 
# 1. Scale model clearsky (PVLib)
# 2. Calculate training metrics
# 3. Train model

# In[3]:


nsrdb_srrl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_abq.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_ornl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[4]:


utils.calc_all_window_metrics(nsrdb_srrl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_abq.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_ornl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


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

vis = visualize.Visualizer()
vis.plot_corr_matrix(nsrdb_srrl.df[feature_cols].corr().values, labels=feature_cols)
# In[6]:


best_params = {'max_depth': 4, 'n_estimators': 128}


# In[7]:


clf = ensemble.RandomForestClassifier(**best_params, n_jobs=-1)


# In[8]:


X = np.vstack((nsrdb_srrl.df[feature_cols].values, 
               nsrdb_abq.df[feature_cols].values,
               nsrdb_ornl.df[feature_cols].values))
y = np.vstack((nsrdb_srrl.df[target_cols].values, 
               nsrdb_abq.df[target_cols].values,
               nsrdb_ornl.df[target_cols].values))

vis = visualize.Visualizer()
vis.plot_corr_matrix(nsrdb_srrl.df[feature_cols].corr().values, labels=feature_cols)
# In[ ]:





# In[9]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X, y.flatten())')


# # Test on ground data

# ## SRRL

# In[10]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[11]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[12]:


ground.trim_dates('10-01-2011', '10-16-2011')


# In[13]:


ground.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')


# In[14]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[15]:


test = ground


# In[16]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[17]:


vis = visualize.Visualizer()


# In[18]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[19]:


vis.show()


# In[20]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[21]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[22]:


ground.trim_dates('10-01-2011', '10-16-2011')


# In[23]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[24]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[25]:


test= ground


# In[26]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[27]:


vis = visualize.Visualizer()


# In[28]:


srrl_tmp = cs_detection.ClearskyDetection(nsrdb_srrl.df)
srrl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[29]:


vis.show()


# In[ ]:





# ## Sandia RTC

# In[30]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[31]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[32]:


ground.trim_dates('10-01-2015', '10-16-2015')


# In[33]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[34]:


test = ground


# In[35]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[36]:


vis = visualize.Visualizer()


# In[37]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[38]:


vis.show()


# In[39]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[40]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[41]:


ground.trim_dates('10-01-2015', '10-16-2015')


# In[42]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[43]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[44]:


test= ground


# In[45]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[46]:


vis = visualize.Visualizer()


# In[47]:


abq_tmp = cs_detection.ClearskyDetection(nsrdb_abq.df)
abq_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[48]:


vis.show()


# ## ORNL

# In[49]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[50]:


ground.trim_dates('10-01-2008', '10-16-2008')


# In[51]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[52]:


ground.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')


# In[53]:


test = ground


# In[54]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[55]:


vis = visualize.Visualizer()


# In[56]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[57]:


vis.show()


# In[58]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[59]:


ground.df.index = ground.df.index.tz_convert('EST')


# In[60]:


ground.trim_dates('10-01-2008', '10-16-2008')


# In[61]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[62]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[63]:


test= ground


# In[64]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[65]:


vis = visualize.Visualizer()


# In[66]:


ornl_tmp = cs_detection.ClearskyDetection(nsrdb_ornl.df)
ornl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[67]:


vis.show()


# In[68]:


vis = visualize.Visualizer()
vis.add_bar(feature_cols, clf.feature_importances_)
vis.show()


# In[69]:


import pickle


# In[70]:


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[71]:


with open('trained_model.pkl', 'rb') as f:
    new_clf = pickle.load(f)


# In[72]:


new_clf is clf


# In[73]:


clf.get_params()


# In[74]:


new_clf.get_params()


# In[ ]:




