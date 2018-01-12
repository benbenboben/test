
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div>

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
# import visualize
# from bokeh.plotting import output_notebook
# output_notebook()

import visualize_plotly as visualize

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Ground predictions

# ## PVLib Clearsky

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


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[3]:


nsrdb_srrl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_abq.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_ornl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[4]:


nsrdb_srrl.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)
nsrdb_abq.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)
nsrdb_ornl.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)


# In[5]:


feature_cols = [
'GHI',
'Clearsky GHI pvlib',
'tfn',
'GHI mean',
'GHI std',
'GHI max',
'GHI min',
'GHI range',
'Clearsky GHI pvlib mean',
'Clearsky GHI pvlib std',
'Clearsky GHI pvlib max',
'Clearsky GHI pvlib min',
'Clearsky GHI pvlib range',
'GHI gradient',
'GHI gradient mean',
'GHI gradient std',
'GHI gradient max',
'GHI gradient min',
'GHI gradient range',
'GHI gradient second',
'GHI gradient second mean',
'GHI gradient second std',
'GHI gradient second max',
'GHI gradient second min',
'GHI gradient second range',
'Clearsky GHI pvlib gradient',
'Clearsky GHI pvlib gradient mean',
'Clearsky GHI pvlib gradient std',
'Clearsky GHI pvlib gradient max',
'Clearsky GHI pvlib gradient min',
'Clearsky GHI pvlib gradient second',
'Clearsky GHI pvlib gradient second mean',
'Clearsky GHI pvlib gradient second std',
'Clearsky GHI pvlib gradient second max',
'Clearsky GHI pvlib gradient second min',
'abs_ideal_ratio_diff',
'abs_ideal_ratio_diff mean',
'abs_ideal_ratio_diff std',
'abs_ideal_ratio_diff max',
'abs_ideal_ratio_diff min',
'abs_ideal_ratio_diff range',
'abs_ideal_ratio_diff gradient',
'abs_ideal_ratio_diff gradient mean',
'abs_ideal_ratio_diff gradient std',
'abs_ideal_ratio_diff gradient max',
'abs_ideal_ratio_diff gradient min',
'abs_ideal_ratio_diff gradient range',
'abs_ideal_ratio_diff gradient second',
'abs_ideal_ratio_diff gradient second mean',
'abs_ideal_ratio_diff gradient second std',
'abs_ideal_ratio_diff gradient second max',
'abs_ideal_ratio_diff gradient second min',
'abs_ideal_ratio_diff gradient second range',
'abs_diff',
'abs_diff mean',
'abs_diff std',
'abs_diff max',
'abs_diff min',
'abs_diff range',
'abs_diff gradient',
'abs_diff gradient mean',
'abs_diff gradient std',
'abs_diff gradient max',
'abs_diff gradient min',
'abs_diff gradient range',
'abs_diff gradient second',
'abs_diff gradient second mean',
'abs_diff gradient second std',
'abs_diff gradient second max',
'abs_diff gradient second min',
'abs_diff gradient second range',
'GHI line length',
'Clearsky GHI pvlib line length',
'GHI Clearsky GHI pvlib line length ratio',
# 'GHI bpct change',
# 'GHI bpct change mean', 'GHI bpct change std', 'GHI bpct change max', 'GHI bpct change min', 'GHI bpct change range',
# 'Clearsky GHI pvlib bpct change',
# 'Clearsky GHI pvlib bpct change mean', 'Clearsky GHI pvlib bpct change std', 'Clearsky GHI pvlib bpct change max', 'Clearsky GHI pvlib bpct change min', 'Clearsky GHI pvlib bpct change range',
# 'GHI Clearsky GHI pvlib bpct change ratio', 
# 'GHI Clearsky GHI pvlib bpct change ratio mean', 'GHI Clearsky GHI pvlib bpct change ratio std', 'GHI Clearsky GHI pvlib bpct change ratio max', 'GHI Clearsky GHI pvlib bpct change ratio min', 'GHI Clearsky GHI pvlib bpct change ratio range' 
]

target_cols = ['sky_status']


# In[6]:


# clf = ensemble.RandomForestClassifier(class_weight='balanced', n_estimators=64, min_samples_leaf=0.0035, n_jobs=-1)
clf = tree.DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=0.0035)

# from sklearn import model_selection
# gscv = model_selection.GridSearchCV(clf, {'min_samples_leaf': np.arange(.001, .011, .001)}, scoring='f1')


# In[7]:


X = np.vstack((nsrdb_srrl.df[feature_cols].values, 
               nsrdb_abq.df[feature_cols].values,
               nsrdb_ornl.df[feature_cols].values))
y = np.vstack((nsrdb_srrl.df[target_cols].values, 
               nsrdb_abq.df[target_cols].values,
               nsrdb_ornl.df[target_cols].values))

X.nbytesgscv.fit(X, y.flatten())gscv.cv_results_gscv.best_params_clf = gscv.best_estimator_
# In[8]:


clf.fit(X, y.flatten())


# In[9]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[10]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[11]:


ground.trim_dates('10-01-2010', '10-08-2010')


# In[12]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[13]:


test = ground


# In[14]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[15]:


vis = visualize.Visualizer()


# In[16]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[17]:


vis.show()

tree.export_graphviz(clf, 'clf.dot', feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)
# In[18]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[19]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[20]:


ground.trim_dates('10-01-2015', '10-08-2015')


# In[21]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[22]:


test = ground


# In[23]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[24]:


vis = visualize.Visualizer()


# In[25]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[26]:


vis.show()


# In[27]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[28]:


ground.df.index.tz

ground.df.index = ground.df.index.tz_convert('EST')
# In[29]:


ground.trim_dates('05-01-2008', '05-08-2008')


# In[30]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[31]:


test = ground


# In[32]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[33]:


vis = visualize.Visualizer()


# In[34]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[35]:


vis.show()


# In[ ]:





# In[36]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[37]:


ground.df.index.tz

ground.df.index = ground.df.index.tz_convert('EST')
# In[38]:


ground.trim_dates('05-01-2008', '05-08-2008')


# In[39]:


ground.df = ground.df.resample('5min').mean()


# In[40]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[41]:


test = ground


# In[42]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 6, smooth=True)
pred = pred.astype(bool)


# In[43]:


vis = visualize.Visualizer()


# In[44]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(pred)]['GHI'], 'ML+PVLib clear only')


# In[45]:


vis.show()


# In[46]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[47]:


ground.df.index.tz

ground.df.index = ground.df.index.tz_convert('EST')
# In[48]:


ground.trim_dates('05-01-2008', '05-08-2008')


# In[49]:


ground.df = ground.df.resample('10min').mean()


# In[50]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[51]:


test = ground


# In[52]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 6, smooth=True)
pred = pred.astype(bool)


# In[53]:


vis = visualize.Visualizer()


# In[54]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(pred)]['GHI'], 'ML+PVLib clear only')


# In[55]:


vis.show()


# In[56]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[57]:


ground.df.index.tz

ground.df.index = ground.df.index.tz_convert('EST')
# In[58]:


ground.trim_dates('05-01-2008', '05-08-2008')


# In[59]:


ground.df = ground.df.resample('15min').mean()


# In[60]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[61]:


test = ground


# In[62]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 6, smooth=True)
pred = pred.astype(bool)


# In[63]:


vis = visualize.Visualizer()


# In[64]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(pred)]['GHI'], 'ML+PVLib clear only')


# In[65]:


vis.show()


# In[66]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[67]:


ground.df.index.tz

ground.df.index = ground.df.index.tz_convert('EST')
# In[68]:


ground.trim_dates('05-01-2008', '05-08-2008')


# In[69]:


ground.df = ground.df.resample('30min').mean()


# In[70]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[71]:


test = ground


# In[72]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, smooth=True)
pred = pred.astype(bool)


# In[73]:


vis = visualize.Visualizer()


# In[74]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
# vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(pred)]['GHI'], 'ML+PVLib clear only')


# In[75]:


vis.show()


# In[76]:


vis = visualize.Visualizer()
vis.add_bar(feature_cols, clf.feature_importances_)
vis.show()


# In[77]:


len(feature_cols)


# In[ ]:





# In[78]:


tree.export_graphviz(clf, 'dt.dot', feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)


# In[ ]:




