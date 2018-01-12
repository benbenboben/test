
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


nsrdb = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('EST')


# In[3]:


nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[4]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('EST')

ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
# In[ ]:





# We will reduce the frequency of ground based measurements to match NSRDB.

# In[5]:


ground.intersection(nsrdb.df.index)


# In[6]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[7]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)


# In[8]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)

for i in ground.df.keys():
    print("'" + i + "',")
# In[9]:


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
'GHI Clearsky GHI pvlib line length ratio']

target_cols = ['sky_status']


# In[10]:


ground.trim_dates('10-01-2009', '10-15-2009')


# In[11]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')

vis = visualize.Visualizer()
vis.add_line_ser(ground.df['Clearsky GHI pvlib'], 'GHIcs')
vis.add_line_ser(ground.df['tfn'], 'tfn')
vis.show()
# In[12]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[13]:


from sklearn import ensemble
# clf = ensemble.RandomForestClassifier(class_weight='balanced', n_estimators=100, max_leaf_nodes=40)  # max_leaf_nodes=30, n_estimators=100)
clf = tree.DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=.005)
# clf = ensemble.RandomForestClassifier(class_weight='balanced', min_samples_leaf=.0025, n_estimators=100)

# clf = ensemble.RandomForestClassifier(class_weight='balanced', min_samples_leaf=.05, n_estimators=100)
# clf = tree.DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=.005, max_depth=10)


# In[14]:


# nsrdb.df = nsrdb.df[nsrdb.df['GHI'] > 0]


# In[15]:


# train.df = train.df[train.df['Clearsky GHI pvlib'] > 0]


# In[16]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[17]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
pred = pred.astype(bool)


# In[18]:


train.intersection(test.df.index)


# In[19]:


cm = metrics.confusion_matrix(train.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[20]:


metrics.f1_score(train.df['sky_status'].values, pred)


# In[21]:


vis = visualize.Visualizer()


# In[22]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[23]:


vis.show()


# In[24]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[25]:


ground.df.index = ground.df.index.tz_convert('EST')


# In[26]:


ground.trim_dates('11-01-2009', '11-15-2009')


# In[27]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[28]:


test = ground


# In[29]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[30]:


vis = visualize.Visualizer()


# In[31]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[32]:


vis.show()


# In[33]:


tree.export_graphviz(clf, 'ornl3.dot', feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)


# In[ ]:




