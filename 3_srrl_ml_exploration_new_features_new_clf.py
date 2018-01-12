
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
import xgboost as xgb

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


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')

nsrdb.trim_dates('01-01-2014', '01-01-2016')
# In[3]:


nsrdb.time_from_solar_noon('Clearsky GHI', 'tfn')


# In[4]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')

ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nsrdb.scale_by_day()
# In[5]:


ground.df.index


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[6]:


ground.intersection(nsrdb.df.index)


# In[7]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[8]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)


# In[9]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)

for i in ground.df.keys():
    print("'" + i + "',")
# In[10]:


feature_cols = [
'tfn',
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
'GHI Clearsky GHI pvlib gradient ratio', 
'GHI Clearsky GHI pvlib gradient ratio mean', 
'GHI Clearsky GHI pvlib gradient ratio std', 
'GHI Clearsky GHI pvlib gradient ratio min', 
'GHI Clearsky GHI pvlib gradient ratio max', 
'GHI Clearsky GHI pvlib gradient ratio range', 
'GHI Clearsky GHI pvlib gradient second ratio', 
'GHI Clearsky GHI pvlib gradient second ratio mean', 
'GHI Clearsky GHI pvlib gradient second ratio std', 
'GHI Clearsky GHI pvlib gradient second ratio min', 
'GHI Clearsky GHI pvlib gradient second ratio max', 
'GHI Clearsky GHI pvlib gradient second ratio range',
'GHI Clearsky GHI pvlib line length ratio',
'GHI Clearsky GHI pvlib line length ratio gradient',
'GHI Clearsky GHI pvlib line length ratio gradient second',
# 'abs_ideal_ratio_diff pct_change', 
# 'abs_ideal_ratio_diff pct_change mean', 
# 'abs_ideal_ratio_diff pct_change std', 
# 'abs_ideal_ratio_diff pct_change max', 
# 'abs_ideal_ratio_diff pct_change min', 
# 'abs_ideal_ratio_diff pct_change range'
]

target_cols = ['sky_status']


# In[11]:


ground.trim_dates('10-01-2010', '10-08-2010')


# In[12]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')

vis = visualize.Visualizer()
vis.add_line_ser(ground.df['Clearsky GHI pvlib'], 'GHIcs')
vis.add_line_ser(ground.df['tfn'], 'tfn')
vis.show()
# In[13]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[14]:


from sklearn import ensemble, linear_model
# clf = ensemble.RandomForestClassifier(class_weight='balanced', n_estimators=100, max_leaf_nodes=40)  # max_leaf_nodes=30, n_estimators=100)
# clf = tree.DecisionTreeClassifier(min_samples_leaf=.001)
# clf = linear_model.LogisticRegression(C=.05)
# clf = ensemble.RandomForestClassifier(class_weight='balanced', min_samples_leaf=.01, n_estimators=24, n_jobs=-1)
clf = ensemble.RandomForestClassifier(class_weight='balanced', min_samples_leaf=.00275, n_estimators=64, n_jobs=-1)
clf = ensemble.GradientBoostingClassifier(learning_rate=.01, n_estimators=100)


# In[15]:


import xgboost as xgb


# In[16]:


# clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=.0075, nthread=4, min_child_weight=1)
clf = xgb.XGBClassifier(max_depth=4, n_estimators=325, learning_rate=.01, nthread=4, min_child_weight=1)


# In[17]:


# nsrdb.df = nsrdb.df[nsrdb.df['GHI'] > 0]


# In[18]:


# train.df = train.df[train.df['Clearsky GHI pvlib'] > 0]


# In[19]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[20]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
pred = pred.astype(bool)


# In[21]:


train.intersection(test.df.index)

cm = metrics.confusion_matrix(train.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])metrics.f1_score(train.df['sky_status'].values, pred)
# In[22]:


vis = visualize.Visualizer()


# In[23]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[24]:


vis.show()


# In[25]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[26]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[27]:


ground.trim_dates('10-01-2010', '10-15-2010')


# In[28]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')

ground.scale_by_day()
# In[29]:


test = ground


# In[30]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 11, smooth=False, tol=1.0e-6)
pred = pred.astype(bool)


# In[31]:


vis = visualize.Visualizer()


# In[32]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[33]:


vis.show()


# In[34]:


for f, i in zip(feature_cols, clf.feature_importances_):
    print(f, i)

tree.export_graphviz(clf, 'abq.dot', feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)feature_cols = [
# 'GHI',
# 'Clearsky GHI pvlib',
'tfn',
# 'GHI mean',
# 'GHI std',
# 'GHI max',
# 'GHI min',
# 'GHI range',
# 'Clearsky GHI pvlib mean',
# 'Clearsky GHI pvlib std',
# 'Clearsky GHI pvlib max',
# 'Clearsky GHI pvlib min',
# 'Clearsky GHI pvlib range',
# 'GHI gradient',
# 'GHI gradient mean',
# 'GHI gradient std',
# 'GHI gradient max',
# 'GHI gradient min',
# 'GHI gradient range',
# 'GHI gradient second',
# 'GHI gradient second mean',
# 'GHI gradient second std',
# 'GHI gradient second max',
# 'GHI gradient second min',
# 'GHI gradient second range',
# 'Clearsky GHI pvlib gradient',
# 'Clearsky GHI pvlib gradient mean',
# 'Clearsky GHI pvlib gradient std',
# 'Clearsky GHI pvlib gradient max',
# 'Clearsky GHI pvlib gradient min',
# 'Clearsky GHI pvlib gradient second',
# 'Clearsky GHI pvlib gradient second mean',
# 'Clearsky GHI pvlib gradient second std',
# 'Clearsky GHI pvlib gradient second max',
# 'Clearsky GHI pvlib gradient second min',
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
# 'abs_diff',
# 'abs_diff mean',
# 'abs_diff std',
# 'abs_diff max',
# 'abs_diff min',
# 'abs_diff range',
# 'abs_diff gradient',
# 'abs_diff gradient mean',
# 'abs_diff gradient std',
# 'abs_diff gradient max',
# 'abs_diff gradient min',
# 'abs_diff gradient range',
# 'abs_diff gradient second',
# 'abs_diff gradient second mean',
# 'abs_diff gradient second std',
# 'abs_diff gradient second max',
# 'abs_diff gradient second min',
# 'abs_diff gradient second range',
# 'GHI line length',
# 'Clearsky GHI pvlib line length',
'GHI Clearsky GHI pvlib line length ratio',
'GHI Clearsky GHI pvlib line length ratio',
# 'GHI bpct change',
# 'GHI bpct change mean', 'GHI bpct change std', 'GHI bpct change max', 'GHI bpct change min', 'GHI bpct change range',
# 'Clearsky GHI pvlib bpct change',
# 'Clearsky GHI pvlib bpct change mean', 'Clearsky GHI pvlib bpct change std', 'Clearsky GHI pvlib bpct change max', 'Clearsky GHI pvlib bpct change min', 'Clearsky GHI pvlib bpct change range',
# 'GHI Clearsky GHI pvlib bpct change ratio', 
# 'GHI Clearsky GHI pvlib bpct change ratio mean', 'GHI Clearsky GHI pvlib bpct change ratio std', 'GHI Clearsky GHI pvlib bpct change ratio max', 'GHI Clearsky GHI pvlib bpct change ratio min', 'GHI Clearsky GHI pvlib bpct change ratio range' 
]

target_cols = ['sky_status']
# In[ ]:




