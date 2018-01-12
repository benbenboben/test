
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data" data-toc-modified-id="Train/test-on-NSRDB-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data</a></div>

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
import cs_detection_refactor as cs_detection
# import cs_detection_refactor
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


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')

nsrdb.trim_dates('01-01-2014', '01-01-2016')
# In[3]:


nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[4]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')


# In[5]:


feature_cols = [
'tfn',
'abs_ideal_ratio_diff',
'abs_ideal_ratio_diff mean',
'abs_ideal_ratio_diff std',
'abs_ideal_ratio_diff max',
'abs_ideal_ratio_diff min',
# 'abs_ideal_ratio_diff range',
# 'abs_ideal_ratio_diff gradient',
# 'abs_ideal_ratio_diff gradient mean',
# 'abs_ideal_ratio_diff gradient std',
# 'abs_ideal_ratio_diff gradient max',
# 'abs_ideal_ratio_diff gradient min',
# 'abs_ideal_ratio_diff gradient range',
# 'abs_ideal_ratio_diff gradient second',
# 'abs_ideal_ratio_diff gradient second mean',
# 'abs_ideal_ratio_diff gradient second std',
# 'abs_ideal_ratio_diff gradient second max',
# 'abs_ideal_ratio_diff gradient second min',
# 'abs_ideal_ratio_diff gradient second range',
'GHI Clearsky GHI pvlib gradient ratio', 
'GHI Clearsky GHI pvlib gradient ratio mean', 
'GHI Clearsky GHI pvlib gradient ratio std', 
'GHI Clearsky GHI pvlib gradient ratio min', 
'GHI Clearsky GHI pvlib gradient ratio max', 
# 'GHI Clearsky GHI pvlib gradient ratio range', 
'GHI Clearsky GHI pvlib gradient second ratio', 
'GHI Clearsky GHI pvlib gradient second ratio mean', 
'GHI Clearsky GHI pvlib gradient second ratio std', 
'GHI Clearsky GHI pvlib gradient second ratio min', 
'GHI Clearsky GHI pvlib gradient second ratio max', 
# 'GHI Clearsky GHI pvlib gradient second ratio range',
'GHI Clearsky GHI pvlib line length ratio',
'GHI Clearsky GHI pvlib line length ratio gradient',
'GHI Clearsky GHI pvlib line length ratio gradient second'
]

target_cols = ['sky_status']


# # Train/test on NSRDB data

# In[6]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates('01-01-2010', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[7]:


import xgboost as xgb
# clf = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=.01, nthread=4, min_child_weight=1)
# clf = xgb.XGBClassifier(max_depth=4, n_estimators=400, learning_rate=.01, nthread=4)
clf = xgb.XGBClassifier(**{'max_depth': 4, 'n_estimators': 500, 'learning_rate': 0.005, 'reg_lambda': 1}, n_jobs=4)


# In[8]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[9]:


train.calc_all_window_metrics(train.df, 3, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)

from scipy import statsn_iter = 10
for _ in range(n_iter):
    max_depth = np.random.choice([3, 4, 5, 6])
    n_estimators = np.random.randint(100, 1001)
    learning_rate = np.random.uniform(.001, .011)
    clf = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=4)
    clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())
    pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
    pred = pred.astype(bool)
    print('max_depth: {}, n_estimators: {}, learning_rate: {}'.format(max_depth, n_estimators, learning_rate))
    print('\t {}'.format(metrics.f1_score(test.df['sky_status'], pred)))
    

# param_probs = {'max_depth': [3, 4, 5], 'n_estimators': np.random.randint(100, 1001)}#, 'learning_rate': np.random.uniform(.001, .2)}
# param_probs = {'max_depth': [3, 4, 5], 'n_estimators': stats.uniform(scale=1000), 'learning_rate': stats.uniform(scale)}from sklearn import model_selectionrscv = model_selection.RandomizedSearchCV(clf, param_probs, n_iter=2)rscv.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())clf = xgb.XGBClassifier(**{'max_depth': 5, 'n_estimators': 264, 'learning_rate': 0.00616078554624614})
# In[10]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[11]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
pred = pred.astype(bool)


# In[12]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.show()


# In[13]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[14]:


print(metrics.f1_score(test.df['sky_status'].values, pred))


# 

# In[15]:


train = cs_detection.ClearskyDetection(nsrdb.df)
# train.trim_dates('01-01-2010', None)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.calc_all_window_metrics(train.df, 3, col1='GHI', col2='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[16]:


test = cs_detection.ClearskyDetection(ground.df)


# In[17]:


test.intersection(train.df.index)


# In[18]:


test.trim_dates('10-08-2015', '10-16-2015')


# In[19]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[ ]:





# In[20]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
pred = pred.astype(bool)


# In[21]:


train.intersection(test.df.index)


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


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[26]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[27]:


ground.trim_dates('10-01-2015', '10-15-2015')


# In[28]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')

ground.scale_by_day()
# In[29]:


test = ground

n_iter = 30
for _ in range(n_iter):
    max_depth = np.random.choice([3, 4, 5, 6])
    n_estimators = np.random.randint(100, 1001)
    learning_rate = np.random.uniform(.001, .011)
    # reg_lambda = np.random.uniform(.001, 11)
    reg_lambda = 1
    clf = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, reg_lambda=reg_lambda, n_jobs=4)
    clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())
    pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 60)
    pred = pred.astype(bool)
    print('max_depth: {}, n_estimators: {}, learning_rate: {}, reg_lambda: {}'.format(max_depth, n_estimators, learning_rate, reg_lambda))
    print('\t {}'.format(metrics.f1_score(test.df['sky_status pvlib'], pred)))
    

# param_probs = {'max_depth': [3, 4, 5], 'n_estimators': np.random.randint(100, 1001)}#, 'learning_rate': np.random.uniform(.001, .2)}
# param_probs = {'max_depth': [3, 4, 5], 'n_estimators': stats.uniform(scale=1000), 'learning_rate': stats.uniform(scale)}clf = xgb.XGBClassifier(**{'max_depth': 4, 'n_estimators': 500, 'learning_rate': 0.005, 'reg_lambda': 1}, n_jobs=4)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())test.df['GHI'] = test.df['Clearsky GHI pvlib'].copy()
# In[65]:


get_ipython().run_cell_magic('time', '', 'import warnings\nwarnings.simplefilter("ignore")\npred = test.iter_predict_daily(feature_cols, \'GHI\', \'Clearsky GHI pvlib\', clf, 60)\npred = pred.astype(bool)')

pred.to_csv('tmp_old.csv')
# In[66]:


vis = visualize.Visualizer()


# In[67]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'], '|1 - ratio|')


# In[68]:


vis.show()


# In[69]:


cm = metrics.confusion_matrix(test.df['sky_status pvlib'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[70]:


print(metrics.f1_score(test.df['sky_status pvlib'].values, pred))


# In[71]:


for f, i in zip(feature_cols, clf.feature_importances_):
    print(f, i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




