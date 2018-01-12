
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#30-min-freq-ground-data" data-toc-modified-id="30-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>30 min freq ground data</a></div><div class="lev2 toc-item"><a href="#15-min-freq-ground-data" data-toc-modified-id="15-min-freq-ground-data-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>15 min freq ground data</a></div><div class="lev2 toc-item"><a href="#10-min-freq-ground-data" data-toc-modified-id="10-min-freq-ground-data-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>10 min freq ground data</a></div><div class="lev2 toc-item"><a href="#5-min-freq-ground-data" data-toc-modified-id="5-min-freq-ground-data-34"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>5 min freq ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-35"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree, ensemble

import pytz
import itertools
import visualize
import utils
import pydotplus
import xgboost as xgb

from sklearn import metrics
from sklearn import model_selection

import pvlib
import cs_detection

import visualize_plotly as visualize
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

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


nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[4]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')


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


# # Train/test on NSRDB data to find optimal parameters

# In[6]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates('01-01-2010', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[7]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[8]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)

param_grid = {'max_depth': [3, 4, 5], 'n_estimators': [200, 300, 400], 'learning_rate': [.1, .01, .001]}import itertools
import warnings
best_score = 0
best_params = {}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for depth, n_est, lr in itertools.product(param_grid['max_depth'], param_grid['n_estimators'], param_grid['learning_rate']):
        clf = xgb.XGBClassifier(max_depth=depth, n_estimators=n_est, learning_rate=lr, n_jobs=4)
        clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())
        test = cs_detection.ClearskyDetection(nsrdb.df)
        test.trim_dates('06-01-2015', None)    
        pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=False)
        score = metrics.f1_score(test.df['sky_status'], pred)
        indicator = ''
        if score > best_score:
            best_score = score
            best_params['max_depth'] = depth
            best_params['n_estimators'] = n_est
            best_params['learnin_rate'] = lr
            indicator = '*'
        print('max_depth: {}, n_estimators: {}, learning_rate: {}, accuracy: {} {}'.format(depth, n_est, lr, score, indicator))max_depth: 5, n_estimators: 300, learning_rate: 0.1, accuracy: 0.9149868536371605 *
# In[9]:


best_params = {'max_depth': 6, 'n_estimators': 128}


# In[10]:


# clf = xgb.XGBClassifier(**best_params, n_jobs=4)
clf = ensemble.RandomForestClassifier(**best_params, n_jobs=-1)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[11]:


test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[12]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)")


# In[13]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.show()


# In[14]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[15]:


print(metrics.f1_score(test.df['sky_status'].values, pred))


# # Train on all NSRDB data, test various freq of ground data

# In[16]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[17]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## 30 min freq ground data

# In[18]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[19]:


test.trim_dates('10-01-2011', '10-16-2011')


# In[20]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[21]:


test.df = test.df[test.df.index.minute % 30 == 0]


# In[22]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[23]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.intersection(test.df.index)


# In[24]:


nsrdb_clear = train2.df['sky_status'].values
ml_clear = pred
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# ## 15 min freq ground data

# In[25]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[26]:


test.trim_dates('10-01-2011', '10-16-2011')


# In[27]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[28]:


test.df = test.df[test.df.index.minute % 15 == 0]


# In[29]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 5, by_day=True, multiproc=True).astype(bool)


# In[30]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2011', '10-16-2011')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='15min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[31]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# ## 10 min freq ground data

# In[32]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[33]:


test.trim_dates('10-01-2011', '10-16-2011')


# In[34]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[35]:


test.df = test.df[test.df.index.minute % 10 == 0]


# In[36]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 7, by_day=True, multiproc=True).astype(bool)


# In[37]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2011', '10-16-2011')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='10min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[38]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# ## 5 min freq ground data

# In[39]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[40]:


test.trim_dates('10-01-2011', '10-16-2011')


# In[41]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[42]:


test.df = test.df[test.df.index.minute % 5 == 0]


# In[43]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 13, by_day=True, multiproc=True).astype(bool)


# In[44]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2011', '10-16-2011')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='5min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[45]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# ## 1 min freq ground data

# In[46]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[47]:


test.trim_dates('10-01-2011', '10-16-2011')


# In[48]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[49]:


test.df = test.df[test.df.index.minute % 1 == 0]


# In[50]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True).astype(bool)


# In[51]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2011', '10-16-2011')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='1min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[52]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)
vis.show()


# # Save model

# In[53]:


import pickle


# In[54]:


with open('srrl_trained.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[55]:


get_ipython().system('ls srrl*')


# # Conclusion

# In general, the clear sky identification looks good.  At lower frequencies (30 min, 15 min) we see good agreement with NSRDB labeled points.  I suspect this could be further improved my doing a larger hyperparameter search, or even doing some feature extraction/reduction/additions.  

# In[ ]:




