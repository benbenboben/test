
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Import-and-setup-data" data-toc-modified-id="Import-and-setup-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import and setup data</a></div><div class="lev1 toc-item"><a href="#Train-model" data-toc-modified-id="Train-model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train model</a></div><div class="lev1 toc-item"><a href="#Test-on-ground-data" data-toc-modified-id="Test-on-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Test on ground data</a></div><div class="lev2 toc-item"><a href="#SRRL" data-toc-modified-id="SRRL-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>SRRL</a></div><div class="lev2 toc-item"><a href="#Sandia-RTC" data-toc-modified-id="Sandia-RTC-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Sandia RTC</a></div><div class="lev2 toc-item"><a href="#ORNL" data-toc-modified-id="ORNL-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>ORNL</a></div>

# In[11]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree
import xgboost as xgb

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
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')

import warnings
warnings.filterwarnings('ignore')


# # Import and setup data

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[12]:


nsrdb_srrl = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
nsrdb_srrl.df.index = nsrdb_srrl.df.index.tz_convert('MST')
nsrdb_srrl.scale_by_irrad('Clearsky GHI pvlib')
nsrdb_srrl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb_abq = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb_abq.df.index = nsrdb_abq.df.index.tz_convert('MST')
nsrdb_abq.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb_abq.scale_by_irrad('Clearsky GHI pvlib')
nsrdb_ornl = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb_ornl.df.index = nsrdb_ornl.df.index.tz_convert('EST')
nsrdb_ornl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb_ornl.scale_by_irrad('Clearsky GHI pvlib')


# # Train model

# * Train model on all available NSRBD data
#     * ORNL
#     * Sandia RTC
#     * SRRL
# 
# 1. Scale model clearsky (PVLib)
# 2. Calculate training metrics
# 3. Train model

# In[13]:


nsrdb_srrl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_abq.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_ornl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[14]:


utils.calc_all_window_metrics(nsrdb_srrl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_abq.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_ornl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[20]:


feature_cols = [
'ghi_status',
'tfn',
'abs_ideal_ratio_diff',
'abs_ideal_ratio_diff mean',
'abs_ideal_ratio_diff std',
# 'abs_ideal_ratio_diff max',
# 'abs_ideal_ratio_diff min',
'GHI Clearsky GHI pvlib gradient ratio', 
'GHI Clearsky GHI pvlib gradient ratio mean', 
'GHI Clearsky GHI pvlib gradient ratio std', 
# 'GHI Clearsky GHI pvlib gradient ratio min', 
# 'GHI Clearsky GHI pvlib gradient ratio max', 
'GHI Clearsky GHI pvlib gradient second ratio', 
'GHI Clearsky GHI pvlib gradient second ratio mean', 
'GHI Clearsky GHI pvlib gradient second ratio std', 
# 'GHI Clearsky GHI pvlib gradient second ratio min', 
# 'GHI Clearsky GHI pvlib gradient second ratio max', 
'GHI Clearsky GHI pvlib line length ratio',
# 'GHI Clearsky GHI pvlib line length ratio gradient',
# 'GHI Clearsky GHI pvlib line length ratio gradient second'
]

target_cols = ['sky_status']

vis = visualize.Visualizer()
vis.plot_corr_matrix(nsrdb_srrl.df[feature_cols].corr().values, labels=feature_cols)
# In[21]:


# best_params = {'max_depth': 5, 'n_estimators': 256}
# best_params = {'max_depth': 4, 'n_estimators': 256, 'class_weight': 'balanced'}
# best_params = {'max_depth': 4, 'n_estimators': 256, 'class_weight': None}
best_params = {'max_depth': 4, 'n_estimators': 256, 'class_weight': None}


# In[22]:


clf = ensemble.RandomForestClassifier(**best_params, n_jobs=-1)


# In[23]:


X = np.vstack((nsrdb_srrl.df[feature_cols].values, 
               nsrdb_abq.df[feature_cols].values,
               nsrdb_ornl.df[feature_cols].values))
y = np.vstack((nsrdb_srrl.df[target_cols].values, 
               nsrdb_abq.df[target_cols].values,
               nsrdb_ornl.df[target_cols].values))

vis = visualize.Visualizer()
vis.plot_corr_matrix(nsrdb_srrl.df[feature_cols].corr().values, labels=feature_cols)
# In[24]:


print(int(X.shape[0] / 3) * 1000)


# In[25]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X, y.flatten())')


# # Test on ground data

# ## SRRL

# In[109]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[110]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[111]:


ground.trim_dates('10-01-2011', '10-08-2011')


# In[112]:


ground.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')


# In[113]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[114]:


test = ground


# In[115]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[116]:


vis = visualize.Visualizer()


# In[117]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[118]:


vis.show()


# In[119]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# In[120]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[121]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[122]:


ground.trim_dates('10-01-2011', '10-08-2011')


# In[123]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[124]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[125]:


test= ground


# In[126]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[127]:


vis = visualize.Visualizer()


# In[128]:


srrl_tmp = cs_detection.ClearskyDetection(nsrdb_srrl.df)
srrl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[129]:


vis.show()

probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 10, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])
# In[130]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## Sandia RTC

# In[131]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[132]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[133]:


ground.trim_dates('10-01-2015', '10-08-2015')


# In[134]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[135]:


test = ground


# In[136]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[137]:


vis = visualize.Visualizer()


# In[138]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[139]:


vis.show()


# In[140]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# In[141]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[142]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[143]:


ground.trim_dates('10-01-2015', '10-08-2015')


# In[144]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[145]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[146]:


test= ground


# In[147]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[148]:


vis = visualize.Visualizer()


# In[149]:


abq_tmp = cs_detection.ClearskyDetection(nsrdb_abq.df)
abq_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[150]:


vis.show()


# In[151]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## ORNL

# In[152]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[153]:


ground.trim_dates('10-01-2008', '10-08-2008')


# In[154]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[155]:


ground.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')


# In[156]:


test = ground


# In[157]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[158]:


vis = visualize.Visualizer()


# In[159]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[160]:


vis.show()


# In[161]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# In[173]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[174]:


ground.df.index = ground.df.index.tz_convert('EST')


# In[175]:


ground.trim_dates('10-01-2008', '10-08-2008')


# In[176]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[177]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[178]:


test= ground


# In[179]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[180]:


vis = visualize.Visualizer()


# In[181]:


ornl_tmp = cs_detection.ClearskyDetection(nsrdb_ornl.df)
ornl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[182]:


vis.show()


# In[172]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# In[ ]:


vis = visualize.Visualizer()
vis.add_bar(feature_cols, clf.feature_importances_)
vis.show()


# In[ ]:


import pickle


# In[ ]:


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:


with open('trained_model.pkl', 'rb') as f:
    new_clf = pickle.load(f)


# In[ ]:


new_clf is clf


# In[ ]:


clf.get_params()


# In[ ]:


new_clf.get_params()


# In[ ]:




