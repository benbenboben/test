
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#30-min-freq-ground-data" data-toc-modified-id="30-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>30 min freq ground data</a></div><div class="lev2 toc-item"><a href="#15-min-freq-ground-data" data-toc-modified-id="15-min-freq-ground-data-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>15 min freq ground data</a></div><div class="lev2 toc-item"><a href="#10-min-freq-ground-data" data-toc-modified-id="10-min-freq-ground-data-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>10 min freq ground data</a></div><div class="lev2 toc-item"><a href="#5-min-freq-ground-data" data-toc-modified-id="5-min-freq-ground-data-34"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>5 min freq ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-35"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

# In[4]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model

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

# In[5]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')

nsrdb.trim_dates('01-01-2014', '01-01-2016')
# In[6]:


nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb.time_from_solar_noon_ratio2('Clearsky GHI pvlib')


# In[7]:


feature_cols = [
'ghi_status',
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

# In[8]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[9]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.time_from_solar_noon_ratio2('Clearsky GHI pvlib')
train.scale_by_irrad('Clearsky GHI pvlib')


# In[ ]:


params={}
params['penalty'] = ['l1', 'l2']
params['alpha'] = [.00001, .0001, .001, .01, .1, 1, 10]
# params['n_estimators'] = [32, 64, 128, 256]
best_score = -1
for p, a in itertools.product(params['penalty'], params['alpha']):
    train2 = cs_detection.ClearskyDetection(train.df)
    train2.trim_dates('01-01-1999', '01-01-2014')
    utils.calc_all_window_metrics(train2.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
    test2 = cs_detection.ClearskyDetection(train.df)
    test2.trim_dates('01-01-2014', '01-01-2015')
    clf = linear_model.SGDClassifier(loss='log', penalty=p, alpha=a)
    clf.fit(train2.df[feature_cols].values, train2.df[target_cols].values.flatten())
    pred = test2.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)
    f1_score = metrics.f1_score(test2.df['sky_status'], pred)
    recall_score = metrics.recall_score(test2.df['sky_status'], pred)
    precision_score = metrics.precision_score(test2.df['sky_status'], pred)
#     if score > best_score:
#         best_params = {}
#         best_params['max_depth'] = depth
#         best_params['n_estimators'] = nest
#         best_score = score
    print(f1_score, recall_score, precision_score, p, a)
# print(score, best_params)


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.time_from_solar_noon_ratio2('Clearsky GHI pvlib')
train.scale_by_irrad('Clearsky GHI pvlib')


# In[11]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[12]:


# f1
best_params = {'penalty': 'l1', 'loss': 'log', 'alpha': .01}
# recall
# best_params = {'penalty': 'l1', 'loss': 'log', 'alpha': 1e-5}
# precision - same as f1
# best_params = {}


# In[13]:


clf = linear_model.SGDClassifier(**best_params)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[14]:


test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
test.scale_by_irrad('Clearsky GHI pvlib')


# In[15]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)")


# In[16]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.show()


# In[17]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[18]:


print(metrics.f1_score(test.df['sky_status'].values, pred))


# # Train on all NSRDB data, test various freq of ground data

# In[19]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.scale_by_irrad('Clearsky GHI pvlib')
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[20]:


bar = go.Bar(x=feature_cols, y=clf.coef_)
iplot([bar])


# ## 30 min freq ground data

# In[21]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[22]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[23]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.time_from_solar_noon_ratio2('Clearsky GHI pvlib')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[24]:


test.df = test.df[test.df.index.minute % 30 == 0]


# In[25]:


# test.df.loc[np.round(test.df['GHI'], 6) == 14.48218, 'GHI'] = 40


# In[26]:


# test.df = test.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[27]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[28]:


train2 = cs_detection.ClearskyDetection(nsrdb.df)
train2.intersection(test.df.index)


# In[29]:


nsrdb_clear = train2.df['sky_status'].values
ml_clear = pred
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib line length ratio'] * 100)
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib gradient ratio'] * 100)
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib gradient second ratio'] * 100)
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib gradient ratio std'] * 100)
# vis.add_line_ser(test.df['irrad_scaler'] * 100)
vis.show()


# In[30]:


probas = clf.predict_proba(test.df[feature_cols].values)


# In[31]:


test.df['probas'] = 0


# In[32]:


test.df['probas'] = probas


# In[33]:


trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 12, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text='prob: ' + test.df['probas'].astype(str))
iplot([trace0, trace1, trace2])


# In[34]:


visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 15 min freq ground data

# In[35]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[36]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[37]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[38]:


test.df = test.df[test.df.index.minute % 15 == 0]
# test.df = test.df.resample('15T').apply(lambda x: x[len(x) // 2])


# In[39]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 5, multiproc=True, by_day=True).astype(bool)


# In[40]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='15min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[41]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[42]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 12, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])


# In[43]:


visualize.plot_ts_slider_highligther(test.df, prob='probas')


# In[ ]:





# ## 10 min freq ground data

# In[44]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[45]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[46]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[47]:


test.df = test.df[test.df.index.minute % 10 == 0]


# In[48]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 7, multiproc=True, by_day=True).astype(bool)


# In[49]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='10min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[50]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[51]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 12, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])


# In[52]:


visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 5 min freq ground data

# In[53]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[54]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[55]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[56]:


test.df = test.df[test.df.index.minute % 5 == 0]


# In[57]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 13, multiproc=True, by_day=True).astype(bool)


# In[58]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='5min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[59]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[60]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 10, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])


# In[61]:


visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 1 min freq ground data

# In[62]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[63]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[64]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[65]:


test.df = test.df[test.df.index.minute % 1 == 0]


# In[66]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, multiproc=True, by_day=True).astype(bool)


# In[67]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='1min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[68]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 10, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])


# In[ ]:


visualize.plot_ts_slider_highligther(test.df, prob='probas')


# # Save model

# In[ ]:


import pickle


# In[ ]:


with open('abq_trained.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:


get_ipython().system('ls abq*')


# # Conclusion

# In general, the clear sky identification looks good.  At lower frequencies (30 min, 15 min) we see good agreement with NSRDB labeled points.  I suspect this could be further improved my doing a larger hyperparameter search, or even doing some feature extraction/reduction/additions.  

# In[ ]:




