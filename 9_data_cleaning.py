
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Investigate-input-data" data-toc-modified-id="Investigate-input-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Investigate input data</a></div><div class="lev2 toc-item"><a href="#ABQ" data-toc-modified-id="ABQ-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>ABQ</a></div><div class="lev2 toc-item"><a href="#SRRL" data-toc-modified-id="SRRL-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>SRRL</a></div><div class="lev2 toc-item"><a href="#ORNL" data-toc-modified-id="ORNL-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>ORNL</a></div>

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree
from sklearn import ensemble

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
import cufflinks as cf
cf.go_offline()
init_notebook_mode(connected=True)

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# In[2]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[3]:


len(nsrdb.df)


# # Investigate input data

# ## ABQ

# In[43]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[44]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates('01-01-2013', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[45]:


clf = ensemble.RandomForestClassifier(random_state=42)


# In[46]:


feature_cols = [
    'tfn',
    'abs_ideal_ratio_diff grad',
    'abs_ideal_ratio_diff grad mean', 
    'abs_ideal_ratio_diff grad std',
    'abs_ideal_ratio_diff grad second',
    'abs_ideal_ratio_diff grad second mean',
    'abs_ideal_ratio_diff grad second std',
    'GHI Clearsky GHI pvlib line length ratio',
    'GHI Clearsky GHI pvlib ratio', 
    'GHI Clearsky GHI pvlib ratio mean',
    'GHI Clearsky GHI pvlib ratio std',
    'GHI Clearsky GHI pvlib diff',
    'GHI Clearsky GHI pvlib diff mean', 
    'GHI Clearsky GHI pvlib diff std'
]

target_cols = ['sky_status']


# In[47]:


clf = train.fit_model(clf, 'GHI', 'Clearsky GHI pvlib', feature_cols, 3, target_cols=target_cols)


# In[48]:


train.df['sky_status2'] = ((np.abs(1 - train.df['GHI Clearsky GHI pvlib ratio']) <= .1) & (train.df['Clearsky GHI pvlib'] > 0)).astype(bool)


# In[49]:



t1 = go.Scatter(x=train.df['GHI'].index, y=train.df['GHI'], name='GHI')
t2 = go.Scatter(x=train.df['Clearsky GHI pvlib'].index, y=train.df['Clearsky GHI pvlib'], name='GHIcs')
nsrdb_mask = train.df['sky_status'].astype(bool) & ~train.df['sky_status2'].astype(bool)
ratio_mask = ~train.df['sky_status'].astype(bool) & train.df['sky_status2'].astype(bool)
both_mask = train.df['sky_status'].astype(bool) & train.df['sky_status2'].astype(bool)
t3 = go.Scatter(x=train.df[nsrdb_mask].index, y=train.df[nsrdb_mask]['GHI'], name='NSRDB', mode='markers')
t4 = go.Scatter(x=train.df[ratio_mask].index, y=train.df[ratio_mask]['GHI'], name='Ratio', mode='markers')
t5 = go.Scatter(x=train.df[both_mask].index, y=train.df[both_mask]['GHI'], name='Both', mode='markers')

iplot([t1, t2, t3, t4, t5])



# In[50]:


print(np.sum(nsrdb_mask))


# In[51]:


from sklearn import neighbors


# In[52]:


nn = neighbors.KNeighborsClassifier(n_neighbors=8)


# In[53]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[54]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[55]:


from sklearn import preprocessing


# In[56]:


ss = preprocessing.StandardScaler()
X_std = ss.fit_transform(train.df[feature_cols])


# In[57]:


nn.fit(X_std, train.df[target_cols].values.ravel())


# In[58]:


pred = nn.predict(X_std)


# In[59]:


np.array_equal(pred, train.df[target_cols].values.ravel())


# In[60]:


t1 = go.Scatter(x=train.df['GHI'].index, y=train.df['GHI'], name='GHI')
t2 = go.Scatter(x=train.df['Clearsky GHI pvlib'].index, y=train.df['Clearsky GHI pvlib'], name='GHIcs')
nsrdb_mask = train.df['sky_status'].astype(bool) & ~pred
ratio_mask = ~train.df['sky_status'].astype(bool) & pred
both_mask = train.df['sky_status'].astype(bool) & pred
t3 = go.Scatter(x=train.df[nsrdb_mask].index, y=train.df[nsrdb_mask]['GHI'], name='NSRDB', mode='markers')
t4 = go.Scatter(x=train.df[ratio_mask].index, y=train.df[ratio_mask]['GHI'], name='NN', mode='markers')
t5 = go.Scatter(x=train.df[both_mask].index, y=train.df[both_mask]['GHI'], name='Both', mode='markers')

iplot([t1, t2, t3, t4, t5])


# In[61]:


test.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
utils.calc_all_window_metrics(test.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
X_test_std = ss.transform(test.df[feature_cols].values)


# In[62]:


pred = nn.predict(X_test_std)


# In[63]:


t1 = go.Scatter(x=test.df['GHI'].index, y=test.df['GHI'], name='GHI')
t2 = go.Scatter(x=test.df['Clearsky GHI pvlib'].index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
nsrdb_mask = test.df['sky_status'].astype(bool) & ~pred
ratio_mask = ~test.df['sky_status'].astype(bool) & pred
both_mask = test.df['sky_status'].astype(bool) & pred
t3 = go.Scatter(x=test.df[nsrdb_mask].index, y=test.df[nsrdb_mask]['GHI'], name='NSRDB', mode='markers')
t4 = go.Scatter(x=test.df[ratio_mask].index, y=test.df[ratio_mask]['GHI'], name='NN', mode='markers')
t5 = go.Scatter(x=test.df[both_mask].index, y=test.df[both_mask]['GHI'], name='Both', mode='markers')

iplot([t1, t2, t3, t4, t5])


# In[64]:


cm = metrics.confusion_matrix(test.df['sky_status'], pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'MST')
# ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df, 'MST')


# In[ ]:


test.df['tfn'].plot()


# In[71]:


test.trim_dates('10-01-2015', '11-01-2015')
test.df = test.df[test.df.index.minute % 30 == 0]
test.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')
test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
utils.calc_all_window_metrics(test.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
X_test_std = ss.transform(test.df[feature_cols])


# In[ ]:


pred = nn.predict(X_test_std)


# In[ ]:


t1 = go.Scatter(x=test.df['GHI'].index, y=test.df['GHI'], name='GHI')
t2 = go.Scatter(x=test.df['Clearsky GHI pvlib'].index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
t3 = go.Scatter(x=test.df[pred].index, y=test.df[pred]['GHI'], name='NSRDB', mode='markers')

iplot([t1, t2, t3])


# ## SRRL

# In[50]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[51]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates('01-01-2013', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[52]:


clf = ensemble.RandomForestClassifier(random_state=42)


# In[53]:


feature_cols = [
    'tfn',
    'abs_ideal_ratio_diff grad',
    'abs_ideal_ratio_diff grad mean', 
    'abs_ideal_ratio_diff grad std',
    'abs_ideal_ratio_diff grad second',
    'abs_ideal_ratio_diff grad second mean',
    'abs_ideal_ratio_diff grad second std',
    'GHI Clearsky GHI pvlib line length ratio',
    'GHI Clearsky GHI pvlib ratio', 
    'GHI Clearsky GHI pvlib ratio mean',
    'GHI Clearsky GHI pvlib ratio std',
    'GHI Clearsky GHI pvlib diff',
    'GHI Clearsky GHI pvlib diff mean', 
    'GHI Clearsky GHI pvlib diff std'
]

target_cols = ['sky_status']


# In[54]:


clf = train.fit_model(clf, 'GHI', 'Clearsky GHI pvlib', feature_cols, 3, target_cols=target_cols)


# In[55]:


train.df['sky_status2'] = ((np.abs(1 - train.df['GHI Clearsky GHI pvlib ratio']) <= .1) & (train.df['Clearsky GHI pvlib'] > 0)).astype(bool)


# In[56]:



t1 = go.Scatter(x=train.df['GHI'].index, y=train.df['GHI'], name='GHI')
t2 = go.Scatter(x=train.df['Clearsky GHI pvlib'].index, y=train.df['Clearsky GHI pvlib'], name='GHIcs')
nsrdb_mask = train.df['sky_status'].astype(bool) & ~train.df['sky_status2'].astype(bool)
ratio_mask = ~train.df['sky_status'].astype(bool) & train.df['sky_status2'].astype(bool)
both_mask = train.df['sky_status'].astype(bool) & train.df['sky_status2'].astype(bool)
t3 = go.Scatter(x=train.df[nsrdb_mask].index, y=train.df[nsrdb_mask]['GHI'], name='NSRDB', mode='markers')
t4 = go.Scatter(x=train.df[ratio_mask].index, y=train.df[ratio_mask]['GHI'], name='Ratio', mode='markers')
t5 = go.Scatter(x=train.df[both_mask].index, y=train.df[both_mask]['GHI'], name='Both', mode='markers')

iplot([t1, t2, t3, t4, t5])



# In[57]:


print(np.sum(nsrdb_mask))


# ## ORNL

# In[58]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('EST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[59]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates('01-01-2013', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[60]:


clf = ensemble.RandomForestClassifier(random_state=42)


# In[61]:


feature_cols = [
    'tfn',
    'abs_ideal_ratio_diff grad',
    'abs_ideal_ratio_diff grad mean', 
    'abs_ideal_ratio_diff grad std',
    'abs_ideal_ratio_diff grad second',
    'abs_ideal_ratio_diff grad second mean',
    'abs_ideal_ratio_diff grad second std',
    'GHI Clearsky GHI pvlib line length ratio',
    'GHI Clearsky GHI pvlib ratio', 
    'GHI Clearsky GHI pvlib ratio mean',
    'GHI Clearsky GHI pvlib ratio std',
    'GHI Clearsky GHI pvlib diff',
    'GHI Clearsky GHI pvlib diff mean', 
    'GHI Clearsky GHI pvlib diff std'
]

target_cols = ['sky_status']


# In[62]:


clf = train.fit_model(clf, 'GHI', 'Clearsky GHI pvlib', feature_cols, 3, target_cols=target_cols)


# In[63]:


train.df['sky_status2'] = ((np.abs(1 - train.df['GHI Clearsky GHI pvlib ratio']) <= .1) & (train.df['Clearsky GHI pvlib'] > 0)).astype(bool)


# In[64]:



t1 = go.Scatter(x=train.df['GHI'].index, y=train.df['GHI'], name='GHI')
t2 = go.Scatter(x=train.df['Clearsky GHI pvlib'].index, y=train.df['Clearsky GHI pvlib'], name='GHIcs')
nsrdb_mask = train.df['sky_status'].astype(bool) & ~train.df['sky_status2'].astype(bool)
ratio_mask = ~train.df['sky_status'].astype(bool) & train.df['sky_status2'].astype(bool)
both_mask = train.df['sky_status'].astype(bool) & train.df['sky_status2'].astype(bool)
t3 = go.Scatter(x=train.df[nsrdb_mask].index, y=train.df[nsrdb_mask]['GHI'], name='NSRDB', mode='markers')
t4 = go.Scatter(x=train.df[ratio_mask].index, y=train.df[ratio_mask]['GHI'], name='Ratio', mode='markers')
t5 = go.Scatter(x=train.df[both_mask].index, y=train.df[both_mask]['GHI'], name='Both', mode='markers')

iplot([t1, t2, t3, t4, t5])



# In[65]:


print(np.sum(nsrdb_mask))


# In[ ]:




