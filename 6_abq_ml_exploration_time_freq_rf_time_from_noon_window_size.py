
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

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


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')

nsrdb.trim_dates('01-01-2014', '01-01-2016')
# In[3]:


nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb.time_from_solar_noon_ratio2('Clearsky GHI pvlib')


# In[4]:


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
'GHI Clearsky GHI pvlib line length ratio gradient',
'GHI Clearsky GHI pvlib line length ratio gradient second'
]

target_cols = ['sky_status']


# # Train/test on NSRDB data to find optimal parameters

# In[5]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[6]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.time_from_solar_noon_ratio2('Clearsky GHI pvlib')
train.scale_by_irrad('Clearsky GHI pvlib')


# In[7]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[8]:


best_params = {'max_depth': 4, 'n_estimators': 128, 'class_weight': 'balanced'}
best_params = {'max_depth': 4, 'n_estimators': 256, 'class_weight': None}
best_params = {'max_depth': 4, 'n_estimators': 256, 'class_weight': 'balanced'}


# In[9]:


clf = ensemble.RandomForestClassifier(**best_params, n_jobs=-1)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[10]:


test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
test.scale_by_irrad('Clearsky GHI pvlib')


# In[11]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)")


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


# # Train on all NSRDB data, test various freq of ground data

# In[15]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.scale_by_irrad('Clearsky GHI pvlib')
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[16]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## 1 min freq ground data

# In[17]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[18]:


test.trim_dates('10-01-2015', '10-15-2015')


# In[19]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[20]:


test.df = test.df[test.df.index.minute % 1 == 0]


# In[21]:


print(test.df.index)


# In[22]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, multiproc=True, by_day=True).astype(bool)


# In[23]:


for w in [3, 15, 31, 61]:
    print(w)
    pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, w, multiproc=True, by_day=True).astype(bool)
    test.df['pred' + str(w)] = pred


# In[24]:


from plotly import tools as tls


# In[25]:


fig = tls.make_subplots(rows=4, cols=1, shared_xaxes=True, print_grid=True)

fig['layout'].update(height=1000)

colors = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow': '#bcbd22',
    'teal': '#17becf'
}

ghi_line = {'color': colors['blue']}
ghics_line = {'color': colors['orange']}
ml_only = {'color': colors['green'], 'size': 10}
nsrdb_only = {'color': colors['red'], 'size': 10}
both = {'color': colors['purple'], 'size': 10}

for i, c in enumerate(['pred3', 'pred15', 'pred31', 'pred61']):
    legend = False
    if i == 0: legend = True
    trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], line=ghi_line, showlegend=legend, name='GHI')
    trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], line=ghics_line, showlegend=legend, name='GHIcs')
    trace2 = go.Scatter(x=test.df[test.df[c] & ~test.df['sky_status pvlib']].index, y=test.df[test.df[c] & ~test.df['sky_status pvlib']]['GHI'], mode='markers', marker=ml_only, name='RF', showlegend=legend)
    trace3 = go.Scatter(x=test.df[~test.df[c] & test.df['sky_status pvlib']].index, y=test.df[~test.df[c] & test.df['sky_status pvlib']]['GHI'], mode='markers', marker=nsrdb_only, name='PVLib', showlegend=legend)
    trace4 = go.Scatter(x=test.df[test.df[c] & test.df['sky_status pvlib']].index, y=test.df[test.df[c] & test.df['sky_status pvlib']]['GHI'], mode='markers', marker=both, name='RF+PVLib', showlegend=legend)
    fig.append_trace(trace0, i + 1, 1)
    fig.append_trace(trace1, i + 1, 1)
    fig.append_trace(trace2, i + 1, 1)
    fig.append_trace(trace3, i + 1, 1)
    fig.append_trace(trace4, i + 1, 1)

iplot(fig)


# In[ ]:




