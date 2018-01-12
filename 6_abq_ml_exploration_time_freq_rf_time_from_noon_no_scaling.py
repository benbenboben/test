
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
# 'GHI Clearsky GHI pvlib line length ratio gradient',
# 'GHI Clearsky GHI pvlib line length ratio gradient second'
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


train.df['irrad_scaler'] = 1


# In[8]:


import warnings


# In[9]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    params={}
    params['max_depth'] = [4, 5, 6]
    params['n_estimators'] = [32, 64, 128]
    params['class_weight'] = [None] # , 'balanced']
    # best_score = -1
    for depth, nest, cw in itertools.product(params['max_depth'], params['n_estimators'], params['class_weight']):
        train2 = cs_detection.ClearskyDetection(train.df)
        train2.trim_dates('01-01-1999', '01-01-2014')
        utils.calc_all_window_metrics(train2.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
        test2 = cs_detection.ClearskyDetection(train.df)
        test2.trim_dates('01-01-2014', '01-01-2015')
        clf = ensemble.RandomForestClassifier(max_depth=depth, n_estimators=nest, class_weight=cw, n_jobs=-1)
        clf.fit(train2.df[feature_cols].values, train2.df[target_cols].values.flatten())
        pred = test2.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)
        f1_score = metrics.f1_score(test2.df['sky_status'], pred)
        recall_score = metrics.recall_score(test2.df['sky_status'], pred)
        precision_score = metrics.precision_score(test2.df['sky_status'], pred)
        print(f1_score, recall_score, precision_score, depth, nest, cw)

## output from above cell
## search max_depth 4-7, n_estimators [32, 64, 128], class_weight [None, 'balanced']

0.905814071887 0.925317245683 0.88711607499 4 32 None
0.897951332115 0.971083836072 0.835062611807 4 32 balanced
0.906055979644 0.925941335552 0.887006775608 4 64 None
0.902622087923 0.963178697732 0.849229640499 4 64 balanced
0.906875190413 0.928853754941 0.885912698413 4 128 None
0.900357591572 0.969003536509 0.840794223827 4 128 balanced
0.905852417303 0.925733305596 0.886807493025 4 256 None
0.899348439171 0.961930517995 0.844411979547 4 256 balanced
0.906856272219 0.928645724984 0.886065899166 5 32 None
0.911399767352 0.977948824631 0.853330913051 5 32 balanced
0.906504065041 0.927813605159 0.886151400755 5 64 None
0.911108947522 0.973372165592 0.856332357247 5 64 balanced
0.90590987692 0.926357395465 0.886345541401 5 128 None
0.913212687293 0.97628458498 0.85779564979 5 128 balanced
0.906040268456 0.926773455378 0.886214442013 5 256 None
0.911833398209 0.97462034533 0.8566465533 5 256 balanced
0.907537381752 0.928021635115 0.887937898089 6 32 None
0.912993039443 0.982317453713 0.852808379989 6 32 balanced
0.907335907336 0.928853754941 0.88679245283 6 64 None
0.912191794172 0.973580195548 0.858085808581 6 64 balanced
0.907298231348 0.928437695028 0.887099980123 6 128 None
0.912887248713 0.977740794674 0.856102003643 6 128 balanced
0.907132696606 0.928645724984 0.886593843098 6 256 None
0.912988894254 0.983357603495 0.852018745494 6 256 balanced
0.90777539998 0.926565425421 0.889732321215 7 32 None
0.913311500674 0.986270022883 0.850403587444 7 32 balanced
0.908185053381 0.929061784897 0.888225934765 7 64 None
0.913635047766 0.984813813189 0.852051835853 7 64 balanced
0.907518567504 0.927813605159 0.888092393469 7 128 None
0.914109245319 0.985229873102 0.852565256526 7 128 balanced
0.907759585071 0.928437695028 0.887982491047 7 256 None
0.913734939759 0.986061992927 0.851293103448 7 256 balanced
# In[ ]:





# In[ ]:





# In[21]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.time_from_solar_noon_ratio2('Clearsky GHI pvlib')
train.scale_by_irrad('Clearsky GHI pvlib')


# In[22]:


train.df['irrad_scaler'] = 1


# In[23]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[24]:


# f1
best_params = {'max_depth': 5, 'n_estimators': 128, 'class_weight': 'balanced'}
# recall
# best_params = {'max_depth': 6, 'n_estimators': 128, 'class_weight': 'balanced'}
# precision
# best_params = {'max_depth': 7, 'n_estimators': 32, 'class_weight': 'None}


# In[25]:


clf = ensemble.RandomForestClassifier(**best_params, n_jobs=-1)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[ ]:





# In[26]:


test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
test.scale_by_irrad('Clearsky GHI pvlib')
test.df['irrad_scaler'] = 1


# In[27]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)")


# In[28]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.show()


# In[29]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[30]:


print(metrics.f1_score(test.df['sky_status'].values, pred))


# # Train on all NSRDB data, test various freq of ground data

# In[32]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
train.scale_by_irrad('Clearsky GHI pvlib')
train.df['irrad_scaler'] = 1
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[33]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## 30 min freq ground data

# In[34]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[35]:


test.trim_dates('10-01-2015', '10-21-2015')


# In[36]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.time_from_solar_noon_ratio2('Clearsky GHI pvlib')
test.scale_by_irrad('Clearsky GHI pvlib')
test.df['irrad_scaler'] = 1


# In[37]:


test.df = test.df[test.df.index.minute % 30 == 0]


# In[38]:


# test.df.loc[np.round(test.df['GHI'], 6) == 14.48218, 'GHI'] = 40


# In[39]:


# test.df = test.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[40]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[41]:


train2 = cs_detection.ClearskyDetection(nsrdb.df)
train2.intersection(test.df.index)


# In[42]:


nsrdb_clear = train2.df['sky_status'].values
ml_clear = pred
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()

nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'Method 1')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'Method 2')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Method 1+2')
vis.show()
# In[43]:


nsrdb_clear = train2.df['sky_status'].values
ml_clear = pred

test.df['nsrdb_sky'] = nsrdb_clear
test.df['ml_sky'] = pred


# In[44]:


from plotly import tools as tls

def make_subplots(test, train, nrow, ncol, random_seed, width=800, height=1000):
    # nsrdb_clear = train2.df['sky_status'].values
    # ml_clear = pred

#     test.df['nsrdb_sky'] = nsrdb_clear
#     test.df['ml_sky'] = pred

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

    ghi_line = {'color': colors['blue'], 'width': 1}
    ghics_line = {'color': colors['orange'], 'width': 1}
    ml_only = {'color': colors['green'], 'size': 6}
    nsrdb_only = {'color': colors['red'], 'size': 6}
    both = {'color': colors['purple'], 'size': 6}

    # nrow, ncol = 2, 2

    fig = tls.make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True, print_grid=True)
    
    fig['layout'].update(width=width, height=height)
    
    days = np.unique(test.df.index.date)[:nrow * ncol]
    np.random.seed(random_seed)
    days = np.random.permutation(days)

    for i, day in enumerate(days):
        if i == nrow * ncol: break
        legend = False
        # if i == 0: legend = True
        g = test.df[test.df.index.date == day]
        g = g.between_time('05:00:00', '19:00:00')
        g.index = range(len(g))
        trace0 = go.Scatter(x=g.index, y=g['GHI'], line=ghi_line, showlegend=legend, name='GHI')
        trace1 = go.Scatter(x=g.index, y=g['Clearsky GHI pvlib'], line=ghics_line, showlegend=legend, name='GHIcs')
        trace2 = go.Scatter(x=g[g['ml_sky'] & ~g['nsrdb_sky']].index, y=g[g['ml_sky'] & ~g['nsrdb_sky']]['GHI'], mode='markers', marker=ml_only, showlegend=legend, name='Method 1')
        trace3 = go.Scatter(x=g[~g['ml_sky'] & g['nsrdb_sky']].index, y=g[~g['ml_sky'] & g['nsrdb_sky']]['GHI'], mode='markers', marker=nsrdb_only, showlegend=legend, name='Method 2')
        trace4 = go.Scatter(x=g[g['ml_sky'] & g['nsrdb_sky']].index, y=g[g['ml_sky'] & g['nsrdb_sky']]['GHI'], mode='markers', marker=both, showlegend=legend, name='Method 1 & 2')
        col = (i % ncol) + 1
        row = (i // ncol) + 1
        print(i, row, col, day)
        traces = [trace0, trace1, trace2, trace3, trace4]
        for t in traces:
            fig.append_trace(t, row, col)

    iplot(fig)


# In[45]:


make_subplots(test, train2, 8, 2, 0, width=1000, height=1200)


# In[46]:


probas = clf.predict_proba(test.df[feature_cols].values)


# In[47]:


test.df['probas'] = 0


# In[48]:


test.df['probas'] = probas[:, 1]

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 12, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text='prob: ' + test.df['probas'].astype(str))
iplot([trace0, trace1, trace2])
# In[49]:


visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 15 min freq ground data

# In[54]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[55]:


test.trim_dates('10-01-2015', '10-17-2015')


# In[56]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')
test.df['irrad_scaler'] = 1


# In[57]:


test.df = test.df[test.df.index.minute % 15 == 0]
# test.df = test.df.resample('15T').apply(lambda x: x[len(x) // 2])


# In[58]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 5, multiproc=True, by_day=True).astype(bool)


# In[59]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='15min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[60]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[61]:


nsrdb_clear = train2.df['sky_status'].astype(bool)
ml_clear = pred

test.df['nsrdb_sky'] = nsrdb_clear
test.df['nsrdb_sky'] = test.df['nsrdb_sky'].replace(np.nan, False)
test.df['ml_sky'] = pred

make_subplots(test, train2, 8, 2, 0, width=1000, height=1200)


# In[62]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 10 min freq ground data

# In[63]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[64]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[65]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')
test.df['irrad_scaler'] = 1


# In[66]:


test.df = test.df[test.df.index.minute % 10 == 0]


# In[67]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 7, multiproc=True, by_day=True).astype(bool)


# In[68]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='10min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[69]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[70]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]

visualize.plot_ts_slider_highligther(test.df, prob='probas')


# In[71]:


probas


# In[72]:


nsrdb_clear = train2.df['sky_status'].astype(bool)
ml_clear = pred

test.df['nsrdb_sky'] = nsrdb_clear
test.df['nsrdb_sky'] = test.df['nsrdb_sky'].replace(np.nan, False)
test.df['ml_sky'] = pred

make_subplots(test, train2, 8, 2, 0, width=1000, height=1200)


# ## 5 min freq ground data

# In[78]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[79]:


test.trim_dates('10-01-2015', '10-17-2015')


# In[80]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')
test.df['irrad_scaler'] = 1


# In[81]:


test.df = test.df[test.df.index.minute % 5 == 0]


# In[82]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 13, multiproc=True, by_day=True).astype(bool)


# In[83]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='5min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[84]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()

nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'Method 1')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'Method 2')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Method 1+2')
vis.show()from plotly import tools as tls

nsrdb_clear = train2.df['sky_status']
ml_clear = pred

print(len(nsrdb_clear), len(test.df))

test.df['nsrdb_sky'] = nsrdb_clear
test.df['ml_sky'] = pred

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
ml_only = {'color': colors['green']}
nsrdb_only = {'color': colors['red']}
both = {'color': colors['purple']}

nrow, ncol = 3, 3

fig = tls.make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True, print_grid=True)

for i, (name, g) in enumerate(test.df.groupby(test.df.index.date)):
    if i == nrow * ncol: break
    legend = False
    if i == 0: legend = True
    g = g.between_time('05:00:00', '19:00:00')
    g.index = range(len(g))
    trace0 = go.Scatter(x=g.index, y=g['GHI'], line=ghi_line, showlegend=legend, name='GHI')
    trace1 = go.Scatter(x=g.index, y=g['Clearsky GHI pvlib'], line=ghics_line, showlegend=legend, name='GHIcs')
    trace2 = go.Scatter(x=g[g['ml_sky'] & ~g['nsrdb_sky']].index, y=g[g['ml_sky'] & ~g['nsrdb_sky']]['GHI'], mode='markers', marker=ml_only, showlegend=legend, name='Method 1')
    trace3 = go.Scatter(x=g[~g['ml_sky'] & g['nsrdb_sky']].index, y=g[~g['ml_sky'] & g['nsrdb_sky']]['GHI'], mode='markers', marker=nsrdb_only, showlegend=legend, name='Method 2')
    trace4 = go.Scatter(x=g[g['ml_sky'] & g['nsrdb_sky']].index, y=g[g['ml_sky'] & g['nsrdb_sky']]['GHI'], mode='markers', marker=both, showlegend=legend, name='Method 1 & 2')
    row = i % nrow + 1
    col = i // ncol + 1
    traces = [trace0, trace1, trace2, trace3, trace4]
    for t in traces:
        fig.append_trace(t, row, col)

iplot(fig, layout)probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 10, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])
# In[85]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[: ,1]

visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 1 min freq ground data

# In[93]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[94]:


test.trim_dates('10-01-2015', '10-17-2015')


# In[95]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')
test.df['irrad_scaler'] = 1


# In[96]:


test.df = test.df[test.df.index.minute % 1 == 0]


# In[97]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, multiproc=True, by_day=True).astype(bool)


# In[98]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='1min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[99]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()

nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'Method 1')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'Method 2')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Method 1+2')
vis.show()from plotly import tools as tls

nsrdb_clear = train2.df['sky_status']
ml_clear = pred

print(len(nsrdb_clear), len(test.df))

test.df['nsrdb_sky'] = nsrdb_clear
test.df['ml_sky'] = pred

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
ml_only = {'color': colors['green']}
nsrdb_only = {'color': colors['red']}
both = {'color': colors['purple']}

nrow, ncol = 3, 3

fig = tls.make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True, print_grid=True)

for i, (name, g) in enumerate(test.df.groupby(test.df.index.date)):
    if i == nrow * ncol: break
    legend = False
    if i == 0: legend = True
    g = g.between_time('05:00:00', '19:00:00')
    g.index = range(len(g))
    trace0 = go.Scatter(x=g.index, y=g['GHI'], line=ghi_line, showlegend=legend, name='GHI')
    trace1 = go.Scatter(x=g.index, y=g['Clearsky GHI pvlib'], line=ghics_line, showlegend=legend, name='GHIcs')
    trace2 = go.Scatter(x=g[g['ml_sky'] & ~g['nsrdb_sky']].index, y=g[g['ml_sky'] & ~g['nsrdb_sky']]['GHI'], mode='markers', marker=ml_only, showlegend=legend, name='Method 1')
    trace3 = go.Scatter(x=g[~g['ml_sky'] & g['nsrdb_sky']].index, y=g[~g['ml_sky'] & g['nsrdb_sky']]['GHI'], mode='markers', marker=nsrdb_only, showlegend=legend, name='Method 2')
    trace4 = go.Scatter(x=g[g['ml_sky'] & g['nsrdb_sky']].index, y=g[g['ml_sky'] & g['nsrdb_sky']]['GHI'], mode='markers', marker=both, showlegend=legend, name='Method 1 & 2')
    row = i % nrow + 1
    col = i // ncol + 1
    traces = [trace0, trace1, trace2, trace3, trace4]
    for t in traces:
        fig.append_trace(t, row, col)

iplot(fig, layout)probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas

trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 10, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text=test.df['probas'])
iplot([trace0, trace1, trace2])
# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas
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




