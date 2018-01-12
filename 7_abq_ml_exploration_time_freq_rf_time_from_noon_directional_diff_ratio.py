
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev2 toc-item"><a href="#Default-classifier" data-toc-modified-id="Default-classifier-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Default classifier</a></div><div class="lev2 toc-item"><a href="#Gridsearch" data-toc-modified-id="Gridsearch-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Gridsearch</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#30-min-freq-ground-data" data-toc-modified-id="30-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>30 min freq ground data</a></div><div class="lev2 toc-item"><a href="#15-min-freq-ground-data" data-toc-modified-id="15-min-freq-ground-data-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>15 min freq ground data</a></div><div class="lev2 toc-item"><a href="#10-min-freq-ground-data" data-toc-modified-id="10-min-freq-ground-data-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>10 min freq ground data</a></div><div class="lev2 toc-item"><a href="#5-min-freq-ground-data" data-toc-modified-id="5-min-freq-ground-data-34"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>5 min freq ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-35"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

# In[10]:


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

# In[161]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[162]:


len(nsrdb.df)


# # Train/test on NSRDB data to find optimal parameters

# ## Default classifier

# In[12]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[13]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[14]:


clf = ensemble.RandomForestClassifier(random_state=42)


# In[15]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)

%load_ext line_profiler%lprun -f utils.calc_all_window_metrics utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
# In[16]:


train.df.keys()


# In[17]:


feature_cols = [
    'tfn',
#     'ghi_status',
#     'abs_ideal_ratio_diff',
#     'abs_ideal_ratio_diff mean',
#     'abs_ideal_ratio_diff std',
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


# In[18]:


for k in feature_cols:
    print(k, train.df[k].isnull().values.any())


# In[19]:


vis = visualize.Visualizer()
vis.plot_corr_matrix(train.df[feature_cols].corr(), feature_cols)


# In[163]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[164]:


get_ipython().run_cell_magic('time', '', "utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)\nclf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())\n# clf.fit(train.df[train.df['GHI'] > 0][feature_cols].values, train.df[train.df['GHI'] > 0][target_cols].values.flatten())")


# In[167]:


clf2 = tree.DecisionTreeClassifier(max_depth=3)
clf2.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[168]:


tree.export_graphviz(clf2, 'tree.dot', filled=True, feature_names=feature_cols, class_names=['cloudy', 'clear'])


# In[22]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=False, by_day=False).astype(bool)")


# In[23]:


metrics.accuracy_score(test.df['sky_status'], pred)


# In[24]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.show()


# In[25]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[26]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## Gridsearch
import warningswith warnings.catch_warnings():
    warnings.simplefilter('ignore')
    params={}
    params['max_depth'] = [4, 8, 12, 16]
    params['n_estimators'] = [64]
    params['class_weight'] = [None, 'balanced']
    params['min_samples_leaf'] = [1, 2, 3]
    results = []
    for depth, nest, cw, min_samples in itertools.product(params['max_depth'], params['n_estimators'], params['class_weight'], params['min_samples_leaf']):
        print('Params:')
        print('depth: {}, n_estimators: {}, class_weight: {}, min_samples_leaf: {}'.format(depth, nest, cw, min_samples))
        train2 = cs_detection.ClearskyDetection(train.df)
        train2.trim_dates('01-01-1999', '01-01-2014')
        utils.calc_all_window_metrics(train2.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
        test2 = cs_detection.ClearskyDetection(train.df)
        test2.trim_dates('01-01-2014', '01-01-2015')
        clf = ensemble.RandomForestClassifier(max_depth=depth, n_estimators=nest, class_weight=cw, min_samples_leaf=min_samples, n_jobs=-1)
        clf.fit(train2.df[train2.df['GHI'] > 0][feature_cols].values, train2.df[train2.df['GHI'] > 0][target_cols].values.flatten())
        
        print('\t Scores:')
        test_pred = test2.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True)
        accuracy_score = metrics.accuracy_score(test2.df['sky_status'], test_pred)
        print('\t\t accuracy: {}'.format(accuracy_score))
        f1_score = metrics.f1_score(test2.df['sky_status'], test_pred)
        print('\t\t f1:{}'.format(f1_score))
        recall_score = metrics.recall_score(test2.df['sky_status'], test_pred)
        print('\t\t recall:{}'.format(recall_score))
        precision_score = metrics.precision_score(test2.df['sky_status'], test_pred)
        print('\t\t precision:{}'.format(precision_score))
        results.append({'max_depth': depth, 'n_estimators': nest, 'class_weight': cw, 'min_samples_leaf': min_samples,
                        'accuracy': accuracy_score, 'f1': f1_score, 'recall': recall_score, 'precision': precision_score})
runs_df = pd.DataFrame(results)runs_df.sort_values('accuracy', ascending=False)runs_df.sort_values('f1', ascending=False)runs_df.sort_values('precision', ascending=False)runs_df.sort_values('recall', ascending=False)runs_df.to_csv('7_abq_ml_exploration_time_freq_rf_time_from_noon_directional_diff_ratio.csv')
# In[95]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[96]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[97]:


# f1
best_params = {'max_depth': 8, 'n_estimators': 128, 'class_weight': 'balanced', 'min_samples_leaf': 2}
# recall
# best_params = {'max_depth': 6, 'n_estimators': 128, 'class_weight': 'balanced'}
# precision
# best_params = {'max_depth': 7, 'n_estimators': 32, 'class_weight': 'None}


# In[98]:


clf = ensemble.RandomForestClassifier(**best_params, n_jobs=-1)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[99]:


test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[100]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=False, by_day=False).astype(bool)")


# In[101]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.show()


# In[102]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[103]:


print(metrics.f1_score(test.df['sky_status'].values, pred))


# In[104]:


print(metrics.accuracy_score(test.df['sky_status'], pred))

test2 = cs_detection.ClearskyDetection(test.df)
test2.trim_dates('10-01-2015', None)
probas = clf.predict_proba(test2.df[feature_cols].values)
test2.df['probas'] = 0
test2.df['probas'] = probas[:, 1]

visualize.plot_ts_slider_highligther(test2.df, prob='probas')
# # Train on all NSRDB data, test various freq of ground data

# In[105]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[106]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## 30 min freq ground data

# In[107]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[108]:


test.trim_dates('10-01-2015', '11-01-2015')


# In[109]:


test.df = test.df[test.df.index.minute % 30 == 0]


# In[110]:


test.df.keys()


# In[111]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[112]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=False, by_day=False).astype(bool)


# In[113]:


train2 = cs_detection.ClearskyDetection(nsrdb.df)
train2.intersection(test.df.index)


# In[114]:


nsrdb_clear = train2.df['sky_status'].values
ml_clear = pred
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()

make_subplots(test, train2, 8, 2, 0, width=1000, height=1200)probas = clf.predict_proba(test.df[feature_cols].values)test.df['probas'] = 0test.df['probas'] = probas[:, 1]
trace0 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='GHI')
trace1 = go.Scatter(x=test.df.index, y=test.df['Clearsky GHI pvlib'], name='GHIcs')
trace2 = go.Scatter(x=test.df.index, y=test.df['GHI'], name='prob', mode='markers', marker={'size': 12, 'color': test.df['probas'], 'colorscale': 'Viridis', 'showscale': True}, text='prob: ' + test.df['probas'].astype(str))
iplot([trace0, trace1, trace2])visualize.plot_ts_slider_highligther(test.df, prob='probas')
# ## 15 min freq ground data

# In[115]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[116]:


test.trim_dates('10-01-2015', '10-17-2015')


# In[117]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[118]:


test.df = test.df[test.df.index.minute % 15 == 0]
# test.df = test.df.resample('15T').apply(lambda x: x[len(x) // 2])


# In[119]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 5, multiproc=True, by_day=True).astype(bool)


# In[120]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='15min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[121]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()

nsrdb_clear = train2.df['sky_status'].astype(bool)
ml_clear = pred

test.df['nsrdb_sky'] = nsrdb_clear
test.df['nsrdb_sky'] = test.df['nsrdb_sky'].replace(np.nan, False)
test.df['ml_sky'] = pred

make_subplots(test, train2, 8, 2, 0, width=1000, height=1200)
# In[122]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 10 min freq ground data

# In[123]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[124]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[125]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[126]:


test.df = test.df[test.df.index.minute % 10 == 0]


# In[127]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 7, multiproc=False, by_day=False).astype(bool)


# In[128]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='10min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[129]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()


# In[130]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]

visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 5 min freq ground data

# In[153]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[154]:


test.trim_dates('10-01-2015', '10-04-2015')


# In[155]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[156]:


test.df = test.df[test.df.index.minute % 5 == 0]


# In[157]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 13, multiproc=True, by_day=True).astype(bool)


# In[158]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='5min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[159]:


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
# In[160]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]

visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 1 min freq ground data

# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[ ]:


test.trim_dates('10-01-2015', '10-17-2015')


# In[ ]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[ ]:


test.df = test.df[test.df.index.minute % 1 == 0]


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, multiproc=True, by_day=True).astype(bool)


# In[ ]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='1min'))
train2.df['sky_status'] = train2.df['sky_status'].fillna(False)


# In[ ]:


nsrdb_clear = train2.df['sky_status']
ml_clear = test.df['sky_status iter']
vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[ml_clear & ~nsrdb_clear]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[~ml_clear & nsrdb_clear]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[ml_clear & nsrdb_clear]['GHI'], 'Both clear')
vis.show()

probas = clf.predict_proba(test.df[feature_cols].values)
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




