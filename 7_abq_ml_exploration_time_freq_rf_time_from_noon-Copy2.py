
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev2 toc-item"><a href="#Default-classifier" data-toc-modified-id="Default-classifier-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Default classifier</a></div><div class="lev2 toc-item"><a href="#Gridsearch" data-toc-modified-id="Gridsearch-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Gridsearch</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#30-min-freq-ground-data" data-toc-modified-id="30-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>30 min freq ground data</a></div><div class="lev2 toc-item"><a href="#15-min-freq-ground-data" data-toc-modified-id="15-min-freq-ground-data-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>15 min freq ground data</a></div><div class="lev2 toc-item"><a href="#10-min-freq-ground-data" data-toc-modified-id="10-min-freq-ground-data-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>10 min freq ground data</a></div><div class="lev2 toc-item"><a href="#5-min-freq-ground-data" data-toc-modified-id="5-min-freq-ground-data-34"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>5 min freq ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-35"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

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
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# # Train/test on NSRDB data to find optimal parameters

# ## Default classifier

# In[3]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[4]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[5]:


# clf = ensemble.RandomForestClassifier(n_jobs=-1)
clf = xgb.XGBClassifier()


# In[6]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[7]:


train.df.keys()


# In[8]:


feature_cols = [
    'tfn',
    'ghi_status',
    'abs_ideal_ratio_diff',
    'abs_ideal_ratio_diff mean',
    'abs_ideal_ratio_diff std',
    'abs_ideal_ratio_diff grad',
    'abs_ideal_ratio_diff grad mean', 
    'abs_ideal_ratio_diff grad std',
    'abs_ideal_ratio_diff grad second',
    'abs_ideal_ratio_diff grad second mean',
    'abs_ideal_ratio_diff grad second std',
    'GHI Clearsky GHI pvlib line length ratio',
    'GHI Clearsky GHI pvlib abs_diff',
    'GHI Clearsky GHI pvlib abs_diff mean',
    'GHI Clearsky GHI pvlib abs_diff std'
]

target_cols = ['sky_status']


# In[9]:


for k in feature_cols:
    print(k, train.df[k].isnull().values.any())


# In[10]:


vis = visualize.Visualizer()
vis.plot_corr_matrix(train.df[feature_cols].corr(), feature_cols)


# In[11]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[12]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[13]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib abs_diff'])
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib abs_diff mean'])
# vis.add_line_ser(test.df['GHI Clearsky GHI pvlib abs_diff std'])
vis.add_line_ser(test.df['abs_ideal_ratio_diff grad'] * 100)
vis.add_line_ser(test.df['abs_ideal_ratio_diff grad second'] * 100)
vis.show()


# In[14]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[15]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## Gridsearch

# In[16]:


import warnings


# In[19]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    params={}
    params['max_depth'] = [3, 4, 5]
    params['n_estimators'] = [100, 200]
    params['learning_rate'] = [.1, .01]
    for depth, nest, lr, in itertools.product(params['max_depth'], params['n_estimators'], params['learning_rate']):
        print('Params:')
        print('depth: {}, n_estimators: {}, learning_rate: {}'.format(depth, nest, lr))
        train2 = cs_detection.ClearskyDetection(train.df)
        train2.trim_dates('01-01-1999', '01-01-2014')
        utils.calc_all_window_metrics(train2.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
        test2 = cs_detection.ClearskyDetection(train.df)
        test2.trim_dates('01-01-2014', '01-01-2015')
        clf = xgb.XGBClassifier(max_depth=depth, n_estimators=nest, learning_rate=lr)
        clf.fit(train2.df[feature_cols].values, train2.df[target_cols].values.flatten())
        
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

Params:
depth: 4, n_estimators: 32
	 Train:
		 f1:0.889428
		 recall:0.91632
		 precision:0.864069
	 Test:
		 f1:0.8919171954644752
		 recall:0.8918244227168712
		 precision:0.8920099875156055
Params:
depth: 4, n_estimators: 64
	 Train:
		 f1:0.889789
		 recall:0.912767
		 precision:0.86794
	 Test:
		 f1:0.8914154584332951
		 recall:0.8889120033284793
		 precision:0.8939330543933054
Params:
depth: 4, n_estimators: 128
	 Train:
		 f1:0.889762
		 recall:0.912039
		 precision:0.868547
	 Test:
		 f1:0.8930135557872785
		 recall:0.8907842729353027
		 precision:0.8952540246707088
Params:
depth: 6, n_estimators: 32
	 Train:
		 f1:0.896019
		 recall:0.92724
		 precision:0.866833
	 Test:
		 f1:0.9014956162970603
		 recall:0.9090909090909091
		 precision:0.894026186579378
Params:
depth: 6, n_estimators: 64
	 Train:
		 f1:0.896522
		 recall:0.928609
		 precision:0.866579
	 Test:
		 f1:0.9022680412371135
		 recall:0.9103390888287913
		 precision:0.8943388514203965
Params:
depth: 6, n_estimators: 128
	 Train:
		 f1:0.894717
		 recall:0.922421
		 precision:0.868629
	 Test:
		 f1:0.9000413907284768
		 recall:0.9047222800083212
		 precision:0.895408688490838
Params:
depth: 8, n_estimators: 32
	 Train:
		 f1:0.90242
		 recall:0.94381
		 precision:0.864507
	 Test:
		 f1:0.909925533000102
		 recall:0.9278136051591429
		 precision:0.8927141713370697
Params:
depth: 8, n_estimators: 64
	 Train:
		 f1:0.90265
		 recall:0.94365
		 precision:0.865064
	 Test:
		 f1:0.9098511115643484
		 recall:0.9280216351154567
		 precision:0.8923784756951391
Params:
depth: 8, n_estimators: 128
	 Train:
		 f1:0.903035
		 recall:0.945019
		 precision:0.864622
	 Test:
		 f1:0.910572418007741
		 recall:0.9298939047222801
		 precision:0.8920375174615846
Params:
depth: 10, n_estimators: 32
	 Train:
		 f1:0.909246
		 recall:0.95508
		 precision:0.867609
	 Test:
		 f1:0.9124555160142348
		 recall:0.933430413979613
		 precision:0.8924025457438345
Params:
depth: 10, n_estimators: 64
	 Train:
		 f1:0.909328
		 recall:0.955735
		 precision:0.867218
	 Test:
		 f1:0.9129507364144236
		 recall:0.934886623673809
		 precision:0.8920206431123462
Params:
depth: 10, n_estimators: 128
	 Train:
		 f1:0.909464
		 recall:0.955342
		 precision:0.86779
	 Test:
		 f1:0.9115314215985357
		 recall:0.9323902641980445
		 precision:0.8915854386313905
# In[ ]:





# In[ ]:





# In[32]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[33]:


train.df[train.df['Clearsky GHI pvlib'] > 0]['sky_status'].value_counts()


# In[34]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[35]:


# f1
best_params = {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.1
}
# recall
# best_params = {'max_depth': 6, 'n_estimators': 128, 'class_weight': 'balanced'}
# precision
# best_params = {'max_depth': 7, 'n_estimators': 32, 'class_weight': 'None}


# In[36]:


clf = xgb.XGBClassifier(**best_params)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[37]:


test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[38]:


get_ipython().run_cell_magic('time', '', "pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)")


# In[43]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)
vis.add_line_ser(test.df['GHI Clearsky GHI pvlib abs_diff'])
vis.add_line_ser(test.df['abs_ideal_ratio_diff grad'])
vis.add_line_ser(test.df['abs_ideal_ratio_diff grad second'])
vis.show()


# In[40]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[41]:


print(metrics.f1_score(test.df['sky_status'].values, pred))


# In[42]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])

test2 = cs_detection.ClearskyDetection(test.df)
test2.trim_dates('10-01-2015', None)
probas = clf.predict_proba(test2.df[feature_cols].values)
test2.df['probas'] = 0
test2.df['probas'] = probas[:, 1]

visualize.plot_ts_slider_highligther(test2.df, prob='probas')
# # Train on all NSRDB data, test various freq of ground data

# In[ ]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[ ]:


bar = go.Bar(x=feature_cols, y=clf.feature_importances_)
iplot([bar])


# ## 30 min freq ground data

# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[ ]:


test.trim_dates('10-01-2015', '10-21-2015')


# In[ ]:


test.df = test.df[test.df.index.minute % 30 == 0]


# In[ ]:


test.df.keys()


# In[ ]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[ ]:


train2 = cs_detection.ClearskyDetection(nsrdb.df)
train2.intersection(test.df.index)


# In[ ]:


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
iplot([trace0, trace1, trace2])
# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
# test.df['probas'] = test.df['probas'].rolling(3, center=True).mean()
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 15 min freq ground data

# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[ ]:


test.trim_dates('10-01-2015', '10-17-2015')


# In[ ]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[ ]:


test.df = test.df[test.df.index.minute % 15 == 0]
# test.df = test.df.resample('15T').apply(lambda x: x[len(x) // 2])


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 5, multiproc=False, by_day=False).astype(bool)


# In[ ]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='15min'))
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

nsrdb_clear = train2.df['sky_status'].astype(bool)
ml_clear = pred

test.df['nsrdb_sky'] = nsrdb_clear
test.df['nsrdb_sky'] = test.df['nsrdb_sky'].replace(np.nan, False)
test.df['ml_sky'] = pred

make_subplots(test, train2, 8, 2, 0, width=1000, height=1200)
# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 10 min freq ground data

# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('MST')
test = cs_detection.ClearskyDetection(ground.df)


# In[ ]:


test.trim_dates('10-01-2015', '10-08-2015')


# In[ ]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[ ]:


test.df = test.df[test.df.index.minute % 10 == 0]


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 7, multiproc=False, by_day=False).astype(bool)


# In[ ]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='10min'))
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


# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]

visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## 5 min freq ground data

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


test.df = test.df[test.df.index.minute % 5 == 0]


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 13, multiproc=True, by_day=True).astype(bool)


# In[ ]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-17-2015')
train2.df = train2.df.reindex(pd.date_range(start=train2.df.index[0], end=train2.df.index[-1], freq='5min'))
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




