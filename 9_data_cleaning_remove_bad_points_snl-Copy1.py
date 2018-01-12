
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Advanced-scoring" data-toc-modified-id="Advanced-scoring-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Advanced scoring</a></div><div class="lev2 toc-item"><a href="#set-up-model-trained-on-default-data-(scaled-only)" data-toc-modified-id="set-up-model-trained-on-default-data-(scaled-only)-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>set up model trained on default data (scaled only)</a></div><div class="lev2 toc-item"><a href="#set-up-model-trained-on-cleaned-data-(scaled-+-cutoffs-for-metrics)" data-toc-modified-id="set-up-model-trained-on-cleaned-data-(scaled-+-cutoffs-for-metrics)-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>set up model trained on cleaned data (scaled + cutoffs for metrics)</a></div><div class="lev2 toc-item"><a href="#scores" data-toc-modified-id="scores-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>scores</a></div><div class="lev3 toc-item"><a href="#Default-training,-default-testing" data-toc-modified-id="Default-training,-default-testing-131"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Default training, default testing</a></div><div class="lev3 toc-item"><a href="#Default-training,-cleaned-testing" data-toc-modified-id="Default-training,-cleaned-testing-132"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Default training, cleaned testing</a></div><div class="lev3 toc-item"><a href="#Cleaned-training,-default-testing" data-toc-modified-id="Cleaned-training,-default-testing-133"><span class="toc-item-num">1.3.3&nbsp;&nbsp;</span>Cleaned training, default testing</a></div><div class="lev3 toc-item"><a href="#Cleaned-training,-cleaned-testing" data-toc-modified-id="Cleaned-training,-cleaned-testing-134"><span class="toc-item-num">1.3.4&nbsp;&nbsp;</span>Cleaned training, cleaned testing</a></div><div class="lev1 toc-item"><a href="#CV-scoring-and-model-selection-(2010-2015-only)" data-toc-modified-id="CV-scoring-and-model-selection-(2010-2015-only)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>CV scoring and model selection (2010-2015 only)</a></div>

# In[ ]:


import pandas as pd
import numpy as np
import os
import datetime

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from sklearn import tree
from sklearn import ensemble

import itertools

from sklearn import metrics
from sklearn import model_selection

import pvlib
import cs_detection
import utils
import visualize_plotly as visualize
sns.set_style("white")

matplotlib.rcParams['figure.figsize'] = (20., 8.)

from IPython.display import Image

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()
init_notebook_mode(connected=True)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib inline')

get_ipython().magic("config InlineBackend.figure_format = 'retina'")

matplotlib.rcParams.update({'font.size': 16})

import warnings
warnings.filterwarnings(action='ignore')


plt.close('all')

# Train on default data# nsrdb = pd.read_pickle('abq_nsrdb_1.pkl.gz')
detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')train_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates(None, '01-01-2015')
test_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)clf = train_obj.fit_model(clf)pred = test_obj.predict(clf)print(metrics.accuracy_score(test_obj.df['sky_status'], pred))print(metrics.recall_score(test_obj.df['sky_status'], pred))cm = metrics.confusion_matrix(test_obj.df['sky_status'], pred)visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))fig, ax = plt.subplots(figsize=(12, 8))

_ = ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
_ = ax.set_xticks(range(len(clf.feature_importances_)))
_ = ax.set_xticklabels(test_obj.features_, rotation=45)

_ = ax.set_ylabel('Importance')
_ = ax.set_xlabel('Feature')

_ = fig.tight_layout()fig, ax = plt.subplots(figsize=(12, 8))

nsrdb_mask = test_obj.df['sky_status'].values

ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
ax.scatter(test_obj.df[pred & ~nsrdb_mask].index, test_obj.df[pred & ~nsrdb_mask]['GHI'], label='RF only', zorder=10, s=10)
ax.scatter(test_obj.df[nsrdb_mask & ~pred].index, test_obj.df[nsrdb_mask & ~pred]['GHI'], label='NSRDB only', zorder=10, s=10)
ax.scatter(test_obj.df[nsrdb_mask & pred].index, test_obj.df[nsrdb_mask & pred]['GHI'], label='Both', zorder=10, s=10)
_ = ax.legend(bbox_to_anchor=(1.25, 1))vis = visualize.Visualizer()
vis.plot_corr_matrix(test_obj.df[test_obj.features_].corr().values, test_obj.features_)# Train on 'cleaned' dataClean data by setting some cutoffs (by hand).  For this study, GHI/GHIcs mean has to be >= .9 and the coefficient of variance must be less than .1.  Also need to limit GHI-GHIcs (due to low irradiance periods) to detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')train_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates(None, '01-01-2015')
test_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)clf = train_obj.fit_model(clf, ratio_mean_val=0.95, diff_mean_val=50)pred = test_obj.predict(clf)test_obj.filter_labels(ratio_mean_val=0.95, diff_mean_val=50)print(metrics.accuracy_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))print(metrics.recall_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))print(metrics.recall_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))print(metrics.accuracy_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))cm = metrics.confusion_matrix(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']])visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))fig, ax = plt.subplots(figsize=(12, 8))

_ = ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
_ = ax.set_xticks(range(len(clf.feature_importances_)))
_ = ax.set_xticklabels(test_obj.features_, rotation=45)

_ = ax.set_ylabel('Importance')
_ = ax.set_xlabel('Feature')

_ = fig.tight_layout()fig, ax = plt.subplots(figsize=(24, 8))

nsrdb_mask = test_obj.df['sky_status'].values
ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
ax.scatter(test_obj.df[nsrdb_mask & ~pred].index, test_obj.df[nsrdb_mask & ~pred]['GHI'], label='NSRDB only', zorder=10, s=50)
ax.scatter(test_obj.df[pred & ~nsrdb_mask].index, test_obj.df[pred & ~nsrdb_mask]['GHI'], label='RF only', zorder=10, s=50)
ax.scatter(test_obj.df[nsrdb_mask & pred].index, test_obj.df[nsrdb_mask & pred]['GHI'], label='Both', zorder=10, s=50)
_ = ax.legend(bbox_to_anchor=(1.25, 1))

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')# fig, ax = plt.subplots(figsize=(12, 8))

nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})
# _ = ax.legend(bbox_to_anchor=(1.25, 1))

# _ = ax.set_xlabel('Date')
# _ = ax.set_ylabel('GHI / Wm$^{-2}$')
iplot([trace1, trace2, trace3, trace4, trace5])print(len(test_obj.df[nsrdb_mask & ~pred]))Thus far, we have trained on default and cleaned data.  When scoring these methods, we have not cleaned the testing set.  This needs to be done to provide a fair comparison between cleaned and default data sets.  We will also score between default-trained/cleaned-testing and cleaned-trained/default-testing data sets.
# # Advanced scoring

# ## set up model trained on default data (scaled only)

# In[259]:


dflt_detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
dflt_detect_obj.df.index = dflt_detect_obj.df.index.tz_convert('MST')


# In[260]:


dflt_train_obj = cs_detection.ClearskyDetection(dflt_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
dflt_train_obj.trim_dates(None, '01-01-2015')
dflt_test_obj = cs_detection.ClearskyDetection(dflt_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
dflt_test_obj.trim_dates('01-01-2015', None)


# In[261]:


dflt_model = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[263]:


dflt_model = dflt_train_obj.fit_model(dflt_model)


# In[264]:


dflt_pred = dflt_test_obj.predict(dflt_model)


# In[265]:


np.bincount(dflt_test_obj.df['mask'])

dflt_test_obrj.df[['GHI', 'GHI mean', 'GHI gaussian1 mean', 'GHI gaussian10 mean', 'GHI gaussian.1 mean']].iplot()
# ## set up model trained on cleaned data (scaled + cutoffs for metrics)

# In[266]:


clean_detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
clean_detect_obj.df.index = clean_detect_obj.df.index.tz_convert('MST')


# In[267]:


clean_train_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
clean_train_obj.trim_dates(None, '01-01-2015')
clean_test_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
clean_test_obj.trim_dates('01-01-2015', None)


# In[268]:


clean_model = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[269]:


clean_model = clean_train_obj.fit_model(clean_model, ratio_mean_val=0.95, diff_mean_val=50)


# In[270]:


clean_pred = clean_test_obj.predict(clean_model)


# In[271]:


clean_test_obj.filter_labels(ratio_mean_val=0.95, diff_mean_val=50)


# In[272]:


np.bincount(clean_test_obj.df['mask'])


# ## scores

# ### Default training, default testing

# In[273]:


true = dflt_test_obj.df['sky_status']
pred = dflt_pred


# In[274]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[275]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[276]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[277]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[278]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on default trained and default scored')


# ### Default training, cleaned testing

# In[279]:


true = clean_test_obj.df[clean_test_obj.df['mask']]['sky_status']
pred = dflt_pred[clean_test_obj.df['mask']]


# In[280]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[281]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[282]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[283]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[284]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on default trained and cleaned scored')


# ### Cleaned training, default testing

# In[285]:


true = dflt_test_obj.df['sky_status']
pred = clean_pred


# In[286]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[287]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[288]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[289]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[290]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and default scored')


# ### Cleaned training, cleaned testing

# In[291]:


true = clean_test_obj.df[clean_test_obj.df['mask']]['sky_status']
pred = clean_pred[clean_test_obj.df['mask']]


# In[292]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[293]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[294]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[295]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[296]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# # CV scoring and model selection (2010-2015 only)

# In[297]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')

train_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates('01-01-2010', '01-01-2015')
test_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)

## defaultclf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)scores = train_obj.cross_val_score(clf, scoring='f1')print('{} +/- {}'.format(np.mean(scores), np.std(scores)))clf = train_obj.fit_model(clf)pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])true = test_obj.df['sky_status']
cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')## cleand (very lax)scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.90, 'diff_mean_val': 100}, filter_fit=True, filter_score=True)print('{} +/- {}'.format(np.mean(scores), np.std(scores)))clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.90, 'diff_mean_val': 100})pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')## cleaned (lax)scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.93, 'diff_mean_val': 70}, filter_fit=True, filter_score=True)print('{} +/- {}'.format(np.mean(scores), np.std(scores)))clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.93, 'diff_mean_val': 70})pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')## cleanedscores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.95, 'diff_mean_val': 50}, filter_fit=True, filter_score=True)print('{} +/- {}'.format(np.mean(scores), np.std(scores)))clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.95, 'diff_mean_val': 50})pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')## cleaned (aggressive)scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.97, 'diff_mean_val': 30}, filter_fit=True, filter_score=True)print('{} +/- {}'.format(np.mean(scores), np.std(scores)))clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.97, 'diff_mean_val': 30})pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')## cleaned (very aggressive)scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.99, 'diff_mean_val': 10}, filter_fit=True, filter_score=True)print('{} +/- {}'.format(np.mean(scores), np.std(scores)))clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.99, 'diff_mean_val': 10})pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')
for ml in [10, 20, 30]:
    for nest in [32]:
        clf = ensemble.RandomForestClassifier(n_jobs=-1, min_samples_leaf=ml, n_estimators=nest)
        scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.99, 'diff_mean_val': 10}, filter_fit=True, filter_score=True)
        print('n_estimators: {}, min_samples_leaf: {}'.format(nest, ml))
        print('    {} +/- {}'.format(np.mean(scores), np.std(scores)))
# In[298]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=32)
clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.95, 'diff_mean_val': 50})

pred = test_obj.predict(clf)nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])
# In[299]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('10-01-2015', '11-01-2015')


# In[ ]:


pred = detect_obj.predict(clf)


# In[ ]:


trace1 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=detect_obj.df[pred].index, y=detect_obj.df[pred]['GHI'], name='Clear', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3])


# In[ ]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('10-01-2015', '11-01-2015')


# In[ ]:


detect_obj.downsample(5)


# In[ ]:


pred = detect_obj.predict(clf)


# In[ ]:


trace1 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=detect_obj.df[pred].index, y=detect_obj.df[pred]['GHI'], name='Clear', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3])


# In[ ]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('10-01-2015', '11-01-2015')


# In[ ]:


detect_obj.downsample(10)


# In[ ]:


pred = detect_obj.predict(clf)


# In[ ]:


trace1 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=detect_obj.df[pred].index, y=detect_obj.df[pred]['GHI'], name='Clear', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3])


# In[ ]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('10-01-2015', '11-01-2015')


# In[ ]:


detect_obj.downsample(15)


# In[ ]:


pred = detect_obj.predict(clf)


# In[ ]:


trace1 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=detect_obj.df[pred].index, y=detect_obj.df[pred]['GHI'], name='Clear', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3])


# In[ ]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('10-01-2015', '11-01-2015')


# In[ ]:


detect_obj.downsample(30)


# In[ ]:


pred = detect_obj.predict(clf)


# In[ ]:


trace1 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=detect_obj.df.index, y=detect_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=detect_obj.df[pred].index, y=detect_obj.df[pred]['GHI'], name='Clear', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3])


# In[125]:


test_obj.df.index[-1]


# In[ ]:





# In[ ]:




