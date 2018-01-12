
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev2 toc-item"><a href="#Default-classifier" data-toc-modified-id="Default-classifier-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Default classifier</a></div><div class="lev2 toc-item"><a href="#Gridsearch" data-toc-modified-id="Gridsearch-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Gridsearch</a></div><div class="lev2 toc-item"><a href="#Best-recall-model" data-toc-modified-id="Best-recall-model-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Best recall model</a></div><div class="lev2 toc-item"><a href="#Best-accuracy-model" data-toc-modified-id="Best-accuracy-model-24"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Best accuracy model</a></div><div class="lev2 toc-item"><a href="#Best-precision-model" data-toc-modified-id="Best-precision-model-25"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Best precision model</a></div><div class="lev2 toc-item"><a href="#Best-f1-model" data-toc-modified-id="Best-f1-model-26"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Best f1 model</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#30-min-freq-ground-data" data-toc-modified-id="30-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>30 min freq ground data</a></div><div class="lev2 toc-item"><a href="#15-min-freq-ground-data" data-toc-modified-id="15-min-freq-ground-data-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>15 min freq ground data</a></div><div class="lev2 toc-item"><a href="#10-min-freq-ground-data" data-toc-modified-id="10-min-freq-ground-data-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>10 min freq ground data</a></div><div class="lev2 toc-item"><a href="#5-min-freq-ground-data" data-toc-modified-id="5-min-freq-ground-data-34"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>5 min freq ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-35"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

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


# # Ground predictions

# ## PVLib Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[27]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[28]:


len(nsrdb.df)


# # Train/test on NSRDB data to find optimal parameters

# ## Default classifier

# In[29]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates('01-01-2010', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[30]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[31]:


clf = ensemble.RandomForestClassifier(random_state=42, max_depth=6, n_estimators=64, n_jobs=-1)


# In[32]:


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


# In[33]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[34]:


# rough prediction for now, no by_day
pred = train.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=False, by_day=False)
pred = train.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True)


# In[35]:


probas = clf.predict_proba(train.df[feature_cols].values)


# In[36]:


vis = visualize.Visualizer()
vis.add_line_ser(train.df['GHI'])
vis.add_line_ser(train.df['Clearsky GHI pvlib'])
vis.add_circle_ser(train.df[pred]['GHI'])
vis.add_line(train.df.index, probas[:, 1] * 1000, label='prob')
vis.show()


# In[37]:


test_pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[38]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'])
vis.add_line_ser(test.df['Clearsky GHI pvlib'])
vis.add_circle_ser(test.df[test_pred & ~test.df['sky_status'].astype(bool)]['GHI'], label='ml')
vis.add_circle_ser(test.df[~test_pred & test.df['sky_status'].astype(bool)]['GHI'], label='nsrdb')
vis.add_circle_ser(test.df[test_pred & test.df['sky_status'].astype(bool)]['GHI'], label='both')
vis.show()


# In[39]:


cm = metrics.confusion_matrix(test.df['sky_status'], test_pred)
visualize.Visualizer().plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[40]:


sky_status_old = train.df['sky_status']


# In[41]:


sky_status_old.value_counts()


# In[42]:


sky_status_new = pd.Series((probas[:, 1] >= .7) & (train.df['sky_status']), index=train.df.index)


# In[43]:


sky_status_new.value_counts()


# In[44]:


sky_status_old.equals(sky_status_new)


# In[45]:


train.df['sky_status_new'] = 0
train.df['sky_status_new'] = sky_status_new


# In[46]:


target_cols = ['sky_status_new']


# In[47]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status_new')


# In[48]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[49]:


test_pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True).astype(bool)


# In[50]:


vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'])
vis.add_line_ser(test.df['Clearsky GHI pvlib'])
vis.add_circle_ser(test.df[test_pred & ~test.df['sky_status'].astype(bool)]['GHI'], label='ml')
vis.add_circle_ser(test.df[~test_pred & test.df['sky_status'].astype(bool)]['GHI'], label='nsrdb')
vis.add_circle_ser(test.df[test_pred & test.df['sky_status'].astype(bool)]['GHI'], label='both')
vis.show()


# In[51]:


cm = metrics.confusion_matrix(test.df['sky_status'], test_pred)
visualize.Visualizer().plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# Same model as best recall - scroll up.

# # Train on all NSRDB data, test various freq of ground data

# In[ ]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
clf = ensemble.RandomForestClassifier(**best_precision[['max_depth', 'class_weight', 'min_samples_leaf']].to_dict(), n_estimators=100)
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


test.trim_dates('10-01-2015', '11-01-2015')


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


# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
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


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 5, multiproc=True, by_day=True).astype(bool)


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


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 7, multiproc=True, by_day=True).astype(bool)


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


test.trim_dates('10-01-2015', '10-04-2015')


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


# In[ ]:


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


test.trim_dates('10-01-2015', '10-08-2015')


# In[ ]:


test.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
test.scale_by_irrad('Clearsky GHI pvlib')


# In[ ]:


test.df = test.df[test.df.index.minute % 1 == 0]


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, multiproc=True, by_day=True).astype(bool)


# In[ ]:


train2 = cs_detection.ClearskyDetection(train.df)
train2.trim_dates('10-01-2015', '10-08-2015')
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


# In[ ]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# # Save model

# In[ ]:


import pickle


# In[ ]:


with open('8_abq_direction_features_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:


get_ipython().system('ls *abq*')


# # Conclusion

# In general, the clear sky identification looks good.  At lower frequencies (30 min, 15 min) we see good agreement with NSRDB labeled points.  I suspect this could be further improved my doing a larger hyperparameter search, or even doing some feature extraction/reduction/additions.  

# In[ ]:




