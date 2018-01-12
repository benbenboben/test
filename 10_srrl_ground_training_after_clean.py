
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Read-and-process-data" data-toc-modified-id="Read-and-process-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read and process data</a></div><div class="lev1 toc-item"><a href="#Visual-inspection-of-ground/nsrdb-data" data-toc-modified-id="Visual-inspection-of-ground/nsrdb-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Visual inspection of ground/nsrdb data</a></div><div class="lev1 toc-item"><a href="#Set-up-ML" data-toc-modified-id="Set-up-ML-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Set up ML</a></div><div class="lev1 toc-item"><a href="#Set-up-cleaned-ML" data-toc-modified-id="Set-up-cleaned-ML-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Set up cleaned ML</a></div><div class="lev1 toc-item"><a href="#Larger-training-set" data-toc-modified-id="Larger-training-set-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Larger training set</a></div><div class="lev1 toc-item"><a href="#10-mins" data-toc-modified-id="10-mins-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>10 mins</a></div><div class="lev1 toc-item"><a href="#15-min" data-toc-modified-id="15-min-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>15 min</a></div><div class="lev1 toc-item"><a href="#30-min" data-toc-modified-id="30-min-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>30 min</a></div><div class="lev1 toc-item"><a href="#SNL" data-toc-modified-id="SNL-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>SNL</a></div>

# In[2]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree

import pytz
import itertools
import visualize
import utils
import pydotplus
import xgboost as xgb

from sklearn import metrics

import pvlib
import cs_detection

import visualize_plotly as visualize

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib inline')


# # Read and process data

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[3]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')


# In[4]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
ground.df.index = ground.df.index.tz_convert('MST')


# # Visual inspection of ground/nsrdb data

# In[5]:


ground.df.index[0], ground.df.index[-1]


# In[6]:


nsrdb.df.index[0], nsrdb.df.index[-1]


# In[7]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[8]:


ground2.trim_dates('01-01-2002', '01-01-2015')
ground2.df = ground2.df[ground2.df.index.minute % 30 == 0]


# In[9]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[10]:


nsrdb2.trim_dates('01-01-2002', '01-01-2015')
nsrdb2.df = nsrdb2.df[nsrdb2.df.index.minute % 30 == 0]


# In[11]:


vis = visualize.Visualizer()
vis.add_line_ser(ground2.df['GHI'], 'Grnd GHI')
vis.add_line_ser(ground2.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
vis.add_line_ser(nsrdb2.df['GHI'], 'NSRDB GHI')
vis.add_line_ser(nsrdb2.df['Clearsky GHI pvlib'], 'NSRDB GHIcs')
vis.show()


# Data missing from ground collection from Jan 1 2006 to Sep 1 2007 and from Jun 1 2012 onwards.

# In[12]:


print(np.sqrt(np.mean((nsrdb2.df[ground2.df['GHI'] > 0]['GHI'] - ground2.df[ground2.df['GHI'] > 0]['GHI'])**2)))


# In[13]:


list_of_demands = (ground2.df['GHI'] > 0) & (nsrdb2.df['sky_status'])
print(np.sqrt(np.mean((nsrdb2.df[list_of_demands]['GHI'] - ground2.df[list_of_demands]['GHI'])**2)))


# In[14]:


list_of_demands = (ground2.df['GHI'] > 0) & (~nsrdb2.df['sky_status'])
print(np.sqrt(np.mean((nsrdb2.df[list_of_demands]['GHI'] - ground2.df[list_of_demands]['GHI'])**2)))


# In[15]:


print((np.mean(np.abs(nsrdb2.df[ground2.df['GHI'] > 0]['GHI'] - ground2.df[ground2.df['GHI'] > 0]['GHI']))))


# In[16]:


list_of_demands = (ground2.df['GHI'] > 0) & (nsrdb2.df['sky_status'])
print((np.mean(np.abs(nsrdb2.df[list_of_demands]['GHI'] - ground2.df[list_of_demands]['GHI']))))


# In[17]:


list_of_demands = (ground2.df['GHI'] > 0) & (~nsrdb2.df['sky_status'])
print((np.mean(np.abs(nsrdb2.df[list_of_demands]['GHI'] - ground2.df[list_of_demands]['GHI']))))


# # Set up ML

# In[18]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[19]:


ground2.trim_dates('09-01-2007', '06-01-2012')

ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[20]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[21]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[22]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[23]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[24]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[25]:


ground2.df.keys()


# In[26]:


nsrdb2.scale_model()

nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)ground2.df['good_training_set'] = nsrdb2.df['quality_mask']ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)
# In[27]:


ground2_train = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[28]:


ground2_train.trim_dates(None, '03-01-2012')


# In[29]:


ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[30]:


ground2_test.trim_dates('03-01-2012', None)


# In[31]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=32, n_jobs=-1)


# In[32]:


from sklearn import model_selection


# In[33]:


ground2_train.df.head()


# In[34]:


ground2_train.df.keys()


# In[35]:


ground2_train.masks_


# In[36]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
# ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[37]:


ground2_train.masks_


# In[38]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:





# In[39]:


# error_mask = val <= 25


# In[40]:


# ground2_train.add_mask('error_mask', error_mask, overwrite=True)


# In[41]:


# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)


# In[42]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 10


# In[43]:


ground2_test.window = 10


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = ground2_test.df
df_ground = df_ground[(df_ground.index >= '05-01-2012') & (df_ground.index < '05-06-2012')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
# vis.add_line_ser(ground2_test.df['GHI'], 'Grnd GHI')
# vis.add_line_ser(ground2_test.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
# vis.add_circle_ser(ground2_test.df[pred]['GHI'], 'ML clear')
# vis.add_circle_ser(ground2_test.df[ground2_test.df['sky_status']]['GHI'], 'NSRDB clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df['every_30th_min']]['sky_status']
pred_labels = pred[ground2_test.df['every_30th_min']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# # Set up cleaned ML

# In[ ]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2.trim_dates('09-01-2007', '06-01-2012')

ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[ ]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[ ]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[ ]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[ ]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[ ]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[ ]:


ground2.df.keys()


# In[ ]:


nsrdb2.scale_model()


# In[ ]:


nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)


# In[ ]:


ground2.df['good_training_set'] = nsrdb2.df['quality_mask']


# In[ ]:


ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)


# In[ ]:


ground2_train = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_train.trim_dates(None, '03-01-2012')


# In[ ]:


ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_test.trim_dates('03-01-2012', None)


# In[ ]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=32, n_jobs=-1)


# In[ ]:


from sklearn import model_selection


# In[ ]:


ground2_train.df.head()


# In[ ]:


ground2_train.df.keys()


# In[ ]:


ground2_train.masks_


# In[ ]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[ ]:


ground2_train.masks_


# In[ ]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:




# error_mask = val <= 25# ground2_train.add_mask('error_mask', error_mask, overwrite=True)# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)
# In[ ]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 10


# In[ ]:


ground2_test.window = 10


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = ground2_test.df
df_ground = df_ground[(df_ground.index >= '05-01-2012') & (df_ground.index < '05-06-2012')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
# vis.add_line_ser(ground2_test.df['GHI'], 'Grnd GHI')
# vis.add_line_ser(ground2_test.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
# vis.add_circle_ser(ground2_test.df[pred]['GHI'], 'ML clear')
# vis.add_circle_ser(ground2_test.df[ground2_test.df['sky_status']]['GHI'], 'NSRDB clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]['sky_status']
pred_labels = pred[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# # Larger training set

# In[ ]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2.trim_dates('09-01-2007', '06-01-2012')


# In[ ]:


ground2.downsample(5)

ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[ ]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[ ]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[ ]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[ ]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[ ]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[ ]:


ground2.df.keys()


# In[ ]:


nsrdb2.scale_model()


# In[ ]:


nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)


# In[ ]:


ground2.df['good_training_set'] = nsrdb2.df['quality_mask']


# In[ ]:


ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)


# In[ ]:


ground2_train = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_train.trim_dates(None, '03-01-2012')


# In[ ]:


ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_test.trim_dates('03-01-2012', None)


# In[ ]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=32, n_jobs=-1)


# In[ ]:


from sklearn import model_selection


# In[ ]:


ground2_train.df.head()


# In[ ]:


ground2_train.df.keys()


# In[ ]:


ground2_train.masks_


# In[ ]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[ ]:


ground2_train.masks_


# In[ ]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:




# error_mask = val <= 25# ground2_train.add_mask('error_mask', error_mask, overwrite=True)# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)
# In[ ]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 6


# In[ ]:


ground2_test.window = 6


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = ground2_test.df
df_ground = df_ground[(df_ground.index >= '05-01-2012') & (df_ground.index < '05-06-2012')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
# vis.add_line_ser(ground2_test.df['GHI'], 'Grnd GHI')
# vis.add_line_ser(ground2_test.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
# vis.add_circle_ser(ground2_test.df[pred]['GHI'], 'ML clear')
# vis.add_circle_ser(ground2_test.df[ground2_test.df['sky_status']]['GHI'], 'NSRDB clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]['sky_status']
pred_labels = pred[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# # 10 mins

# In[ ]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2.trim_dates('09-01-2007', '06-01-2012')


# In[ ]:


ground2.downsample(10)

ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[ ]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[ ]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[ ]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[ ]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[ ]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[ ]:


ground2.df.keys()


# In[ ]:


nsrdb2.scale_model()


# In[ ]:


nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)


# In[ ]:


ground2.df['good_training_set'] = nsrdb2.df['quality_mask']


# In[ ]:


ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)


# In[ ]:


ground2_train = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_train.trim_dates(None, '03-01-2012')


# In[ ]:


ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_test.trim_dates('03-01-2012', None)


# In[ ]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=32, n_jobs=-1)


# In[ ]:


from sklearn import model_selection


# In[ ]:


ground2_train.df.head()


# In[ ]:


ground2_train.df.keys()


# In[ ]:


ground2_train.masks_


# In[ ]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[ ]:


ground2_train.masks_


# In[ ]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:




# error_mask = val <= 25# ground2_train.add_mask('error_mask', error_mask, overwrite=True)# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)
# In[ ]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 6


# In[ ]:


ground2_test.window = 6


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = ground2_test.df
df_ground = df_ground[(df_ground.index >= '05-01-2012') & (df_ground.index < '05-06-2012')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
# vis.add_line_ser(ground2_test.df['GHI'], 'Grnd GHI')
# vis.add_line_ser(ground2_test.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
# vis.add_circle_ser(ground2_test.df[pred]['GHI'], 'ML clear')
# vis.add_circle_ser(ground2_test.df[ground2_test.df['sky_status']]['GHI'], 'NSRDB clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]['sky_status']
pred_labels = pred[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# # 15 min

# In[ ]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2.trim_dates('09-01-2007', '06-01-2012')


# In[ ]:


ground2.downsample(15)

ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[ ]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[ ]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[ ]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[ ]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[ ]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[ ]:


ground2.df.keys()


# In[ ]:


nsrdb2.scale_model()


# In[ ]:


nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)


# In[ ]:


ground2.df['good_training_set'] = nsrdb2.df['quality_mask']


# In[ ]:


ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)


# In[ ]:


ground2_train = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_train.trim_dates(None, '03-01-2012')


# In[ ]:


ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_test.trim_dates('03-01-2012', None)


# In[ ]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=32, n_jobs=-1)


# In[ ]:


from sklearn import model_selection


# In[ ]:


ground2_train.df.head()


# In[ ]:


ground2_train.df.keys()


# In[ ]:


ground2_train.masks_


# In[ ]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[ ]:


ground2_train.masks_


# In[ ]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:




# error_mask = val <= 25# ground2_train.add_mask('error_mask', error_mask, overwrite=True)# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)
# In[ ]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 5


# In[ ]:


ground2_test.window = 5


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = ground2_test.df
df_ground = df_ground[(df_ground.index >= '05-01-2012') & (df_ground.index < '05-06-2012')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
# vis.add_line_ser(ground2_test.df['GHI'], 'Grnd GHI')
# vis.add_line_ser(ground2_test.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
# vis.add_circle_ser(ground2_test.df[pred]['GHI'], 'ML clear')
# vis.add_circle_ser(ground2_test.df[ground2_test.df['sky_status']]['GHI'], 'NSRDB clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]['sky_status']
pred_labels = pred[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# # 30 min

# In[ ]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2.trim_dates('09-01-2007', '06-01-2012')


# In[ ]:


ground2.downsample(30)

ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[ ]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[ ]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[ ]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[ ]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[ ]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[ ]:


ground2.df.keys()


# In[ ]:


nsrdb2.scale_model()


# In[ ]:


nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)


# In[ ]:


ground2.df['good_training_set'] = nsrdb2.df['quality_mask']


# In[ ]:


ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)


# In[ ]:


ground2_train = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_train.trim_dates(None, '03-01-2012')


# In[ ]:


ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2_test.trim_dates('03-01-2012', None)


# In[ ]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=32, n_jobs=-1)


# In[ ]:


from sklearn import model_selection


# In[ ]:


ground2_train.df.head()


# In[ ]:


ground2_train.df.keys()


# In[ ]:


ground2_train.masks_


# In[ ]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[ ]:


ground2_train.masks_


# In[ ]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:




# error_mask = val <= 25# ground2_train.add_mask('error_mask', error_mask, overwrite=True)# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)
# In[ ]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 3


# In[ ]:


ground2_test.window = 3


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = ground2_test.df
df_ground = df_ground[(df_ground.index >= '05-01-2012') & (df_ground.index < '05-06-2012')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
# vis.add_line_ser(ground2_test.df['GHI'], 'Grnd GHI')
# vis.add_line_ser(ground2_test.df['Clearsky GHI pvlib'], 'Grnd GHIcs')
# vis.add_circle_ser(ground2_test.df[pred]['GHI'], 'ML clear')
# vis.add_circle_ser(ground2_test.df[ground2_test.df['sky_status']]['GHI'], 'NSRDB clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]['sky_status']
pred_labels = pred[ground2_test.df['every_30th_min'] & ground2_test.df['good_training_set']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# # SNL

# In[ ]:


ground2 = cs_detection.ClearskyDetection(ground.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')


# In[ ]:


ground2.trim_dates('09-01-2007', '06-01-2012')

ground2.downsample(15)ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007'))
# ground2.add_mask('good_dates', (ground2.df.index <= '01-01-2006') | (ground2.df.index >= '09-01-2007') & ground2.df.index <= '06-01-2012')ground2.df[ground2.df['good_dates']]['GHI'].plot()
# In[ ]:


nsrdb2 = cs_detection.ClearskyDetection(nsrdb.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status', solar_noon_col='abs(t-tnoon)')


# In[ ]:


nsrdb2.trim_dates('09-01-2007', '06-01-2012')

nsrdb2.add_mask('good_dates', (nsrdb2.df.index <= '01-01-2006') | (nsrdb2.df.index >= '09-01-2007'))
# In[ ]:


ground2.df['sky_status'] = nsrdb2.df['sky_status']


# In[ ]:


ground2.df['sky_status'] = ground2.df['sky_status'].fillna(False)


# In[ ]:


ground2.add_mask('every_30th_min', ground2.df.index.minute % 30 == 0)


# In[ ]:


ground2.df.keys()


# In[ ]:


nsrdb2.scale_model()


# In[ ]:


nsrdb2.filter_labels(ratio_mean_val=.95, diff_mean_val=50)


# In[ ]:


ground2.df['good_training_set'] = nsrdb2.df['quality_mask']


# In[ ]:


ground2.df['good_training_set'] = ground2.df['good_training_set'].fillna(False)


# In[ ]:


ground2_train = ground2 # cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')

ground2_train.trim_dates(None, '03-01-2012')ground2_test = cs_detection.ClearskyDetection(ground2.df, 'GHI', 'Clearsky GHI pvlib', solar_noon_col='abs(t-tnoon)')ground2_test.trim_dates('03-01-2012', None)
# In[ ]:


from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
clf = ensemble.RandomForestClassifier(max_depth=8, n_estimators=64, n_jobs=-1)


# In[ ]:


from sklearn import model_selection


# In[ ]:


ground2_train.df.head()


# In[ ]:


ground2_train.df.keys()


# In[ ]:


ground2_train.masks_


# In[ ]:


ground2_train.masks_ = []
ground2_train.masks_.append('every_30th_min')
ground2_train.masks_.append('good_training_set')
# ground2_train.masks_.append('good_dates')


# In[ ]:


ground2_train.masks_


# In[ ]:


len(ground2_train.get_masked_df())

nsrdb2.calc_all_metrics()
ground2.calc_all_metrics()val = np.abs(nsrdb2.df['GHI mean'] - ground2.df['GHI mean'])pd.DataFrame({'nsrdb': nsrdb2.df['GHI'], 'ground': ground2.df['GHI']}).dropna(how='any').plot()
# In[ ]:




# error_mask = val <= 25# ground2_train.add_mask('error_mask', error_mask, overwrite=True)# ground2_train.df['error_mask'] = ground2_train.df['error_mask'].fillna(False)
# In[ ]:


ground2_train.target_col = 'sky_status'
ground2_train.window = 10


# In[ ]:


ground2_test.window = 10


# In[ ]:


clf = ground2_train.fit_model(clf)


# In[ ]:


snl = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')
snl.df.index = snl.df.index.tz_convert('MST')
snl.trim_dates('10-01-2015', '11-01-2015')
ground2_test = snl
ground2_test.window = 10


# In[ ]:


nsrdbsnl = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[ ]:


nsrdbsnl.df.index = nsrdbsnl.df.index.tz_convert('MST')


# In[ ]:


ground2_test.df['sky_status'] = nsrdbsnl.df['sky_status']
ground2_test.df['sky_status'] = ground2_test.df['sky_status'].fillna(False)


# In[ ]:


pred = ground2_test.predict(clf)


# In[ ]:


pred.describe()


# In[ ]:


vis = visualize.Visualizer()
df_ground = snl.df
df_ground = df_ground[(df_ground.index >= '10-01-2015') & (df_ground.index < '10-06-2015')]
vis.add_line_ser(df_ground['GHI'], 'GHI')
vis.add_line_ser(df_ground['Clearsky GHI pvlib'], 'GHIcs')
vis.add_circle_ser(df_ground[pred & ~df_ground['sky_status']]['GHI'], 'ML clear only')
vis.add_circle_ser(df_ground[~pred & df_ground['sky_status']]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(df_ground[pred & df_ground['sky_status']]['GHI'], 'Both clear')
vis.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


ground2_test.df.keys()


# In[ ]:


true_labels = ground2_test.df[ground2_test.df.index.minute % 30 == 0]['sky_status']
pred_labels = pred[ground2_test.df.index.minute % 30 == 0]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# In[ ]:


nsrdbsnl.filter_labels(ratio_mean_val=.95, diff_mean_val=50, overwrite=True)


# In[ ]:


ground2_test.df['good_training_set'] = nsrdbsnl.df['quality_mask']


# In[ ]:


ground2_test.df['good_training_set'] = ground2_test.df['good_training_set'].fillna(False)


# In[ ]:


true_labels = ground2_test.df[(ground2_test.df.index.minute % 30 == 0) & ground2_test.df['good_training_set']]['sky_status']
pred_labels = pred[(ground2_test.df.index.minute % 30 == 0) & ground2_test.df['good_training_set']]


# In[ ]:


metrics.accuracy_score(true_labels, pred_labels)


# In[ ]:


metrics.f1_score(true_labels, pred_labels)


# In[ ]:





# In[ ]:




