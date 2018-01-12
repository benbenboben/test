
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Import-and-setup-data" data-toc-modified-id="Import-and-setup-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import and setup data</a></div><div class="lev1 toc-item"><a href="#Train-model" data-toc-modified-id="Train-model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train model</a></div><div class="lev1 toc-item"><a href="#Train/test-split" data-toc-modified-id="Train/test-split-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train/test split</a></div><div class="lev1 toc-item"><a href="#Rough-grid-search" data-toc-modified-id="Rough-grid-search-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Rough grid search</a></div><div class="lev2 toc-item"><a href="#Best-accuracy" data-toc-modified-id="Best-accuracy-41"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Best accuracy</a></div><div class="lev2 toc-item"><a href="#Best-F1" data-toc-modified-id="Best-F1-42"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Best F1</a></div><div class="lev2 toc-item"><a href="#Best-precision" data-toc-modified-id="Best-precision-43"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Best precision</a></div><div class="lev2 toc-item"><a href="#Best-recall" data-toc-modified-id="Best-recall-44"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Best recall</a></div><div class="lev2 toc-item"><a href="#Train-method-on-best-model" data-toc-modified-id="Train-method-on-best-model-45"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Train method on best model</a></div><div class="lev1 toc-item"><a href="#Test-on-ground-data" data-toc-modified-id="Test-on-ground-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Test on ground data</a></div><div class="lev2 toc-item"><a href="#SRRL" data-toc-modified-id="SRRL-51"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>SRRL</a></div><div class="lev2 toc-item"><a href="#Sandia-RTC" data-toc-modified-id="Sandia-RTC-52"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Sandia RTC</a></div><div class="lev2 toc-item"><a href="#ORNL" data-toc-modified-id="ORNL-53"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>ORNL</a></div>

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
cf.set_config_file(theme='white')
cf.go_offline()
init_notebook_mode(connected=True)

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Import and setup data

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[2]:


nsrdb_srrl = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
nsrdb_srrl.df.index = nsrdb_srrl.df.index.tz_convert('MST')
nsrdb_srrl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
nsrdb_abq = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb_abq.df.index = nsrdb_abq.df.index.tz_convert('MST')
nsrdb_abq.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
# nsrdb_ornl = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
# nsrdb_ornl.df.index = nsrdb_ornl.df.index.tz_convert('EST')
# nsrdb_ornl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# # Train model

# * Train model on all available NSRBD data
#     * ORNL
#     * Sandia RTC
#     * SRRL
# 
# 1. Scale model clearsky (PVLib)
# 2. Calculate training metrics
# 3. Train model

# In[3]:


nsrdb_abq.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
nsrdb_srrl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
# nsrdb_ornl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[4]:


utils.calc_all_window_metrics(nsrdb_srrl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_abq.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
# utils.calc_all_window_metrics(nsrdb_ornl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[5]:


feature_cols = [
    'tfn',
    'abs_ideal_ratio_diff grad',
    'abs_ideal_ratio_diff grad mean', 
    'abs_ideal_ratio_diff grad std',
#     'abs_ideal_ratio_diff grad min',
#     'abs_ideal_ratio_diff grad max',
    'abs_ideal_ratio_diff grad second',
    'abs_ideal_ratio_diff grad second mean',
    'abs_ideal_ratio_diff grad second std',
#     'abs_ideal_ratio_diff grad second min',
#     'abs_ideal_ratio_diff grad second max',
    'GHI Clearsky GHI pvlib line length ratio',
    'GHI Clearsky GHI pvlib ratio', 
    'GHI Clearsky GHI pvlib ratio mean',
    'GHI Clearsky GHI pvlib ratio std',
#     'GHI Clearsky GHI pvlib ratio min',
#     'GHI Clearsky GHI pvlib ratio max',
    'GHI Clearsky GHI pvlib diff',
    'GHI Clearsky GHI pvlib diff mean', 
    'GHI Clearsky GHI pvlib diff std',
#     'GHI Clearsky GHI pvlib diff min',
#     'GHI Clearsky GHI pvlib diff max'
]

target_cols = ['sky_status']


# # Train/test split

# In[6]:


def split_df_by_date(obj, start, mid, end):
    train = cs_detection.ClearskyDetection(obj.df)
    train.trim_dates(start, mid)
    test = cs_detection.ClearskyDetection(obj.df)
    test.trim_dates(mid, end)
    return train, test


# In[7]:


abq_train, abq_test = split_df_by_date(nsrdb_abq, '01-01-1999', '01-01-2015', None)
srrl_train, srrl_test = split_df_by_date(nsrdb_srrl, '01-01-1999', '01-01-2015', None)
# ornl_train, ornl_test = split_df_by_date(nsrdb_ornl, '01-01-1999', '01-01-2015', None)


# In[9]:


X_train = np.vstack((abq_train.df[feature_cols].values, 
                     srrl_train.df[feature_cols].values))
y_train = np.vstack((abq_train.df[target_cols].values, 
               srrl_train.df[target_cols].values))


# # Rough grid search

# In[10]:


import warnings
with warnings.catch_warnings():
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
        clf = ensemble.RandomForestClassifier(max_depth=depth, n_estimators=nest, class_weight=cw, min_samples_leaf=min_samples, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        
        print('\t Scores:')
        abq_pred = abq_test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True)
        abq_accuracy_score = metrics.accuracy_score(abq_test.df['sky_status'], abq_pred)
        abq_f1_score = metrics.f1_score(abq_test.df['sky_status'], abq_pred)
        abq_precision_score = metrics.precision_score(abq_test.df['sky_status'], abq_pred)
        abq_recall_score = metrics.recall_score(abq_test.df['sky_status'], abq_pred)

        srrl_pred = srrl_test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True)
        srrl_accuracy_score = metrics.accuracy_score(srrl_test.df['sky_status'], srrl_pred)
        srrl_f1_score = metrics.f1_score(srrl_test.df['sky_status'], srrl_pred)
        srrl_precision_score = metrics.precision_score(srrl_test.df['sky_status'], srrl_pred)
        srrl_recall_score = metrics.recall_score(srrl_test.df['sky_status'], srrl_pred)
        
        # ornl_pred = ornl_test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True)
        # ornl_accuracy_score = metrics.accuracy_score(ornl_test.df['sky_status'], ornl_pred)
        # ornl_f1_score = metrics.f1_score(abq_test.df['sky_status'], ornl_pred)
        # ornl_precision_score = metrics.precision_score(abq_test.df['sky_status'], ornl_pred)
        # ornl_recall_score = metrics.recall_score(abq_test.df['sky_status'], ornl_pred)

        print('\t\t abq accuracy: {}'.format(abq_accuracy_score))
        print('\t\t srrl accuracy: {}'.format(srrl_accuracy_score))
        # print('\t\t ornl accuracy: {}'.format(ornl_accuracy_score))
        
        print('\t\t abq f1: {}'.format(abq_f1_score))
        print('\t\t srrl f1: {}'.format(srrl_f1_score))
        # print('\t\t ornl f1: {}'.format(ornl_f1_score))
        
        print('\t\t abq recall: {}'.format(abq_recall_score))
        print('\t\t srrl recall: {}'.format(srrl_recall_score))
        # print('\t\t ornl recall: {}'.format(ornl_recall_score))
        
        print('\t\t abq precision: {}'.format(abq_precision_score))
        print('\t\t srrl precision: {}'.format(srrl_precision_score))
        # print('\t\t ornl precision: {}'.format(ornl_precision_score))
        
        results.append({'max_depth': depth, 'n_estimators': nest, 'class_weight': cw, 'min_samples_leaf': min_samples,
                        'abq_accuracy': abq_accuracy_score, 'abq_f1': abq_f1_score, 'abq_recall': abq_recall_score, 'abq_precision': abq_precision_score,
                        'srrl_accuracy': srrl_accuracy_score, 'srrl_f1': srrl_f1_score, 'srrl_recall': srrl_recall_score, 'srrl_precision': srrl_precision_score})


# In[11]:


results_df = pd.DataFrame(results)


# In[13]:


results_df['mean accuracy'] = np.mean(results_df[['abq_accuracy', 'srrl_accuracy']], axis=1)
results_df['mean f1'] = np.mean(results_df[['abq_f1', 'srrl_f1']], axis=1)
results_df['mean precision'] = np.mean(results_df[['abq_precision', 'srrl_precision']], axis=1)
results_df['mean recall'] = np.mean(results_df[['abq_recall', 'srrl_recall']], axis=1)


# In[14]:


results_df


# In[15]:


results_df.to_csv('8_combined_direction_features_results_drop_ornl.csv')

results_df[['mean accuracy', 'mean f1', 'mean recall', 'mean precision']].iplot(kind='box')
# In[16]:


results_df = pd.read_csv('8_combined_direction_features_results_drop_ornl.csv')


# In[17]:


results_df[['mean accuracy', 'mean f1', 'mean precision', 'mean recall']].iplot(kind='box')


# In[19]:


results_df[['abq_accuracy', 'srrl_accuracy',
            'abq_f1', 'srrl_f1',
            'abq_precision',  'srrl_precision',
            'abq_recall', 'srrl_recall']].iplot(kind='box')


# ## Best accuracy

# In[22]:


best_accuracy = results_df.iloc[results_df['mean accuracy'].idxmax()]


# In[23]:


best_accuracy


# In[24]:


accuracy_params = best_accuracy[['max_depth', 'min_samples_leaf', 'n_estimators']].to_dict()


# ## Best F1

# In[25]:


best_f1 = results_df.iloc[results_df['mean f1'].idxmax()]


# In[26]:


best_f1


# In[27]:


best_f1.equals(best_accuracy)


# In[28]:


f1_params = best_f1[['max_depth', 'min_samples_leaf', 'n_estimators', 'class_weight']].to_dict()


# ## Best precision

# In[29]:


best_precision = results_df.iloc[results_df['mean precision'].idxmax()]


# In[30]:


best_precision


# In[31]:


print(best_precision.equals(best_accuracy))
print(best_precision.equals(best_f1))


# In[32]:


precision_params = best_precision[['max_depth', 'min_samples_leaf', 'n_estimators']].to_dict()


# ## Best recall

# In[33]:


best_recall = results_df.iloc[results_df['mean recall'].idxmax()]


# In[34]:


best_recall


# In[35]:


print(best_recall.equals(best_accuracy))
print(best_recall.equals(best_f1))
print(best_recall.equals(best_precision))


# In[36]:


recall_params = best_recall[['max_depth', 'min_samples_leaf', 'n_estimators', 'class_weight']].to_dict()


# ## Train method on best model

# In[37]:


# X_train = np.vstack((nsrdb_abq.df[feature_cols].values, 
#                      nsrdb_srrl.df[feature_cols].values,
#                      nsrdb_ornl.df[feature_cols].values))
# y_train = np.vstack((nsrdb_abq.df[target_cols].values, 
#                      nsrdb_srrl.df[target_cols].values,
#                      nsrdb_ornl.df[target_cols].values))
X_train = np.vstack((nsrdb_abq.df[feature_cols].values, 
                     nsrdb_srrl.df[feature_cols].values))
y_train = np.vstack((nsrdb_abq.df[target_cols].values, 
                     nsrdb_srrl.df[target_cols].values))


# In[38]:


these_params = f1_params


# In[39]:


these_params['n_estimators'] = 200


# In[40]:


clf = ensemble.RandomForestClassifier(**these_params, n_jobs=-1)


# In[41]:


clf.fit(X_train, y_train)


# # Test on ground data

# ## SRRL
ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')ground.df.index = ground.df.index.tz_convert('MST')ground.trim_dates('10-01-2011', '10-08-2011')ground.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')test = groundpred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)vis = visualize.Visualizer()vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)vis.show()probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# In[42]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[43]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[44]:


ground.trim_dates('10-01-2011', '11-01-2011')


# In[45]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[46]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[47]:


test= ground


# In[48]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[49]:


vis = visualize.Visualizer()


# In[50]:


srrl_tmp = cs_detection.ClearskyDetection(nsrdb_srrl.df)
srrl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[51]:


vis.show()


# In[52]:


probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')


# ## Sandia RTC
ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')ground.df.index = ground.df.index.tz_convert('MST')ground.trim_dates('10-01-2015', '10-08-2015')ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')test = groundpred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)vis = visualize.Visualizer()vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)vis.show()probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[ ]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[ ]:


ground.trim_dates('10-01-2015', '11-01-2015')


# In[ ]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[ ]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[ ]:


test = ground


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[ ]:


vis = visualize.Visualizer()


# In[ ]:


abq_tmp = cs_detection.ClearskyDetection(nsrdb_abq.df)
abq_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[ ]:


vis.show()

probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# ## ORNL
ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')ground.trim_dates('10-01-2008', '10-08-2008')ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')ground.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status pvlib')test = ground# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 61, by_day=True, multiproc=True)
pred = pred.astype(bool)vis = visualize.Visualizer()vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)vis.show()probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# In[ ]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[ ]:


ground.df.index = ground.df.index.tz_convert('EST')


# In[ ]:


ground.trim_dates('10-01-2008', '10-08-2008')


# In[ ]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[ ]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[ ]:


test= ground


# In[ ]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[ ]:


vis = visualize.Visualizer()


# In[ ]:


ornl_tmp = cs_detection.ClearskyDetection(nsrdb_ornl.df)
ornl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[ ]:


vis.show()

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


with open('8_combined_trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:




