
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
nsrdb_ornl = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb_ornl.df.index = nsrdb_ornl.df.index.tz_convert('EST')
nsrdb_ornl.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


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
nsrdb_ornl.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[4]:


utils.calc_all_window_metrics(nsrdb_srrl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_abq.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
utils.calc_all_window_metrics(nsrdb_ornl.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


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
ornl_train, ornl_test = split_df_by_date(nsrdb_ornl, '01-01-1999', '01-01-2015', None)


# In[8]:


X_train = np.vstack((abq_train.df[feature_cols].values, 
                     srrl_train.df[feature_cols].values,
                     ornl_train.df[feature_cols].values))
y_train = np.vstack((abq_train.df[target_cols].values, 
               srrl_train.df[target_cols].values,
               ornl_train.df[target_cols].values))


# # Rough grid search
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
        
        ornl_pred = ornl_test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True, by_day=True)
        ornl_accuracy_score = metrics.accuracy_score(ornl_test.df['sky_status'], ornl_pred)
        ornl_f1_score = metrics.f1_score(abq_test.df['sky_status'], ornl_pred)
        ornl_precision_score = metrics.precision_score(abq_test.df['sky_status'], ornl_pred)
        ornl_recall_score = metrics.recall_score(abq_test.df['sky_status'], ornl_pred)

        print('\t\t abq accuracy: {}'.format(abq_accuracy_score))
        print('\t\t srrl accuracy: {}'.format(srrl_accuracy_score))
        print('\t\t ornl accuracy: {}'.format(ornl_accuracy_score))
        
        print('\t\t abq f1: {}'.format(abq_f1_score))
        print('\t\t srrl f1: {}'.format(srrl_f1_score))
        print('\t\t ornl f1: {}'.format(ornl_f1_score))
        
        print('\t\t abq recall: {}'.format(abq_recall_score))
        print('\t\t srrl recall: {}'.format(srrl_recall_score))
        print('\t\t ornl recall: {}'.format(ornl_recall_score))
        
        print('\t\t abq precision: {}'.format(abq_precision_score))
        print('\t\t srrl precision: {}'.format(srrl_precision_score))
        print('\t\t ornl precision: {}'.format(ornl_precision_score))
        
        results.append({'max_depth': depth, 'n_estimators': nest, 'class_weight': cw, 'min_samples_leaf': min_samples,
                        'abq_accuracy': abq_accuracy_score, 'abq_f1': abq_f1_score, 'abq_recall': abq_recall_score, 'abq_precision': abq_precision_score,
                        'srrl_accuracy': srrl_accuracy_score, 'srrl_f1': srrl_f1_score, 'srrl_recall': srrl_recall_score, 'srrl_precision': srrl_precision_score,
                        'ornl_accuracy': ornl_accuracy_score, 'ornl_f1': ornl_f1_score, 'ornl_recall': ornl_recall_score, 'ornl_precision': ornl_precision_score})results_df = pd.DataFrame(results)results_df['mean accuracy'] = np.mean(results_df[['abq_accuracy', 'srrl_accuracy', 'ornl_accuracy']], axis=1)
results_df['mean f1'] = np.mean(results_df[['abq_f1', 'srrl_f1', 'ornl_f1']], axis=1)
results_df['mean precision'] = np.mean(results_df[['abq_precision', 'srrl_precision', 'ornl_precision']], axis=1)
results_df['mean recall'] = np.mean(results_df[['abq_recall', 'srrl_recall', 'ornl_recall']], axis=1)results_dfresults_df.to_csv('8_combined_direction_features_results.csv')results_df[['mean accuracy', 'mean f1', 'mean recall', 'mean precision']].iplot(kind='box')
# In[9]:


results_df = pd.read_csv('8_combined_direction_features_results.csv')


# In[10]:


results_df[['mean accuracy', 'mean f1', 'mean precision', 'mean recall']].iplot(kind='box')


# In[11]:


results_df[['abq_accuracy', 'srrl_accuracy', 'ornl_accuracy',
            'abq_f1', 'srrl_f1', 'ornl_f1', 
            'abq_precision',  'srrl_precision', 'ornl_precision',
            'abq_recall', 'srrl_recall', 'ornl_recall']].iplot(kind='box')


# In[12]:


results_df['mean accuracy no ornl'] = np.mean(results_df[['abq_accuracy', 'srrl_accuracy']], axis=1)
results_df['mean f1 no ornl'] = np.mean(results_df[['abq_f1', 'srrl_f1']], axis=1)
results_df['mean precision no ornl'] = np.mean(results_df[['abq_precision', 'srrl_precision']], axis=1)
results_df['mean recall no ornl'] = np.mean(results_df[['abq_recall', 'srrl_recall']], axis=1)


# In[13]:


results_df[['mean accuracy', 'mean accuracy no ornl',
            'mean f1', 'mean f1 no ornl',
            'mean precision', 'mean precision no ornl',
            'mean recall', 'mean recall no ornl']].iplot(kind='box')


# ## Best accuracy

# In[14]:


best_accuracy = results_df.iloc[results_df['mean accuracy'].idxmax()]


# In[15]:


best_accuracy


# In[16]:


accuracy_params = best_accuracy[['max_depth', 'min_samples_leaf', 'n_estimators']].to_dict()


# ## Best F1

# In[17]:


best_f1 = results_df.iloc[results_df['mean f1'].idxmax()]


# In[18]:


best_f1


# In[19]:


best_f1.equals(best_accuracy)


# In[20]:


f1_params = best_f1[['max_depth', 'min_samples_leaf', 'n_estimators', 'class_weight']].to_dict()


# ## Best precision

# In[21]:


best_precision = results_df.iloc[results_df['mean precision'].idxmax()]


# In[22]:


best_precision


# In[23]:


print(best_precision.equals(best_accuracy))
print(best_precision.equals(best_f1))


# In[24]:


precision_params = best_precision[['max_depth', 'min_samples_leaf', 'n_estimators']].to_dict()


# ## Best recall

# In[25]:


best_recall = results_df.iloc[results_df['mean recall'].idxmax()]


# In[26]:


best_recall


# In[27]:


print(best_recall.equals(best_accuracy))
print(best_recall.equals(best_f1))
print(best_recall.equals(best_precision))


# In[28]:


recall_params = best_recall[['max_depth', 'min_samples_leaf', 'n_estimators', 'class_weight']].to_dict()


# ## Train method on best model

# In[29]:


X_train = np.vstack((nsrdb_abq.df[feature_cols].values, 
                     nsrdb_srrl.df[feature_cols].values,
                     nsrdb_ornl.df[feature_cols].values))
y_train = np.vstack((nsrdb_abq.df[target_cols].values, 
                     nsrdb_srrl.df[target_cols].values,
                     nsrdb_ornl.df[target_cols].values))


# In[30]:


these_params = f1_params


# In[31]:


these_params['n_estimators'] = 200


# In[32]:


clf = ensemble.RandomForestClassifier(**these_params, n_jobs=-1)


# In[33]:


clf.fit(X_train, y_train)


# # Test on ground data

# ## SRRL

# In[69]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[70]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[71]:


ground.trim_dates('10-01-2011', '11-01-2011')


# In[72]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[73]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[74]:


test= ground


# In[75]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[76]:


vis = visualize.Visualizer()


# In[77]:


srrl_tmp = cs_detection.ClearskyDetection(nsrdb_srrl.df)
srrl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(srrl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[78]:


vis.show()

probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# In[80]:


cm = metrics.confusion_matrix(srrl_tmp.df[srrl_tmp.df['Clearsky GHI pvlib'] > 0]['sky_status'].values, pred[srrl_tmp.df['Clearsky GHI pvlib'] > 0])
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[81]:


true_vals, pred_vals = srrl_tmp.df[srrl_tmp.df['Clearsky GHI pvlib'] > 0]['sky_status'].values, pred[srrl_tmp.df['Clearsky GHI pvlib'] > 0]

print(metrics.f1_score(true_vals, pred_vals))
print(metrics.precision_score(true_vals, pred_vals))
print(metrics.recall_score(true_vals, pred_vals))
print(metrics.accuracy_score(true_vals, pred_vals))


# ## Sandia RTC

# In[82]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[83]:


ground.df.index = ground.df.index.tz_convert('MST')


# In[84]:


ground.trim_dates('10-01-2015', '11-01-2015')


# In[85]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[86]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[87]:


test = ground


# In[88]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[89]:


vis = visualize.Visualizer()


# In[90]:


abq_tmp = cs_detection.ClearskyDetection(nsrdb_abq.df)
abq_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(abq_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[91]:


vis.show()


# In[94]:


cm = metrics.confusion_matrix(abq_tmp.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[95]:


cm = metrics.confusion_matrix(abq_tmp.df[abq_tmp.df['Clearsky GHI pvlib'] > 0]['sky_status'].values, pred[abq_tmp.df['Clearsky GHI pvlib'] > 0])
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[97]:


true_vals, pred_vals = abq_tmp.df[abq_tmp.df['Clearsky GHI pvlib'] > 0]['sky_status'].values, pred[abq_tmp.df['Clearsky GHI pvlib'] > 0]

print(metrics.f1_score(true_vals, pred_vals))
print(metrics.precision_score(true_vals, pred_vals))
print(metrics.recall_score(true_vals, pred_vals))
print(metrics.accuracy_score(true_vals, pred_vals))

probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas[:, 1]
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# ## ORNL

# In[111]:


ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')


# In[112]:


ground.df.index = ground.df.index.tz_convert('EST')


# In[113]:


ground.trim_dates('10-01-2008', '11-01-2008')


# In[114]:


ground.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')
ground.scale_by_irrad('Clearsky GHI pvlib')


# In[115]:


ground.df = ground.df.resample('30T').apply(lambda x: x[len(x) // 2])


# In[116]:


test= ground


# In[117]:


pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, by_day=True, multiproc=True)
pred = pred.astype(bool)


# In[118]:


vis = visualize.Visualizer()


# In[119]:


ornl_tmp = cs_detection.ClearskyDetection(nsrdb_ornl.df)
ornl_tmp.intersection(ground.df.index)
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(ornl_tmp.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')
# vis.add_line_ser(test.df['abs_ideal_ratio_diff'] * 100)


# In[120]:


vis.show()


# In[121]:


cm = metrics.confusion_matrix(ornl_tmp.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[122]:


true_vals, pred_vals = ornl_tmp.df[ornl_tmp.df['Clearsky GHI pvlib'] > 0]['sky_status'].values, pred[ornl_tmp.df['Clearsky GHI pvlib'] > 0]

print(metrics.f1_score(true_vals, pred_vals))
print(metrics.precision_score(true_vals, pred_vals))
print(metrics.recall_score(true_vals, pred_vals))
print(metrics.accuracy_score(true_vals, pred_vals))

cm = metrics.confusion_matrix(true_vals, pred_vals)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])

probas = clf.predict_proba(test.df[feature_cols].values)
test.df['probas'] = 0
test.df['probas'] = probas
visualize.plot_ts_slider_highligther(test.df, prob='probas')
# In[67]:


import pickle


# In[68]:


with open('8_combined_trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:




