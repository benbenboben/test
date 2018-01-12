
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Load-data" data-toc-modified-id="Load-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load data</a></div><div class="lev1 toc-item"><a href="#ML-on-NSRDB-data" data-toc-modified-id="ML-on-NSRDB-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>ML on NSRDB data</a></div><div class="lev2 toc-item"><a href="#NSRDB-Clearsky" data-toc-modified-id="NSRDB-Clearsky-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>NSRDB Clearsky</a></div><div class="lev3 toc-item"><a href="#Visualize" data-toc-modified-id="Visualize-211"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Visualize</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev3 toc-item"><a href="#Visualize" data-toc-modified-id="Visualize-221"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Visualize</a></div><div class="lev2 toc-item"><a href="#Stat-smooth-Clearsky" data-toc-modified-id="Stat-smooth-Clearsky-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Stat smooth Clearsky</a></div><div class="lev3 toc-item"><a href="#Visualize" data-toc-modified-id="Visualize-231"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Visualize</a></div><div class="lev2 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-24"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Conclusion</a></div><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev2 toc-item"><a href="#Statistical-Clearsky" data-toc-modified-id="Statistical-Clearsky-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Statistical Clearsky</a></div>

# In[1]:


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

from sklearn import metrics

import pvlib
import cs_detection
# import visualize
# from bokeh.plotting import output_notebook
# output_notebook()

import visualize_plotly as visualize

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Load data

# Read pickled data from data exploration notebook.

# In[2]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')


# In[3]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# # ML on NSRDB data

# ## NSRDB Clearsky

# In[4]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio')


# In[5]:


nsrdb.df.keys()


# In[6]:


feature_cols = ['GHI', 'Clearsky GHI',
                'abs_diff_ratio', 'GHI mean', 'GHI std', 'GHI max', 'GHI min',
                'Clearsky GHI mean', 'Clearsky GHI std',
                'Clearsky GHI max', 'Clearsky GHI min',
                'abs_diff_ratio mean', 'abs_diff_ratio std', 'abs_diff_ratio max',
                'abs_diff_ratio min',  
                'GHI gradient',
                'GHI gradient mean', 'GHI gradient std', 'GHI gradient max',
                'GHI gradient min',  
                'Clearsky GHI gradient',
                'Clearsky GHI gradient mean', 'Clearsky GHI gradient std',
                'Clearsky GHI gradient max', 'Clearsky GHI gradient min',
                'abs_diff_ratio gradient',
                'abs_diff_ratio gradient mean', 'abs_diff_ratio gradient std',
                'abs_diff_ratio gradient max', 'abs_diff_ratio gradient min',
                'GHI line length', 'Clearsky GHI line length', 
                'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[7]:


clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=len(feature_cols))


# In[8]:


clf = nsrdb.fit_model(feature_cols, target_cols, clf)


# Training vs the clearsky model in NSRDB is quite accurate.  I don't really want to use this clearsky curve though since it's unavailable for ground based measurements.

# ### Visualize

# In[9]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[10]:


clf.fit(train.df[feature_cols], train.df[target_cols])


# In[11]:


pred = clf.predict(test.df[feature_cols]).flatten()


# In[12]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[13]:


vis = visualize.Visualizer()


# In[14]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'Both')


# In[15]:


vis.show()


# In[16]:


fdot = 'srrl_nsrdb_nsrdb_cs.dot'
fpng = 'srrl_nsrdb_nsrdb_cs.png'
tree.export_graphviz(clf, out_file=fdot, feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)
get_ipython().system('dot -Tpng {fdot} -o {fpng}')
Image(fpng)


# ## PVLib Clearsky

# In[17]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[18]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[19]:


nsrdb.df.keys()


# In[20]:


feature_cols = ['GHI', 'Clearsky GHI pvlib',
                'abs_diff_ratio', 'GHI mean', 'GHI std', 'GHI max', 'GHI min',
                'Clearsky GHI pvlib mean', 'Clearsky GHI pvlib std',
                'Clearsky GHI pvlib max', 'Clearsky GHI pvlib min',
                'abs_diff_ratio mean', 'abs_diff_ratio std', 'abs_diff_ratio max',
                'abs_diff_ratio min',  
                'GHI gradient',
                'GHI gradient mean', 'GHI gradient std', 'GHI gradient max',
                'GHI gradient min',  
                'Clearsky GHI pvlib gradient',
                'Clearsky GHI pvlib gradient mean', 'Clearsky GHI pvlib gradient std',
                'Clearsky GHI pvlib gradient max', 'Clearsky GHI pvlib gradient min',
                'abs_diff_ratio gradient',
                'abs_diff_ratio gradient mean', 'abs_diff_ratio gradient std',
                'abs_diff_ratio gradient max', 'abs_diff_ratio gradient min',
                'Clearsky GHI pvlib line length', 'GHI line length', 
                'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[21]:


clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=len(feature_cols))


# In[22]:


clf = nsrdb.fit_model(feature_cols, target_cols, clf)


# Using the PVLib clearsky yields similar results to the NSRDB clearsky curve.  The advantage of the PVLib curve is we can compute it ground based measurements if we have the location.

# ### Visualize

# In[23]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[24]:


clf.fit(train.df[feature_cols], train.df[target_cols])


# In[25]:


pred = clf.predict(test.df[feature_cols]).flatten()


# In[26]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[27]:


vis = visualize.Visualizer()


# In[28]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'Both')


# In[29]:


vis.show()


# In[30]:


fdot = 'srrl_nsrdb_pvlib_cs.dot'
fpng = 'srrl_nsrdb_pvlib_cs.png'
tree.export_graphviz(clf, out_file=fdot, feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)
get_ipython().system('dot -Tpng {fdot} -o {fpng}')
Image(fpng)


# ## Stat smooth Clearsky

# In[31]:


nsrdb.scale_model('GHI', 'Clearsky GHI stat smooth', 'sky_status')


# In[32]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[33]:


nsrdb.df.keys()


# In[34]:


feature_cols = ['GHI', 'Clearsky GHI stat smooth',
                'abs_diff_ratio', 'GHI mean', 'GHI std', 'GHI max', 'GHI min',
                'Clearsky GHI stat smooth mean', 'Clearsky GHI stat smooth std',
                'Clearsky GHI stat smooth max', 'Clearsky GHI stat smooth min',
                'abs_diff_ratio mean', 'abs_diff_ratio std', 'abs_diff_ratio max',
                'abs_diff_ratio min',  
                'GHI gradient',
                'GHI gradient mean', 'GHI gradient std', 'GHI gradient max',
                'GHI gradient min',  
                'Clearsky GHI stat smooth gradient',
                'Clearsky GHI stat smooth gradient mean', 'Clearsky GHI stat smooth gradient std',
                'Clearsky GHI stat smooth gradient max', 'Clearsky GHI stat smooth gradient min',
                'abs_diff_ratio gradient',
                'abs_diff_ratio gradient mean', 'abs_diff_ratio gradient std',
                'abs_diff_ratio gradient max', 'abs_diff_ratio gradient min',
                'Clearsky GHI stat smooth line length', 'GHI line length', 
                'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[35]:


clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=len(feature_cols))


# In[36]:


clf = nsrdb.fit_model(feature_cols, target_cols, clf)


# Using the smoothed statistical clearsky curve gives results that are slightly less accurate than the NSRDB and PVLib curves.  This is also a candidate to be used in future work since it can be computed for ground based measruements (even without location information).

# ### Visualize

# In[37]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[38]:


clf.fit(train.df[feature_cols], train.df[target_cols])


# In[39]:


pred = clf.predict(test.df[feature_cols]).flatten()


# In[40]:


cm = metrics.confusion_matrix(test.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[41]:


vis = visualize.Visualizer()


# In[42]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'Both')


# In[43]:


vis.show()


# In[44]:


fdot = 'srrl_nsrdb_stat_cs.dot'
fpng = 'srrl_nsrdb_stat_cs.png'
tree.export_graphviz(clf, out_file=fdot, feature_names=feature_cols, class_names=['cloudy', 'clear'], filled=True)
get_ipython().system('dot -Tpng {fdot} -o {fpng}')
Image(fpng)


# ## Conclusion

# All three methods perform admirably.  It looks like 'obvious' errors are few and far between.  The big interest now is which method will perform best when NSRDB data is trained and tested on ground data.

# # Ground predictions

# ## PVLib Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[787]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')


# In[788]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[789]:


ground.intersection(nsrdb.df.index)


# In[790]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[791]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[792]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[793]:


feature_cols = ['GHI', 'Clearsky GHI pvlib',
                'abs_diff_ratio', 'GHI mean', 'GHI std', 'GHI max', 'GHI min',
                'Clearsky GHI pvlib mean', 'Clearsky GHI pvlib std',
                'Clearsky GHI pvlib max', 'Clearsky GHI pvlib min',
                'abs_diff_ratio mean', 'abs_diff_ratio std', 'abs_diff_ratio max',
                'abs_diff_ratio min',  
                'GHI gradient',
                'GHI gradient mean', 'GHI gradient std', 'GHI gradient max',
                'GHI gradient min',  
                'Clearsky GHI pvlib gradient',
                'Clearsky GHI pvlib gradient mean', 'Clearsky GHI pvlib gradient std',
                'Clearsky GHI pvlib gradient max', 'Clearsky GHI pvlib gradient min',
                'abs_diff_ratio gradient',
                'abs_diff_ratio gradient mean', 'abs_diff_ratio gradient std',
                'abs_diff_ratio gradient max', 'abs_diff_ratio gradient min',
                'Clearsky GHI pvlib line length', 'GHI line length', 
                'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[794]:


ground.trim_dates('10-01-2010', '12-01-2010')


# In[795]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[796]:


# clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=24, max_depth=8)
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=20, n_estimators=100)
# from sklearn import linear_model
# clf = linear_model.SGDClassifier(class_weight='balanced')


# In[797]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[798]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
pred = pred.astype(bool)


# In[799]:


train.intersection(test.df.index)


# In[800]:


cm = metrics.confusion_matrix(train.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[801]:


metrics.accuracy_score(train.df['sky_status'].values, pred)


# In[802]:


vis = visualize.Visualizer()


# In[803]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[804]:


vis.show()


# In[805]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')

ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
# In[806]:


ground.trim_dates('10-01-2010', '11-01-2010')


# In[807]:


test = ground


# In[808]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[809]:


vis = visualize.Visualizer()


# In[810]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[811]:


vis.show()


# In[812]:


ground = cs_detection.ClearskyDetection.read_pickle('./srrl_ground_1.pkl.gz')


# In[813]:


ground.trim_dates('10-01-2010 06:00:00', '10-02-2010 06:00:00')


# In[814]:


test = ground


# In[815]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[816]:


vis = visualize.Visualizer()


# In[817]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[818]:


vis.show()


# In[819]:


ground = cs_detection.ClearskyDetection.read_pickle('./srrl_ground_1.pkl.gz')


# In[820]:


ground.trim_dates('10-20-2010 06:00:00', '10-21-2010 06:00:00')


# In[821]:


test = ground


# In[822]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[823]:


vis = visualize.Visualizer()


# In[824]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[825]:


vis.show()


# In[ ]:




alpha = 1
n_iter = 5
from scipy.optimize import minimize_scalar
for i in range(n_iter):
    ground.df['Clearsky GHI pvlib'] *= alpha
    ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI pvlib', 
                                   ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
    ground2 = cs_detection.ClearskyDetection(ground.df)
    ground2.trim_dates('10-01-2010', '11-01-2010')
    pred = clf.predict(ground2.df[feature_cols].values)
    clear_vals = ground2.df[pred]['GHI']
    clear_model = ground2.df[pred]['Clearsky GHI pvlib']
    prev_alpha = alpha
    def rmse(alpha):
        return np.sqrt(np.mean((clear_vals - alpha*clear_model)**2))
    alpha = minimize_scalar(rmse).x
    if np.isclose(alpha, prev_alpha):
        break
    
    alphatest = groundpred = clf.predict(test.df[feature_cols].values)vis = visualize.Visualizer()vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')vis.show()
# ## Statistical Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[671]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')


# In[672]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[673]:


ground.intersection(nsrdb.df.index)


# In[674]:


nsrdb.scale_model('GHI', 'Clearsky GHI stat smooth', 'sky_status')


# In[675]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[676]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[677]:


ground.trim_dates('10-01-2010', '11-01-2010')


# In[678]:


feature_cols = ['GHI', 'Clearsky GHI stat smooth',
                'abs_diff_ratio', 'GHI mean', 'GHI std', 'GHI max', 'GHI min',
                'Clearsky GHI stat smooth mean', 'Clearsky GHI stat smooth std',
                'Clearsky GHI stat smooth max', 'Clearsky GHI stat smooth min',
                'abs_diff_ratio mean', 'abs_diff_ratio std', 'abs_diff_ratio max',
                'abs_diff_ratio min',  
                'GHI gradient',
                'GHI gradient mean', 'GHI gradient std', 'GHI gradient max',
                'GHI gradient min',  
                'Clearsky GHI stat smooth gradient',
                'Clearsky GHI stat smooth gradient mean', 'Clearsky GHI stat smooth gradient std',
                'Clearsky GHI stat smooth gradient max', 'Clearsky GHI stat smooth gradient min',
                'abs_diff_ratio gradient',
                'abs_diff_ratio gradient mean', 'abs_diff_ratio gradient std',
                'abs_diff_ratio gradient max', 'abs_diff_ratio gradient min',
                'Clearsky GHI stat smooth line length', 'GHI line length', 'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[679]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[680]:


clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=22)


# In[681]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[682]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 3)
pred = pred.astype(bool)


# In[683]:


train.intersection(test.df.index)


# In[684]:


cm = metrics.confusion_matrix(train.df['sky_status'].values, pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])


# In[685]:


metrics.accuracy_score(train.df['sky_status'].values, pred)


# In[686]:


vis = visualize.Visualizer()


# In[687]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[688]:


vis.show()


# In[689]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')

ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI stat smooth', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
# In[690]:


ground.trim_dates('10-01-2010', '11-01-2010')


# In[691]:


test = ground


# In[692]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[693]:


vis = visualize.Visualizer()


# In[694]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[695]:


vis.show()


# Missing in a big way.  There's large areas of PVLib only.  An iterative fit may help (similar to PVLib).

# In[696]:


ground.trim_dates('10-01-2010 06:00:00', '10-02-2010 06:00:00')


# In[697]:


test = ground


# In[698]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[699]:


vis = visualize.Visualizer()


# In[700]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[701]:


vis.show()


# In[702]:


ground = cs_detection.ClearskyDetection.read_pickle('./srrl_ground_1.pkl.gz')


# In[703]:


ground.trim_dates('10-20-2010 06:00:00', '10-21-2010 06:00:00')


# In[704]:


test = ground


# In[705]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[706]:


vis = visualize.Visualizer()


# In[707]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[708]:


vis.show()

alpha = 1
n_iter = 5
from scipy.optimize import minimize_scalar
for i in range(n_iter):
    ground.df['Clearsky GHI stat smooth'] *= alpha
    ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI stat smooth', 
                                   ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
    ground2 = cs_detection.ClearskyDetection(ground.df)
    ground2.trim_dates('10-01-2010', '11-01-2010')
    pred = clf.predict(ground2.df[feature_cols].values)
    clear_vals = ground2.df[pred]['GHI']
    clear_model = ground2.df[pred]['Clearsky GHI stat smooth']
    prev_alpha = alpha
    def rmse(alpha):
        return np.sqrt(np.mean((clear_vals - alpha*clear_model)**2))
    alpha = minimize_scalar(rmse).x
    if np.isclose(alpha, prev_alpha):
        break
    
    alphatest = groundpred = clf.predict(test.df[feature_cols].values)vis = visualize.Visualizer()vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')vis.show()
# Even small curve fitting helps greatlly...
# Train on ground data## PVLib clearskynsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')Reduce frequency to that of NSRDB.ground.intersection(nsrdb.df.index)ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)ground.df['sky_status nsrdb'] = nsrdb.df['sky_status']train = cs_detection.ClearskyDetection(ground.df)train.trim_dates('01-01-2008', '01-01-2012')feature_cols = ['GHI', 'Clearsky GHI pvlib',
                'abs_diff_ratio', 'GHI mean', 'GHI std', 'GHI max', 'GHI min',
                'Clearsky GHI pvlib mean', 'Clearsky GHI pvlib std',
                'Clearsky GHI pvlib max', 'Clearsky GHI pvlib min',
                'abs_diff_ratio mean', 'abs_diff_ratio std', 'abs_diff_ratio max',
                'abs_diff_ratio min',  
                'GHI gradient',
                'GHI gradient mean', 'GHI gradient std', 'GHI gradient max',
                'GHI gradient min',  
                'Clearsky GHI pvlib gradient',
                'Clearsky GHI pvlib gradient mean', 'Clearsky GHI pvlib gradient std',
                'Clearsky GHI pvlib gradient max', 'Clearsky GHI pvlib gradient min',
                'abs_diff_ratio gradient',
                'abs_diff_ratio gradient mean', 'abs_diff_ratio gradient std',
                'abs_diff_ratio gradient max', 'abs_diff_ratio gradient min']
target_cols = ['sky_status nsrdb']clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=len(feature_cols))train = cs_detection.ClearskyDetection(train.df)
test = cs_detection.ClearskyDetection(train.df)
train.trim_dates(None, '01-1-2011')
test.trim_dates('01-01-2011', None)clf = train.fit_model(feature_cols, target_cols, clf)# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 10)metrics.accuracy_score(train.df['sky_status nsrdb'].values, pred)vis = visualize.Visualizer()
vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'Clearsky GHI pvlib')
vis.add_circle_ser(test.df[(test.df['sky_status nsrdb'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status nsrdb'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(test.df['sky_status nsrdb'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear')
vis.show()The results look pretty good.  The accuracy scores are obviously lower than we see when training on NSRDB data.  There are some ovbious points of issue, but I would say the majority of points fits fairly well.  The NSRDB only points (missed by the DT) do not look 'obviously correct'.Test on NSRDB data.nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)nsrdb.trim_dates('01-01-2011', '01-01-2012')nsrdb_pred = clf.predict(nsrdb.df[feature_cols].values)vis = visualize.Visualizer()
vis.add_line_ser(nsrdb.df['GHI'], 'GHI')
vis.add_line_ser(nsrdb.df['Clearsky GHI pvlib'], 'Clearsky GHI pvlib')
vis.add_circle_ser(nsrdb.df[(nsrdb.df['sky_status'] == 0) & (nsrdb_pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(nsrdb.df[(nsrdb.df['sky_status'] == 1) & (~nsrdb_pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(nsrdb.df[(nsrdb.df['sky_status'] == 1) & (nsrdb_pred)]['GHI'], 'ML+NSRDB clear')
# vis.add_date_slider()
vis.show()metrics.accuracy_score(nsrdb.df['sky_status'].values, nsrdb_pred)It doesn't look like it's viable to train on ground data at this point.  There are many, many more points labeled clear by the DT that aren't in NSRDB.  Zooming in on many of these shows that noisy and low irradiance periods are mislabeled.cm = metrics.confusion_matrix(nsrdb.df['sky_status'].values, nsrdb_pred)
vis = visualize.Visualizer()
vis.plot_confusion_matrix(cm, labels=['cloudy', 'clear'])
# In[ ]:




