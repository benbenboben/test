
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
from sklearn import ensemble

import pvlib
import cs_detection
import visualize_plotly as visualize
# import visualize
# from bokeh.plotting import output_notebook
# output_notebook()

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib notebook')


# # Load data

# Read pickled data from data exploration notebook.

# In[2]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', compression='gzip')


# In[3]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', compression='gzip')


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
                'GHI line length', 'Clearsky GHI line length', 'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[7]:


# clf = tree.DecisionTreeClassifier(class_weight='balanced', max_leaf_nodes=len(feature_cols))
clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=24, n_estimators=100)


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


vis = visualize.Visualizer()


# In[13]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'Both')


# In[14]:


vis.show()


# ## PVLib Clearsky

# In[15]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[16]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[17]:


nsrdb.df.keys()


# In[18]:


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
                'GHI line length', 'Clearsky GHI pvlib line length', 'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[19]:


clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=24, n_estimators=100)


# In[20]:


clf = nsrdb.fit_model(feature_cols, target_cols, clf)


# Using the PVLib clearsky yields similar results to the NSRDB clearsky curve.  The advantage of the PVLib curve is we can compute it ground based measurements if we have the location.

# ### Visualize

# In[21]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[22]:


clf.fit(train.df[feature_cols], train.df[target_cols])


# In[23]:


pred = clf.predict(test.df[feature_cols]).flatten()


# In[24]:


vis = visualize.Visualizer()


# In[25]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'Both')


# In[26]:


vis.show()


# ## Stat smooth Clearsky

# In[27]:


nsrdb.scale_model('GHI', 'Clearsky GHI stat smooth', 'sky_status')


# In[28]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[29]:


nsrdb.df.keys()


# In[30]:


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
                'GHI line length', 'Clearsky GHI stat smooth line length', 'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[31]:


clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=24, n_estimators=100)


# In[32]:


clf = nsrdb.fit_model(feature_cols, target_cols, clf)


# Using the smoothed statistical clearsky curve gives results that are slightly less accurate than the NSRDB and PVLib curves.  This is also a candidate to be used in future work since it can be computed for ground based measruements (even without location information).

# ### Visualize

# In[33]:


train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[34]:


clf.fit(train.df[feature_cols], train.df[target_cols])


# In[35]:


pred = clf.predict(test.df[feature_cols]).flatten()


# In[36]:


vis = visualize.Visualizer()


# In[37]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 0) & (pred)]['GHI'], 'ML only')
vis.add_circle_ser(test.df[(test.df['sky_status'] == 1) & (pred)]['GHI'], 'Both')


# In[38]:


vis.show()


# ## Conclusion

# All three methods perform admirably.  It looks like 'obvious' errors are few and far between.  The big interest now is which method will perform best when NSRDB data is trained and tested on ground data.

# # Ground predictions

# ## PVLib Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[39]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')


# In[40]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[41]:


ground.intersection(nsrdb.df.index)


# In[42]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[43]:


ground.trim_dates('10-01-2015', None)


# In[44]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[45]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[46]:


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
                'GHI line length', 'Clearsky GHI pvlib line length', 
                'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[47]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[48]:


clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=24, n_estimators=100)


# In[49]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[50]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3)
pred = pred.astype(bool)


# In[51]:


train.intersection(test.df.index)


# In[52]:


metrics.accuracy_score(train.df['sky_status'].values, pred)


# In[53]:


vis = visualize.Visualizer()


# In[54]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[55]:


vis.show()


# In[56]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')

ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
# In[57]:


ground.trim_dates('10-01-2015', '11-01-2015')


# In[58]:


test = ground


# In[59]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[60]:


vis = visualize.Visualizer()


# In[61]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[62]:


vis.show()


# ## Statistical Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[113]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')


# In[114]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[115]:


nsrdb.scale_model('GHI', 'Clearsky GHI stat smooth', 'sky_status')


# In[116]:


ground.intersection(nsrdb.df.index)


# In[117]:


ground.trim_dates('10-01-2015', None)


# In[118]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[119]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[120]:


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
                'GHI line length', 'Clearsky GHI stat smooth line length', 
                'abs_ratio_diff line length']
target_cols = ['sky_status']


# In[121]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[122]:


clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=24, n_estimators=100)


# In[123]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[124]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 3)
pred=pred.astype(bool)


# In[125]:


train.intersection(test.df.index)


# In[126]:


metrics.accuracy_score(train.df['sky_status'].values, pred)


# In[127]:


vis = visualize.Visualizer()


# In[128]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[129]:


vis.show()


# In[130]:


ground = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz')


# In[131]:


# ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI stat smooth', 
#                                ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[132]:


ground.trim_dates('10-01-2015', '11-01-2015')


# In[133]:


test = ground


# In[134]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[135]:


vis = visualize.Visualizer()


# In[136]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[137]:


vis.show()


# In[ ]:





# In[ ]:




