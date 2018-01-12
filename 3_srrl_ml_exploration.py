
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


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz', compression='gzip')


# In[3]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz', compression='gzip')


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


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')


# In[40]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[41]:


ground.intersection(nsrdb.df.index)


# In[42]:


nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[43]:


ground.trim_dates('10-01-2010', '12-01-2010')


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
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, smooth=False)
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

for mln in range(4, 2 * len(feature_cols)):
    for md in range(3, 10):
        nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
        nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
        ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
        ground.intersection(nsrdb.df.index)
        ground.trim_dates('01-01-2010', '01-01-2011')
        nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                                      ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
        nsrdb.fit_model(feature_cols, target_cols, clf)
        train = cs_detection.ClearskyDetection(nsrdb.df)
        test = cs_detection.ClearskyDetection(ground.df)
        clf = ensemble.RandomForestClassifier(class_weight='balanced', max_depth=md, max_leaf_nodes=mln)
        clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())
        pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, smooth=False)
        pred = pred.astype(bool)
        train.intersection(test.df.index)
        score = metrics.accuracy_score(train.df['sky_status'].values, pred)
        print(mln, md, score)
# In[ ]:





# In[56]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')

for mln in range(20, 29, 2):
#     for md in range(6, 10):
    md = None
    clf = ensemble.RandomForestClassifier(class_weight='balanced', max_depth=md, max_leaf_nodes=mln, n_jobs=-1)
    nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')
    nsrdb.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')
    ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')
    # ground.intersection(nsrdb.df.index)
    ground.trim_dates('01-01-2010', '01-01-2011')
    nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI pvlib', 
                                  ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
    nsrdb.fit_model(feature_cols, target_cols, clf, cv=False)
    train = cs_detection.ClearskyDetection(nsrdb.df)
    test = cs_detection.ClearskyDetection(ground.df)
    clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())
    pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
    pred = pred.astype(bool)
    # train.intersection(test.df.index)
    score = metrics.f1_score(test.df['sky_status pvlib'].values, pred)
    print(mln, md, score)Train/test split score: 0.9126483115302708
4 None 0.760443622921
Train/test split score: 0.9120145015718487
6 None 0.789303643322
Train/test split score: 0.9114567488084373
8 None 0.789526474159
Train/test split score: 0.9167173714633404
10 None 0.806277334705
Train/test split score: 0.9170089240442146
12 None 0.824806999485
Train/test split score: 0.9177187911976473
14 None 0.80193236715
Train/test split score: 0.9188089443261332
16 None 0.793394819683
Train/test split score: 0.921128688773958
18 None 0.797897914687
Train/test split score: 0.92176249873238
/Users/benellis/duramat/clearsky_detection/cs_detection.py:733: RuntimeWarning:

Scaling did not converge.

20 None 0.810632642212
Train/test split score: 0.9217878511307169
22 None 0.819784211089
Train/test split score: 0.9235751952134672
24 None 0.824168931633
Train/test split score: 0.9204568502180306
26 None 0.822711471611
Train/test split score: 0.9205709360105466
28 None 0.827435047578
Train/test split score: 0.924373795761079
30 None 0.777002479672
Train/test split score: 0.9228653280600345
32 None 0.41388518024
Train/test split score: 0.9247667579353007
34 None 0.754354072597
Train/test split score: 0.9216230605415272
36 None 0.511361129495
Train/test split score: 0.9196455734712504
38 None 0.683550589696
Train/test split score: 0.9239174525910151
40 None 0.618717860816
Train/test split score: 0.924170976574384
42 None 0.724967690319
Train/test split score: 0.9251216915120171
44 None 0.769581482347
Train/test split score: 0.9259709968563026
46 None 0.553980526919

Train/test split score: 0.9048270966433425
4 None 0.830837433097
Train/test split score: 0.9164891998783085
8 None 0.857684895568
Train/test split score: 0.9186441537369435
12 None 0.832876296867
Train/test split score: 0.9192019065003549
16 None 0.822929546584
Train/test split score: 0.9170849812392252
20 None 0.812369719176
Train/test split score: 0.9215343271473482
24 None 0.845680268648
Train/test split score: 0.9226371564750026
28 None 0.753174232007
Train/test split score: 0.9223836324916337
32 None 0.820415075629ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI pvlib', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)
# In[57]:


# clf = ensemble.RandomForestClassifier(class_weight='balanced',  max_leaf_nodes=28, n_jobs=-1)


# In[58]:


ground.trim_dates('10-01-2010', '11-01-2010')


# In[59]:


test = ground


# In[60]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[61]:


vis = visualize.Visualizer()


# In[62]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI pvlib'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')
# vis.add_line_ser(test.df['GHI'].rolling(10, center=True).mean(), 'smooth')


# In[63]:


vis.show()


# In[64]:


clf.feature_importances_


# ## Statistical Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[161]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('srrl_nsrdb_1.pkl.gz')


# In[162]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# We will reduce the frequency of ground based measurements to match NSRDB.

# In[163]:


nsrdb.scale_model('GHI', 'Clearsky GHI stat smooth', 'sky_status')


# In[164]:


ground.intersection(nsrdb.df.index)


# In[165]:


ground.trim_dates('10-01-2010', '12-01-2010')


# In[166]:


nsrdb.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                              ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[167]:


ground.calc_all_window_metrics(3, 30, col1='GHI', col2='Clearsky GHI stat smooth', 
                               ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[168]:


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


# In[169]:


train = cs_detection.ClearskyDetection(nsrdb.df)
test = cs_detection.ClearskyDetection(ground.df)


# In[170]:


clf = ensemble.RandomForestClassifier(class_weight='balanced', max_leaf_nodes=24, n_estimators=100)


# In[171]:


clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())


# In[172]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 3)
pred=pred.astype(bool)


# In[173]:


train.intersection(test.df.index)


# In[174]:


metrics.accuracy_score(train.df['sky_status'].values, pred)


# In[175]:


vis = visualize.Visualizer()


# In[176]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (~pred)]['GHI'], 'NSRDB clear only')
vis.add_circle_ser(test.df[(train.df['sky_status'] == 1) & (pred)]['GHI'], 'ML+NSRDB clear only')


# In[177]:


vis.show()


# In[178]:


ground = cs_detection.ClearskyDetection.read_pickle('srrl_ground_1.pkl.gz')


# In[179]:


# ground.calc_all_window_metrics(1, 10, col1='GHI', col2='Clearsky GHI stat smooth', 
#                                ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=True)


# In[180]:


ground.trim_dates('10-01-2010', '11-01-2010')


# In[181]:


test = ground


# In[182]:


# pred = clf.predict(test.df[feature_cols].values)
pred = test.iter_predict(feature_cols, 'GHI', 'Clearsky GHI stat smooth', clf, 10, smooth=True)
pred = pred.astype(bool)


# In[183]:


vis = visualize.Visualizer()


# In[184]:


vis.add_line_ser(test.df['GHI'], 'GHI')
vis.add_line_ser(test.df['Clearsky GHI stat smooth'], 'GHI_cs')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 0) & (pred)]['GHI'], 'ML clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (~pred)]['GHI'], 'PVLib clear only')
vis.add_circle_ser(test.df[(test.df['sky_status pvlib'] == 1) & (pred)]['GHI'], 'ML+PVLib clear only')


# In[185]:


vis.show()


# In[ ]:





# In[ ]:




