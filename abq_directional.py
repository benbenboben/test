
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#read-data-and-split-for-testing/training" data-toc-modified-id="read-data-and-split-for-testing/training-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>read data and split for testing/training</a></div>

# In[145]:


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
import pv_clf

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

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# ## read data and split for testing/training

# In[178]:


nsrdb = pd.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.index = nsrdb.index.tz_convert('MST')


# In[179]:


test = nsrdb[nsrdb.index >= '01-01-2014']
train = nsrdb[nsrdb.index < '01-01-2014']


# In[180]:


clf = pv_clf.RandomForestClassifierPV()


# In[181]:


X_train = np.asarray([train.index.values, train['GHI'].values, train['Clearsky GHI pvlib'].values]).T
y_train = train['sky_status'].values
X_test = np.asarray([test.index.values, test['GHI'].values, test['Clearsky GHI pvlib'].values]).T
y_test = test['sky_status'].values


# In[182]:


clf.fit(X_train, y_train)


# In[183]:


y_pred = clf.predict(X_test)


# In[184]:


vis = visualize.Visualizer()
vis.add_line_ser(test['GHI'])
vis.add_line_ser(test['Clearsky GHI pvlib'] * clf.alpha_scale)
vis.add_circle_ser(test[y_pred]['GHI'])
vis.show()


# In[185]:


metrics.accuracy_score(y_test, y_pred)


# In[191]:


np.bincount(y_pred) / len(y_pred), np.bincount(y_test) / len(y_test)


# In[192]:


test = nsrdb[nsrdb.index >= '01-01-2013']
train = nsrdb[nsrdb.index < '01-01-2013']


# In[193]:


clf = pv_clf.RandomForestClassifierPV()


# In[194]:


X_train = np.asarray([train.index.values, train['GHI'].values, train['Clearsky GHI pvlib'].values]).T
y_train = train['sky_status'].values
X_test = np.asarray([test.index.values, test['GHI'].values, test['Clearsky GHI pvlib'].values]).T
y_test = test['sky_status'].values


# In[195]:


clf.fit(X_train, y_train)


# In[196]:


y_pred = clf.predict(X_test)


# In[197]:


vis = visualize.Visualizer()
vis.add_line_ser(test['GHI'])
vis.add_line_ser(test['Clearsky GHI pvlib'] * clf.alpha_scale)
vis.add_circle_ser(test[y_pred]['GHI'])
vis.show()


# In[198]:


metrics.accuracy_score(y_test, y_pred)


# In[199]:


np.bincount(y_pred) / len(y_pred), np.bincount(y_test) / len(y_test)


# In[200]:


tscv = TimeSeriesSplit(n_splits=12)


# In[201]:


len(X_train) / 12


# In[206]:


scores = []
for idx1, idx2 in tscv.split(X_train):
    clf = pv_clf.RandomForestClassifierPV()
    clf.fit(X_train[idx1], y_train[idx1])
    pred = clf.predict(X_train[idx2])
    scores.append(metrics.accuracy_score(y_train[idx2], pred))
    print(np.bincount(pred) / len(pred))


# In[207]:


np.mean(scores), np.std(scores)


# In[ ]:




