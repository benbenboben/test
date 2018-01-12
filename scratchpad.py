
# coding: utf-8

# # Table of Contents
#  <p>

# In[506]:


import nsrdb_preprocessor
import os
import cs_detection
import matplotlib.pyplot as plt
import matplotlib
import pv_clf
import numpy as np

get_ipython().magic('matplotlib notebook')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[507]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates(None, '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('01-01-2015', None)


# In[508]:


X = np.asarray([train.df.index.values, train.df['GHI'].values, train.df['Clearsky GHI pvlib'].values]).T


# In[509]:


X.shape


# In[510]:


clf = pv_clf.RandomForestClassifierPV(scale_for_fit=False, random_state=42, n_jobs=-1, max_depth=4, n_estimators=100)

%load_ext line_profiler%lprun -f clf._calc_segment_line_length clf.fit(X, train.df[['sky_status']].values.flatten())
# In[511]:


get_ipython().run_cell_magic('time', '', "clf.fit(X, train.df[['sky_status']].values.flatten())# , sample_weight=clf._calc_abs_1_less_ratio(X[:, 1], X[:, 2]))")


# In[512]:


X = np.asarray([test.df.index.values, test.df['GHI'].values, test.df['Clearsky GHI pvlib'].values]).T


# In[513]:


get_ipython().run_cell_magic('time', '', 'modified = clf.predict(X)')


# In[514]:


from sklearn import metrics


# In[515]:


metrics.accuracy_score(test.df['sky_status'],modified)


# In[516]:


X = np.asarray([test.df.index.values, test.df['GHI'].values, test.df['Clearsky GHI pvlib'].values]).T
features = clf._calculate_extra_metrics(X[:, [1, 2]], 30.0)


# In[517]:


import pandas as pd


# In[518]:


cols = ['ms_cs_diff', 'ms_cs_diff_mean', 'ms_cs_diff_std',
                                  'ms_cs_ratio', 'ms_cs_ratio_mean', 'ms_cs_ratio_std',
                                  'ms_cs_derivative_ratio', 'ms_cs_derivative_ratio_mean', 'ms_cs_derivative_ratio_std',
                                  'ms_cs_derivative2_ratio', 'ms_cs_derivative2_ratio_mean', 'ms_cs_derivative2_ratio_std',
                                  'ms_cs_line_length_ratio']


# In[519]:


features= pd.DataFrame(features, columns=cols)


# In[ ]:





# In[520]:


features.index = test.df.index


# In[521]:


fig, ax = plt.subplots()

ax.plot(test.df['GHI'].index, test.df['GHI'].values)
ax.plot(test.df['Clearsky GHI pvlib'].index, test.df['Clearsky GHI pvlib'].values)
ax.scatter(test.df[modified]['GHI'].index, test.df[modified]['GHI'], c='red', zorder=1e2)
ax.scatter(test.df[test.df['sky_status']]['GHI'].index, test.df[test.df['sky_status']]['GHI'], marker='x', zorder=1e2)


# In[473]:


import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as ff
import cufflinks as cf
init_notebook_mode(True)


# In[474]:


features['GHI'] = test.df['GHI'].values
features['GHIcs'] = test.df['Clearsky GHI pvlib'].values


# In[475]:


features['ms_cs_diff']


# In[476]:


features.iplot()


# In[436]:


from sklearn import ensemble


# In[385]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('MST')
nsrdb.trim_dates('01-01-2000', '01-01-2001')


# In[ ]:


X = np.asarray([nsrdb.df.index.values, nsrdb.df['GHI'].values, nsrdb.df['Clearsky GHI pvlib'].values]).T


# In[ ]:


clf = ensemble.RandomForestClassifier(random_state=42)


# In[ ]:


clf.fit(X, nsrdb.df['sky_status'].values)


# In[ ]:


vanilla = clf.predict(X)


# In[ ]:


np.array_equal(modified, vanilla)


# In[ ]:


np.sum(modified)


# In[ ]:


np.sum(vanilla)


# In[ ]:


clf = ensemble.RandomForestClassifier()


# In[ ]:




