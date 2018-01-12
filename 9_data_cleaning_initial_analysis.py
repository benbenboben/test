
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Investigate-input-data" data-toc-modified-id="Investigate-input-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Investigate input data</a></div><div class="lev2 toc-item"><a href="#ABQ" data-toc-modified-id="ABQ-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>ABQ</a></div><div class="lev3 toc-item"><a href="#Ratio" data-toc-modified-id="Ratio-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Ratio</a></div><div class="lev3 toc-item"><a href="#Difference" data-toc-modified-id="Difference-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Difference</a></div><div class="lev1 toc-item"><a href="#Train-on-default-data" data-toc-modified-id="Train-on-default-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train on default data</a></div><div class="lev1 toc-item"><a href="#Train-on-'cleaned'-data" data-toc-modified-id="Train-on-'cleaned'-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on 'cleaned' data</a></div><div class="lev1 toc-item"><a href="#Clustering-removal" data-toc-modified-id="Clustering-removal-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Clustering removal</a></div><div class="lev1 toc-item"><a href="#Clustering-removal" data-toc-modified-id="Clustering-removal-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Clustering removal</a></div><div class="lev1 toc-item"><a href="#KMEANS" data-toc-modified-id="KMEANS-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>KMEANS</a></div>

# In[1]:


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
# import pv_clf
import cs_detection
import utils
import visualize_plotly as visualize
sns.set_style("white")

matplotlib.rcParams['figure.figsize'] = (20., 8.)

from IPython.display import Image

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib inline')

get_ipython().magic("config InlineBackend.figure_format = 'retina'")

matplotlib.rcParams.update({'font.size': 16})



plt.close('all')


# # Investigate input data

# ## ABQ

# In[2]:


# nsrdb = pd.read_pickle('abq_nsrdb_1.pkl.gz')
detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.scale_model()
detect_obj.calc_all_metrics()
nsrdb = detect_obj.df


# ### Ratio

# In[115]:


# filter off night times, will skew data
nsrdb_nonight = nsrdb[nsrdb['Clearsky GHI pvlib'] > 0]


# In[116]:


clear_pds = nsrdb_nonight[nsrdb_nonight['sky_status']]
cloudy_pds = nsrdb_nonight[~nsrdb_nonight['sky_status']]


# In[117]:


print(np.mean(clear_pds['GHI/GHIcs']), np.std(clear_pds['GHI/GHIcs']), np.median(clear_pds['GHI/GHIcs']))
print(np.mean(cloudy_pds['GHI/GHIcs']), np.std(cloudy_pds['GHI/GHIcs']), np.median(cloudy_pds['GHI/GHIcs']))

fig, ax = plt.subplots()

_ = ax.boxplot([clear_pds['ghi/ghics ratio'], cloudy_pds['ghi/ghics ratio']], showmeans=True, labels=['clear', 'cloudy'])
_ = ax.set_ylabel('GHI / GHIcs')
# In[120]:


fig, ax = plt.subplots()

sns.boxplot(data=[cloudy_pds['GHI/GHIcs'], clear_pds['GHI/GHIcs']], ax=ax)
_ = ax.set_ylabel('GHI / GHI$_\mathrm{CS}$')
_ = ax.set_xticklabels(['cloudy', 'clear'])
_ = ax.set_xlabel('NSRDB labels')


# Clear periods have relatively tight distribution near a ratio of 1.  Some obvious low outliers need to be fixed/removed.  Cloudy periods show much more spread which is expected.  The mean and median of cloudy periods is quite a bit lower than clear periods  (mean and median of clear ~.95, cloudy mean and median is .66 and .74).  Based on the whiskers there might be mislabeled cloudy points, though these points could also have ratios near 1 but be in noisy periods.

# In[119]:


fig, ax = plt.subplots()

# sns.distplot(clear_pds['GHI/GHIcs'], label='clear', ax=ax, color='C1')
# sns.distplot(cloudy_pds['GHI/GHIcs'], label='cloudy', ax=ax, color='C0')
bins = np.linspace(0, 1.2, 25)
ax.hist(clear_pds['GHI/GHIcs'], bins, label='clear', color='C1', alpha=.75)
ax.hist(cloudy_pds['GHI/GHIcs'], bins, label='cloudy', color='C0', alpha=.75)
ax.set_xlabel('GHI / GHI$_\mathrm{CS}$')
_ = ax.legend()
_ = ax.set_ylabel('Frequency')


# In[ ]:




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

a = ax[0].hist2d(clear_pds['abs(t-tnoon)'], clear_pds['GHI/GHIcs'], cmap='seismic')
_ = fig.colorbar(a[3], ax=ax[0])
ax[0].set_title('Clear')
ax[0].set_xlabel('Time from noon')
ax[0].set_ylabel('GHI / GHI$_\mathrm{CS}$')

a = ax[1].hist2d(cloudy_pds['abs(t-tnoon)'], cloudy_pds['GHI/GHIcs'], cmap='seismic')
_ = fig.colorbar(a[3], ax=ax[1])
ax[1].set_title('Cloudy')
ax[1].set_xlabel('Time from noon')
ax[1].set_ylabel('GHI / GHI$_\mathrm{CS}$')

_ = fig.tight_layout()

# Another view of the same data.  Clear periods have ratios very localized just under 1 while cloudy periods are spread out.

# In[121]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI/GHIcs'])
    cloudy_list.append(cloudy_pds['GHI/GHIcs'])
    year_list.append(year)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
_ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1], color='C0')
_ = ax[1].set_xticklabels(year_list, rotation=45)
_ = ax[1].set_title('Cloudy')
_ = ax[1].set_ylim(0, 1.2)

ax[0].set_ylabel('GHI / GHI$_\mathrm{CS}$')
ax[1].set_ylabel('GHI / GHI$_\mathrm{CS}$')

_ = fig.tight_layout()


# Visually it appears that each year behaves similarly for both cloudy and clear data.

# In[9]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year, nsrdb_nonight.index.month // 3]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI/GHIcs'])
    cloudy_list.append(cloudy_pds['GHI/GHIcs'])
    year_list.append(year)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
_ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1],  color='C0')
_ = ax[1].set_xticklabels(year_list, rotation=90)
_ = ax[1].set_title('Cloudy')
_ = ax[1].set_ylim(0, 1.2)

ax[0].set_ylabel('GHI / GHI$_\mathrm{CS}$')
ax[1].set_ylabel('GHI / GHI$_\mathrm{CS}$')
    
_ = fig.tight_layout()


# A finer view of the year-on-year behavior doesn't show any striking seasonal trends.  During cloudy periods, we do notice that as time goes on, there are less 'outlier' points on the low end of the ratios.

# In[122]:


clear_list, cloudy_list, tfn_list = [], [], []
for tfn, grp in nsrdb_nonight.groupby([nsrdb_nonight['abs(t-tnoon)']]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI/GHIcs'])
    cloudy_list.append(cloudy_pds['GHI/GHIcs'])
    tfn_list.append(tfn)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
_ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1], color='C0')
_ = ax[1].set_xticklabels(tfn_list, rotation=45)
_ = ax[1].set_title('Cloudy')
_ = ax[1].set_ylim(0, 1.2)

_ = ax[1].set_xlabel('Minutes from noon')
_ = ax[0].set_ylabel('GHI / GHI$_{\mathrm{CS}}$')
_ = ax[1].set_ylabel('GHI / GHI$_{\mathrm{CS}}$')


_ = fig.tight_layout()


# As we move away from solar noon (defined as the point of maximum model irradiance), the clear labeled periods include more and more outliers with respect to the GHI/GHIcs ratio.

# In[138]:


sample = nsrdb[nsrdb.index >= '01-01-2012']


# In[139]:


fig, ax = plt.subplots(figsize=(16, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
a = ax.scatter(sample[sample['sky_status']].index, sample[sample['sky_status']]['GHI'], 
               c=sample[sample['sky_status']]['GHI/GHIcs'], label=None, zorder=10, cmap='seismic', s=10)

_ = ax.legend(loc='upper right')

_ = ax.set_title('NSRDB Clear periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI / GHI$_{\mathrm{CS}}$')


# For points labeled as clear by NSRDB, the early morning periods have a noticably lower ratio.  This is visual confirmation of the previous plot.  

# In[140]:


fig, ax = plt.subplots(figsize=(16, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=0.5)
a = ax.scatter(sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0].index, 
               sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI'], 
               c=sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI/GHIcs'], zorder=10, cmap='seismic', s=10, label=None)

_ = ax.legend(loc='upper right')

_ = ax.set_title('NSRDB cloudy periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI / GHI$_{\mathrm{CS}}$')


# In[158]:


fig, ax = plt.subplots(figsize=(16, 8))

subsample = sample[(sample.index >= '03-22-2012') & (sample.index < '03-26-2012')]

_ = ax.plot(subsample.index, subsample['GHI'], label='GHI')
_ = ax.plot(subsample.index, subsample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)

a = ax.scatter(subsample[(~subsample['sky_status']) & (subsample['Clearsky GHI pvlib'] > 0)].index, 
               subsample[(~subsample['sky_status']) & (subsample['Clearsky GHI pvlib'] > 0)]['GHI'], 
               c=subsample[(~subsample['sky_status']) & (subsample['Clearsky GHI pvlib'] > 0)]['GHI/GHIcs'], zorder=10, cmap='seismic', s=40, label=None)

_ = ax.legend(bbox_to_anchor=(.85, .85))

_ = ax.set_title('NSRDB cloudy periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI / GHI$_{\mathrm{CS}}$')


# There appear to be many clear points that are labeled as cloudy.  There are many cases where there is a ratio near 1 during a 'noisy' period, which should not be labeled clear.  We can't assume ratio is good enough.  This seems to be a much bigger problem than clear points having a low ratio.

# In[14]:


num_clear_good_ratio = len(nsrdb[(nsrdb['sky_status']) & (nsrdb['GHI/GHIcs'] >= .9) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_clear_bad_ratio = len(nsrdb[(nsrdb['sky_status']) & (nsrdb['GHI/GHIcs'] < .9) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_good_ratio = len(nsrdb[(~nsrdb['sky_status']) & (nsrdb['GHI/GHIcs'] >= .9) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_bad_ratio = len(nsrdb[(~nsrdb['sky_status']) & (nsrdb['GHI/GHIcs'] < .9) & (nsrdb['Clearsky GHI pvlib'] > 0)])


# In[15]:


print('Clear periods, good ratio: {}'.format(num_clear_good_ratio))
print('Clear periods, bad ratio: {}'.format(num_clear_bad_ratio))
print()
print('Cloudy periods, good ratio: {}'.format(num_cloudy_good_ratio))
print('Cloudy periods, good ratio: {}'.format(num_cloudy_bad_ratio))


# In[126]:


tmp = nsrdb[(nsrdb['sky_status']) & (nsrdb['GHI/GHIcs'] < .9)]

fig, ax = plt.subplots()

# bins = np.linspace(0, 1.2, 25)
ax.hist(tmp['abs(t-tnoon)'], label='clear', color='C1', alpha=.75)
# ax.hist(cloudy_pds['GHI/GHIcs'], bins, label='cloudy', color='C0', alpha=.75)
ax.set_xlabel('|t - t$_\mathrm{noon}$|')
ax.set_xlim((0, 450))
# _ = ax.legend()
_ = ax.set_ylabel('Frequency')
_ = ax.set_title('Clear with ratio < 0.90')


# For NSRDB labeled clear periods:
# * We see that clear periods with a good ratio (>= .9) has 81,000+ samples to 1,100+ samples with a bad ratio (< .9).  It's important to note that just be cause a point has a good ratio, does not necessarily clear.  This does indicate though that an overwhelming majority of clear periods are most likely labeled correctly.
# 
# For NSRDB labeled cloudy periods:
# * Cloudy periods have a much more 'even' ratio count.  There are 20,000+ cloudy points with a good ratio (>= .9) and 56,000+ cloudy points with a bad ratio (< .9).  From visual inspection, it appears that cloudy labels are much less reliable than clear labels provided by NSRDB.
# 
# It would be wise to score classifiers based on their recall score, as the vast majority clear labels from NSRDB appear to be correct (based on the $\mathrm{GHI/GHI_{CS}}$ ratio).  The no special method was used to choose the good/bad ratio cutoff of .9.

# ### Difference

# In[165]:


# filter off night times, will skew data
nsrdb_nonight = nsrdb[nsrdb['Clearsky GHI pvlib'] > 0]


# In[166]:


clear_pds = nsrdb_nonight[nsrdb_nonight['sky_status']]
cloudy_pds = nsrdb_nonight[~nsrdb_nonight['sky_status']]


# In[167]:


print(np.mean(clear_pds['GHI-GHIcs']), np.std(clear_pds['GHI-GHIcs']), np.median(clear_pds['GHI-GHIcs']))
print(np.mean(cloudy_pds['GHI-GHIcs']), np.std(cloudy_pds['GHI-GHIcs']), np.median(cloudy_pds['GHI-GHIcs']))

fig, ax = plt.subplots()

_ = ax.boxplot([clear_pds['ghi/ghics ratio'], cloudy_pds['ghi/ghics ratio']], showmeans=True, labels=['clear', 'cloudy'])
_ = ax.set_ylabel('GHI / GHIcs')
# In[168]:


fig, ax = plt.subplots()

sns.boxplot(data=[cloudy_pds['GHI-GHIcs'], clear_pds['GHI-GHIcs']], ax=ax)
_ = ax.set_ylabel('GHI - GHI$_\mathrm{CS}$')
_ = ax.set_xticklabels(['cloudy', 'clear'])
_ = ax.set_xlabel('NSRDB label')


# Clear periods have relatively tight distribution near a ratio of 1.  Some obvious low outliers need to be fixed/removed.  Cloudy periods show much more spread which is expected.  The mean and median of cloudy periods is quite a bit lower than clear periods  (mean and median of clear ~.95, cloudy mean and median is .66 and .74).  Based on the whiskers there might be mislabeled cloudy points, though these points could also have ratios near 1 but be in noisy periods.

# In[169]:


fig, ax = plt.subplots()

# sns.distplot(clear_pds['GHI/GHIcs'], label='clear', ax=ax, color='C1')
# sns.distplot(cloudy_pds['GHI/GHIcs'], label='cloudy', ax=ax, color='C0')
# bins = np.linspace(0, 1.2, 25)
ax.hist(clear_pds['GHI-GHIcs'], label='clear', color='C1', alpha=.75)
ax.hist(cloudy_pds['GHI-GHIcs'], label='cloudy', color='C0', alpha=.75)
ax.set_xlabel('GHI - GHI$_\mathrm{CS}$')
_ = ax.legend()
_ = ax.set_ylabel('Frequency')


# In[ ]:




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

a = ax[0].hist2d(clear_pds['abs(t-tnoon)'], clear_pds['GHI/GHIcs'], cmap='seismic')
_ = fig.colorbar(a[3], ax=ax[0])
ax[0].set_title('Clear')
ax[0].set_xlabel('Time from noon')
ax[0].set_ylabel('GHI / GHI$_\mathrm{CS}$')

a = ax[1].hist2d(cloudy_pds['abs(t-tnoon)'], cloudy_pds['GHI/GHIcs'], cmap='seismic')
_ = fig.colorbar(a[3], ax=ax[1])
ax[1].set_title('Cloudy')
ax[1].set_xlabel('Time from noon')
ax[1].set_ylabel('GHI / GHI$_\mathrm{CS}$')

_ = fig.tight_layout()

# Another view of the same data.  Clear periods have ratios very localized just under 1 while cloudy periods are spread out.

# In[170]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI-GHIcs'])
    cloudy_list.append(cloudy_pds['GHI-GHIcs'])
    year_list.append(year)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
# _ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1], color='C0')
_ = ax[1].set_xticklabels(year_list, rotation=45)
_ = ax[1].set_title('Cloudy')
# _ = ax[1].set_ylim(0, 1.2)

ax[0].set_ylabel('GHI - GHI$_\mathrm{CS}$')
ax[1].set_ylabel('GHI - GHI$_\mathrm{CS}$')

_ = fig.tight_layout()


# Visually it appears that each year behaves similarly for both cloudy and clear data.

# In[22]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year, nsrdb_nonight.index.month // 3]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI-GHIcs'])
    cloudy_list.append(cloudy_pds['GHI-GHIcs'])
    year_list.append(year)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
# _ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1],  color='C0')
_ = ax[1].set_xticklabels(year_list, rotation=90)
_ = ax[1].set_title('Cloudy')
# _ = ax[1].set_ylim(0, 1.2)

ax[0].set_ylabel('GHI - GHI$_\mathrm{CS}$')
ax[1].set_ylabel('GHI - GHI$_\mathrm{CS}$')
    
_ = fig.tight_layout()


# A finer view of the year-on-year behavior doesn't show any striking seasonal trends.  During cloudy periods, we do notice that as time goes on, there are less 'outlier' points on the low end of the ratios.

# In[171]:


clear_list, cloudy_list, tfn_list = [], [], []
for tfn, grp in nsrdb_nonight.groupby([nsrdb_nonight['abs(t-tnoon)']]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI-GHIcs'])
    cloudy_list.append(cloudy_pds['GHI-GHIcs'])
    tfn_list.append(tfn)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
# _ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1], color='C0')
_ = ax[1].set_xticklabels(tfn_list, rotation=45)
_ = ax[1].set_title('Cloudy')
# _ = ax[1].set_ylim(0, 1.2)

_ = ax[1].set_xlabel('Minutes from noon')
_ = ax[0].set_ylabel('GHI - GHI$_{\mathrm{CS}}$')
_ = ax[1].set_ylabel('GHI - GHI$_{\mathrm{CS}}$')


_ = fig.tight_layout()


# As we move away from solar noon (defined as the point of maximum model irradiance), the clear labeled periods include more and more outliers with respect to the GHI/GHIcs ratio.

# In[24]:


sample = nsrdb[nsrdb.index >= '01-01-2012']


# In[25]:


fig, ax = plt.subplots(figsize=(16, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
a = ax.scatter(sample[sample['sky_status']].index, sample[sample['sky_status']]['GHI'], 
               c=sample[sample['sky_status']]['GHI-GHIcs'], label=None, zorder=10, cmap='seismic', s=10)

_ = ax.legend(loc='upper right')

_ = ax.set_title('NSRDB Clear periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI - GHI$_{\mathrm{CS}}$')


# For points labeled as clear by NSRDB, the early morning periods have a noticably lower ratio.  This is visual confirmation of the previous plot.  

# In[26]:


fig, ax = plt.subplots(figsize=(16, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=0.5)
a = ax.scatter(sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0].index, 
               sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI'], 
               c=sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI-GHIcs'], zorder=10, cmap='seismic', s=10, label=None)

_ = ax.legend(loc='upper right')

_ = ax.set_title('NSRDB cloudy periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI - GHI$_{\mathrm{CS}}$')


# In[172]:


num_clear_good_diff = len(nsrdb[(nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs']) <= 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_clear_bad_diff = len(nsrdb[(nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs']) > 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_good_diff = len(nsrdb[(~nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs']) <= 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_bad_diff = len(nsrdb[(~nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs']) > 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])


# In[173]:


print('Clear periods, good diff: {}'.format(num_clear_good_diff))
print('Clear periods, bad diff: {}'.format(num_clear_bad_diff))
print()
print('Cloudy periods, good diff: {}'.format(num_cloudy_good_diff))
print('Cloudy periods, good diff: {}'.format(num_cloudy_bad_diff))


# There appear to be many clear points that are labeled as cloudy.  There are many cases where there is a ratio near 1 during a 'noisy' period, which should not be labeled clear.  We can't assume ratio is good enough.  This seems to be a much bigger problem than clear points having a low ratio.

# # Train on default data

# In[214]:


# nsrdb = pd.read_pickle('abq_nsrdb_1.pkl.gz')
detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')


# In[215]:


train_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates(None, '01-01-2015')
test_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)


# In[216]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=6, random_state=42)


# In[217]:


clf = train_obj.fit_model(clf)


# In[218]:


pred = test_obj.predict(clf)


# In[219]:


print(metrics.accuracy_score(test_obj.df['sky_status'], pred))


# In[220]:


print(metrics.recall_score(test_obj.df['sky_status'], pred))


# In[221]:


cm = metrics.confusion_matrix(test_obj.df['sky_status'], pred)


# In[222]:


visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))


# In[223]:


fig, ax = plt.subplots(figsize=(12, 8))

_ = ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
_ = ax.set_xticks(range(len(clf.feature_importances_)))
_ = ax.set_xticklabels(test_obj.features_, rotation=45)

_ = ax.set_ylabel('Importance')
_ = ax.set_xlabel('Feature')

_ = fig.tight_layout()


# In[224]:


fig, ax = plt.subplots(figsize=(12, 8))

nsrdb_mask = test_obj.df['sky_status'].values

ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
ax.scatter(test_obj.df[pred & ~nsrdb_mask].index, test_obj.df[pred & ~nsrdb_mask]['GHI'], label='RF only', zorder=10, s=10)
ax.scatter(test_obj.df[nsrdb_mask & ~pred].index, test_obj.df[nsrdb_mask & ~pred]['GHI'], label='NSRDB only', zorder=10, s=10)
ax.scatter(test_obj.df[nsrdb_mask & pred].index, test_obj.df[nsrdb_mask & pred]['GHI'], label='Both', zorder=10, s=10)
_ = ax.legend(bbox_to_anchor=(1.25, 1))


# # Train on 'cleaned' data

# Clean data by setting some cutoffs (by hand).  For this study, GHI/GHIcs mean has to be >= .9 and the coefficient of variance must be less than .1.  Also need to limit GHI-GHIcs (due to low irradiance periods) to 

# In[348]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
original_sky_status = detect_obj.df['sky_status']
detect_obj.scale_model()
detect_obj.calc_all_metrics()
# detect_obj.df['sky_status cleaned'] = (detect_obj.df['GHI/GHIcs mean'] >= .95) & \# (detect_obj.df['GHI/GHIcs std'] <= .1)) | \
#                                       (np.abs(detect_obj.df['GHI-GHIcs mean'] <= 25)) &\# (detect_obj.df['GHI-GHIcs std'] < 10)) & \
#                                       (detect_obj.df['GHI'] > 0)
detect_obj.df['sky_status cleaned'] = ((detect_obj.df['GHI/GHIcs mean'] >= .95) | (np.abs(detect_obj.df['GHI-GHIcs mean']) <= 50)) & (detect_obj.df['GHI'] > 0)


# In[349]:


fig, ax = plt.subplots(figsize=(12, 8))

sample = detect_obj.df[detect_obj.df.index.year == 2014]
nsrdb_mask = sample['sky_status'].values
cleaned_mask = sample['sky_status cleaned'].values

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=.5)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
ax.scatter(sample[cleaned_mask & ~nsrdb_mask].index, sample[cleaned_mask & ~nsrdb_mask]['GHI'], label='Cleaned only', zorder=10, s=10)
ax.scatter(sample[nsrdb_mask & ~cleaned_mask].index, sample[nsrdb_mask & ~cleaned_mask]['GHI'], label='NSRDB only', zorder=10, s=10)
ax.scatter(sample[nsrdb_mask & cleaned_mask].index, sample[nsrdb_mask & cleaned_mask]['GHI'], label='Both', zorder=10, s=10)
_ = ax.legend(bbox_to_anchor=(1.25, 1))
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')


# In[356]:


fig, ax = plt.subplots(figsize=(16, 8))

sample = detect_obj.df[(detect_obj.df.index >= '09-01-2014') & (detect_obj.df.index < '09-06-2014')]
nsrdb_mask = sample['sky_status'].values
cleaned_mask = sample['sky_status cleaned'].values

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=1)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)
ax.scatter(sample[cleaned_mask & ~nsrdb_mask].index, sample[cleaned_mask & ~nsrdb_mask]['GHI'], label='Cleaned only', zorder=10, s=60)
ax.scatter(sample[nsrdb_mask & ~cleaned_mask].index, sample[nsrdb_mask & ~cleaned_mask]['GHI'], label='NSRDB only', zorder=10, s=60)
ax.scatter(sample[nsrdb_mask & cleaned_mask].index, sample[nsrdb_mask & cleaned_mask]['GHI'], label='Both', zorder=10, s=60)
_ = ax.legend(bbox_to_anchor=(1.25, 1))
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')


# In[357]:


fig, ax = plt.subplots(figsize=(16, 8))

sample = detect_obj.df[(detect_obj.df.index >= '08-01-2014') & (detect_obj.df.index < '08-06-2014')] 
nsrdb_mask = sample['sky_status'].values
cleaned_mask = sample['sky_status cleaned'].values

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=1)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)
ax.scatter(sample[cleaned_mask & ~nsrdb_mask].index, sample[cleaned_mask & ~nsrdb_mask]['GHI'], label='Cleaned only', zorder=10, s=60)
ax.scatter(sample[nsrdb_mask & ~cleaned_mask].index, sample[nsrdb_mask & ~cleaned_mask]['GHI'], label='NSRDB only', zorder=10, s=60)
ax.scatter(sample[nsrdb_mask & cleaned_mask].index, sample[nsrdb_mask & cleaned_mask]['GHI'], label='Both', zorder=10, s=60)
_ = ax.legend(bbox_to_anchor=(1.2, 1))
# _ = ax.legend(bbox_to_anchor=(1.25, 1))
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')


# In[228]:


train_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status cleaned')
train_obj.trim_dates(None, '01-01-2015')
test_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)


# In[229]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=6, random_state=42)


# In[230]:


clf = train_obj.fit_model(clf)


# In[231]:


pred = test_obj.predict(clf)


# In[232]:


print(metrics.accuracy_score(test_obj.df['sky_status'], pred))


# In[233]:


print(metrics.recall_score(test_obj.df['sky_status'], pred))


# In[234]:


print(metrics.recall_score(test_obj.df['sky_status cleaned'], pred))


# In[235]:


print(metrics.accuracy_score(test_obj.df['sky_status cleaned'], pred))


# In[236]:


cm = metrics.confusion_matrix(test_obj.df['sky_status'], pred)


# In[237]:


visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))


# In[238]:


cm = metrics.confusion_matrix(test_obj.df['sky_status cleaned'], pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))


# In[239]:


fig, ax = plt.subplots(figsize=(12, 8))

_ = ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
_ = ax.set_xticks(range(len(clf.feature_importances_)))
_ = ax.set_xticklabels(test_obj.features_, rotation=45)

_ = ax.set_ylabel('Importance')
_ = ax.set_xlabel('Feature')

_ = fig.tight_layout()


# In[240]:


fig, ax = plt.subplots(figsize=(12, 8))

nsrdb_mask = test_obj.df['sky_status'].values

ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
ax.scatter(test_obj.df[nsrdb_mask & ~pred].index, test_obj.df[nsrdb_mask & ~pred]['GHI'], label='NSRDB only', zorder=10, s=14)
ax.scatter(test_obj.df[pred & ~nsrdb_mask].index, test_obj.df[pred & ~nsrdb_mask]['GHI'], label='RF only', zorder=10, s=14)
ax.scatter(test_obj.df[nsrdb_mask & pred].index, test_obj.df[nsrdb_mask & pred]['GHI'], label='Both', zorder=10, s=14)
_ = ax.legend(bbox_to_anchor=(1.25, 1))

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')


# In[241]:


print(len(test_obj.df[nsrdb_mask & ~pred]))


# # Clustering removal

# In[358]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates(None, '01-01-2015')
detect_obj.scale_model()
original_sky_status = detect_obj.df['sky_status']
detect_obj.calc_all_metrics()


# In[359]:


from sklearn import cluster


# In[360]:


km = cluster.KMeans(n_clusters=8)


# In[361]:


X = detect_obj.df[detect_obj.features_]


# In[362]:


from sklearn import preprocessing


# In[363]:


ss = preprocessing.StandardScaler().fit(X)


# In[364]:


X_std = ss.transform(X)


# In[365]:


pred = km.fit_predict(X_std)


# In[366]:


detect_obj.df['cluster'] = pred


# In[367]:


cluster_dict = {}
for clstr, grp in detect_obj.df.groupby(detect_obj.df['cluster']):
    xstd = ss.transform(grp[detect_obj.features_])
    cluster_dict[clstr] = np.median(xstd, axis=0)
    print(clstr)
    print('\t avg GHI/GHIcs mean: ', np.mean(grp['GHI/GHIcs mean']), 'avg GHI/GHIcs std: ', np.mean(grp['GHI/GHIcs std']))
    print('\t avg GHI-GHIcs mean: ', np.mean(grp['GHI-GHIcs mean']), 'avg GHI-GHIcs std: ', np.mean(grp['GHI-GHIcs std']))
    print('\t', np.sum(grp['sky_status']) / len(grp))
    print('\t', np.sum(grp['sky_status']) / np.sum(detect_obj.df['sky_status']))
    print('\t', len(grp))


# In[368]:


sky_stat_cluster = (pred == 1) | (pred == 4)
detect_obj.df['sky_status cluster'] = sky_stat_cluster


# In[369]:


fig, ax = plt.subplots(figsize=(16, 8))

sample = detect_obj.df[detect_obj.df.index.year == 2014]

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=.5)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)

nsrdb_mask = sample['sky_status'] == 1
cluster_mask = sample['sky_status cluster']

ax.scatter(sample[(nsrdb_mask) & (~cluster_mask)].index, sample[(nsrdb_mask) & (~cluster_mask)]['GHI'], label='NSRDB only', zorder=100, s=12)
ax.scatter(sample[~nsrdb_mask & cluster_mask].index, sample[~nsrdb_mask & cluster_mask]['GHI'], label='Cluster only', zorder=100, s=12)
ax.scatter(sample[nsrdb_mask & cluster_mask].index, sample[nsrdb_mask & cluster_mask]['GHI'], label='Both', zorder=100, s=12)

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

_ = ax.legend(bbox_to_anchor=(1.2, 1))


# In[374]:


fig, ax = plt.subplots(figsize=(16, 8))

sample = detect_obj.df[(detect_obj.df.index >= '08-01-2014') & (detect_obj.df.index < '08-06-2014')]

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=1)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)

nsrdb_mask = sample['sky_status'] == 1
cluster_mask = sample['sky_status cluster']

ax.scatter(sample[~nsrdb_mask & cluster_mask].index, sample[~nsrdb_mask & cluster_mask]['GHI'], label='Cluster only', zorder=100, s=80)

ax.scatter(sample[(nsrdb_mask) & (~cluster_mask)].index, sample[(nsrdb_mask) & (~cluster_mask)]['GHI'], label='NSRDB only', zorder=100, s=80)
ax.scatter(sample[nsrdb_mask & cluster_mask].index, sample[nsrdb_mask & cluster_mask]['GHI'], label='Both', zorder=100, s=80)

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

_ = ax.legend(bbox_to_anchor=(1.2, 1))


# Clustering looks like the best and most consistent model yet.  NSRDB points generally look "wrong".  There are periods though where NSRDB labels points as clear "on the edge" of a large peak or valley.  The clustering-only points look to correct many 'mistakes' in the NSRDB labeling.  The number of clusters and how to choose them will be further investigated, but for this experiment two clusters were chosen out of a possible 10.  These clusters were chosen based on:
# 
# 1. High average GHI/GHIcs (paired with low average GHI/GHIcs standard deviation).
# 2. Low average GHI-GHIcs (and associated standard deviation).
# 
# Cluster 1 captures most points.  Using the GHI/GHIcs ratio generally works for identifying periods of clarity.  It struggles when both GHI and GHIcs are very small, so small deviations lead to large ratios.  Cluster 2 captures these points by using the GHI-GHIcs criteria.

# In[274]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates(None, '01-01-2015')
# detect_obj.scale_model()
# original_sky_status = detect_obj.df['sky_status']
# detect_obj.calc_all_metrics()


# In[275]:


detect_obj.df['sky_status cluster'] = sky_stat_cluster
detect_obj.target_col = 'sky_status cluster'


# In[276]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=6, random_state=42)


# In[277]:


detect_obj.fit_model(clf)


# In[278]:


test_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.df.index = test_obj.df.index.tz_convert('MST')
test_obj.trim_dates('01-01-2015', None)
# test_obj.scale_model()
# test_obj.calc_all_metrics()


# In[279]:


pred = test_obj.predict(clf)


# In[280]:


fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)

ax.scatter(test_obj.df[test_obj.df['sky_status iter'] & ~test_obj.df['sky_status']].index, 
           test_obj.df[test_obj.df['sky_status iter'] & ~test_obj.df['sky_status']]['GHI'], label='RF only', zorder=100, s=12)
ax.scatter(test_obj.df[~test_obj.df['sky_status iter'] & test_obj.df['sky_status']].index, 
           test_obj.df[~test_obj.df['sky_status iter'] & test_obj.df['sky_status']]['GHI'], label='NSRDB only', zorder=100, s=12)
ax.scatter(test_obj.df[test_obj.df['sky_status iter'] & test_obj.df['sky_status']].index, 
           test_obj.df[test_obj.df['sky_status iter'] & test_obj.df['sky_status']]['GHI'], label='both', zorder=100, s=12)

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

_ = ax.legend(bbox_to_anchor=(1.2, 1))

# ax.scatter(test_obj.df[~pred & test_obj.df['sky_status']].index, test_obj.df[~pred & test_obj.df['sky_status']]['GHI'], 'NSRDB only')
# ax.scatter(test_obj.df[pred & test_obj.df['sky_status']].index, test_obj.df[pred & test_obj.df['sky_status']]['GHI'], 'Both')


# In[320]:


fig, ax = plt.subplots(figsize=(16, 8))

sample = test_obj.df[(test_obj.df.index >= '08-01-2015') & (test_obj.df.index < '08-05-2015')]

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=1)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)

ax.scatter(sample[sample['sky_status iter'] & ~sample['sky_status']].index, sample[sample['sky_status iter'] & ~sample['sky_status']]['GHI'], label='ML model only', zorder=100, s=80)
ax.scatter(sample[~sample['sky_status iter'] & sample['sky_status']].index, sample[~sample['sky_status iter'] & sample['sky_status']]['GHI'], label='NSRDB only', zorder=100, s=80)
ax.scatter(sample[sample['sky_status iter'] & sample['sky_status']].index, sample[sample['sky_status iter'] & sample['sky_status']]['GHI'], label='Both', zorder=100, s=80)

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

_ = ax.legend(bbox_to_anchor=(1.2, 1))

# ax.scatter(test_obj.df[~pred & test_obj.df['sky_status']].index, test_obj.df[~pred & test_obj.df['sky_status']]['GHI'], 'NSRDB only')
# ax.scatter(test_obj.df[pred & test_obj.df['sky_status']].index, test_obj.df[pred & test_obj.df['sky_status']]['GHI'], 'Both')


# In[321]:


probas = clf.predict_proba(test_obj.df[test_obj.features_].values)
test_obj.df['probas'] = probas[:, 1]


# In[347]:


fig, ax = plt.subplots(figsize=(16, 8))

sample = test_obj.df[(test_obj.df.index >= '08-01-2015') & (test_obj.df.index < '08-05-2015')]

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=1)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)

# a = ax.scatter(sample[sample['Clearsky GHI pvlib'] > 0].index, sample[sample['Clearsky GHI pvlib'] > 0]['GHI'], 
#                c=sample[sample['Clearsky GHI pvlib'] > 0]['probas'], cmap='seismic', zorder=100, s=80, label=None)

a = ax.scatter(sample.index, sample['GHI'], 
               c=sample['probas'], cmap='seismic', zorder=100, s=80, label=None)

# _ = ax.scatter(sample[sample['probas'] > 0.5].index, sample[sample['probas'] > 0.5]['GHI'], facecolor='none', edgecolor='black', s=200, label='P$_\mathrm{clear} > 0.5$')


# ax.scatter(sample[~sample['sky_status iter'] & sample['sky_status']].index, sample[~sample['sky_status iter'] & sample['sky_status']]['GHI'], label='NSRDB only', zorder=100, s=80)
# ax.scatter(sample[sample['sky_status iter'] & sample['sky_status']].index, sample[sample['sky_status iter'] & sample['sky_status']]['GHI'], label='Both', zorder=100, s=80)

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

_ = ax.legend(bbox_to_anchor=(1.3, 1))
# _ = ax.legend(bbox_to)
cb = plt.colorbar(a)
cb.set_label('P$_\mathrm{clear}$', fontsize=20)

# ax.scatter(test_obj.df[~pred & test_obj.df['sky_status']].index, test_obj.df[~pred & test_obj.df['sky_status']]['GHI'], 'NSRDB only')
# ax.scatter(test_obj.df[pred & test_obj.df['sky_status']].index, test_obj.df[pred & test_obj.df['sky_status']]['GHI'], 'Both')


# In[282]:


cm = metrics.confusion_matrix(test_obj.df['sky_status'], pred)
visualize.plot_confusion_matrix2(cm, ['cloudy', 'clear'])


# In[283]:


metrics.recall_score(test_obj.df['sky_status'], pred)


# In[284]:


metrics.accuracy_score(test_obj.df['sky_status'], pred)


# In[285]:


clusters = km.predict(ss.transform(test_obj.df[test_obj.features_].values))


# In[286]:


test_obj.df['sky_status cleaned'] = (clusters == 1) | (clusters == 4)


# In[288]:


cm = metrics.confusion_matrix(test_obj.df['sky_status cleaned'], pred)
visualize.plot_confusion_matrix2(cm, ['cloudy', 'clear'])


# In[289]:


metrics.recall_score(test_obj.df['sky_status cleaned'], pred)


# In[290]:


metrics.accuracy_score(test_obj.df['sky_status cleaned'], pred)


# In[291]:


from sklearn import manifold


# In[293]:


X_std = ss.transform(test_obj.df[test_obj.features_].values)


# In[295]:


X_embedded = manifold.TSNE().fit_transform(X_std)


# In[299]:


fig, ax = plt.subplots()

ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=test_obj.df['sky_status cleaned'])


# In[304]:


X_embedded_tsne = manifold.TSNE().fit_transform(test_obj.df[test_obj.features_])


# In[306]:


fig, ax = plt.subplots()

ax.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=test_obj.df['sky_status cleaned'])


# In[307]:


X_embedded_isomap = manifold.Isomap().fit_transform(test_obj.df[test_obj.features_])


# In[316]:


fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(X_embedded_isomap[:, 0], X_embedded_isomap[:, 1], c=test_obj.df['sky_status cleaned'], cmap='seismic')

_ = ax.set_xlim((-2000, 500))
_= ax.set_ylim((-1000, 1000))


# In[309]:


Xstd_embedded_isomap = manifold.Isomap().fit_transform(ss.transform(test_obj.df[test_obj.features_]))


# In[310]:


fig, ax = plt.subplots()

ax.scatter(Xstd_embedded_isomap[:, 0], Xstd_embedded_isomap[:, 1], c=test_obj.df['sky_status cleaned'])


# In[78]:


test_obj = cs_detection.ClearskyDetection.read_pickle('abq_ground_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib')


# In[79]:


test_obj.df.index = test_obj.df.index.tz_convert('MST')


# In[80]:


test_obj.trim_dates('10-01-2015', '01-01-2016')


# In[81]:


test_obj.downsample(30)


# In[82]:


test_obj.calc_all_metrics()


# In[83]:


X = test_obj.df[test_obj.features_]


# In[84]:


X_std = ss.transform(X)


# In[85]:


ground_pred = km.predict(X_std)


# In[86]:


ground_pred = (ground_pred == 0) | (ground_pred == 2)

ground_pred = test_obj.predict(clf)
# In[87]:


fig, ax = plt.subplots()

ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)

ax.scatter(test_obj.df[ground_pred].index, test_obj.df[ground_pred]['GHI'], label='RF', zorder=100, s=12)
# ax.scatter(test_obj.df[~pred & test_obj.df['sky_status']].index, test_obj.df[~pred & test_obj.df['sky_status']]['GHI'], label='NSRDB only', zorder=100, s=12)
# ax.scatter(test_obj.df[pred & test_obj.df['sky_status']].index, test_obj.df[pred & test_obj.df['sky_status']]['GHI'], label='both', zorder=100, s=12)

_ = ax.legend()

# ax.scatter(test_obj.df[~pred & test_obj.df['sky_status']].index, test_obj.df[~pred & test_obj.df['sky_status']]['GHI'], 'NSRDB only')
# ax.scatter(test_obj.df[pred & test_obj.df['sky_status']].index, test_obj.df[pred & test_obj.df['sky_status']]['GHI'], 'Both')

# Clustering removaldetect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('01-01-2010', '01-01-2012')
detect_obj.scale_model()
original_sky_status = detect_obj.df['sky_status']
detect_obj.calc_all_metrics()from sklearn import cluster# km = cluster.KMeans(n_clusters=10, init='random')
# km = cluster.DBSCAN()
# km = cluster.SpectralClustering()
km = cluster.AgglomerativeClustering(n_clusters=10)
# km = cluster.Birch(n_clusters=10)X = detect_obj.df[detect_obj.features_]from sklearn import preprocessingss = preprocessing.StandardScaler().fit(X)X_std = ss.transform(X)pred = km.fit_predict(X_std)np.unique(pred)detect_obj.df['cluster'] = predcluster_dict = {}
for clstr, grp in detect_obj.df.groupby(detect_obj.df['cluster']):
    xstd = ss.transform(grp[detect_obj.features_])
    # cluster_dict[clstr] = np.asarray([np.mean(grp[feature]) for feature in detect_obj.features_])
    cluster_dict[clstr] = np.median(xstd, axis=0)
    print(clstr)
    print('avg GHI/GHIcs mean: ', np.mean(grp['GHI/GHIcs mean']))
    print('avg GHI/GHIcs std: ', np.mean(grp['GHI/GHIcs std']))
    print('avg GHI-GHIcs mean: ', np.mean(grp['GHI-GHIcs mean']))
    print('avg GHI-GHIcs std: ', np.mean(grp['GHI-GHIcs mean']))fig, ax = plt.subplots()

sample = detect_obj.df

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=.5)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)

nsrdb_mask = sample['sky_status'] == 1
cluster_mask = (sample['cluster'] == 2)| (sample['cluster'] == 5) #  | (sample['cluster'] == 8) cluster 8 is night time

ax.scatter(sample[(nsrdb_mask) & (~cluster_mask)].index, sample[(nsrdb_mask) & (~cluster_mask)]['GHI'], label='NSRDB only', zorder=100)
ax.scatter(sample[~nsrdb_mask & cluster_mask].index, sample[~nsrdb_mask & cluster_mask]['GHI'], label='Cluster only', zorder=100)
ax.scatter(sample[nsrdb_mask & cluster_mask].index, sample[nsrdb_mask & cluster_mask]['GHI'], label='Both', zorder=100)

_ = ax.legend()Clustering looks like the best and most consistent model yet.  NSRDB points generally look "wrong".  There are periods though where NSRDB labels points as clear "on the edge" of a large peak or valley.  The clustering-only points look to correct many 'mistakes' in the NSRDB labeling.  The number of clusters and how to choose them will be further investigated, but for this experiment two clusters were chosen out of a possible 10.  These clusters were chosen based on:

1. High average GHI/GHIcs (paired with low average GHI/GHIcs standard deviation).
2. Low average GHI-GHIcs (and associated standard deviation).

Cluster 1 captures most points.  Using the GHI/GHIcs ratio generally works for identifying periods of clarity.  It struggles when both GHI and GHIcs are very small, so small deviations lead to large ratios.  Cluster 2 captures these points by using the GHI-GHIcs criteria.cluster_list = [cluster_dict[key] for key in sorted(cluster_dict)]
from scipy import linalg
def pred_with_dict(X_std, clusterlist):
    predictions = []
    for x in X_std:
        distances = [linalg.norm(x - i, ord=np.inf) for i in clusterlist]
        # print(distances)
        predictions.append(np.argmin(distances))
    return predictionsprint(cluster_list)pred2 = pred_with_dict(X_std, cluster_list)pred2detect_obj.df['cluster_pred'] = pred2np.bincount(pred2)fig, ax = plt.subplots()

sample = detect_obj.df

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=.5)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)

nsrdb_mask = sample['sky_status'] == 1
cluster_mask = (sample['cluster'] == 2) | (sample['cluster'] == 5) #  | (sample['cluster'] == 8) cluster 8 is night time
pred_mask = (sample['cluster_pred'] == 2) | (sample['cluster_pred'] == 5)

# ax.scatter(sample[pred_mask].index, sample[pred_mask]['GHI'], label='Pred only', zorder=100)

ax.scatter(sample[(pred_mask) & (~cluster_mask)].index, sample[(pred_mask) & (~cluster_mask)]['GHI'], label='Pred only', zorder=100)
ax.scatter(sample[~pred_mask & cluster_mask].index, sample[~pred_mask & cluster_mask]['GHI'], label='Cluster only', zorder=100)
ax.scatter(sample[pred_mask & cluster_mask].index, sample[pred_mask & cluster_mask]['GHI'], label='Both', zorder=100)

_ = ax.legend()detect_obj.df['sky_status'] = (detect_obj.df['cluster'] == 2) | (detect_obj.df['cluster'] == 5)clf = ensemble.RandomForestClassifier()

clf = detect_obj.fit_model(clf)test_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.df.index = test_obj.df.index.tz_convert('MST')
test_obj.trim_dates('01-01-2015')pred = test_obj.predict(clf)fig, ax = plt.subplots()

ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha = .5)
ax.scatter(test_obj.df[pred].index, test_obj.df[pred]['GHI'], label='Cluster+RF')
ax.scatter(test_obj.df[test_obj.df['sky_status']].index, test_obj.df[test_obj.df['sky_status']]['GHI'], label='NSRDB', marker='x')

_ = ax.legend()# KMEANSdetect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')
detect_obj.trim_dates('01-01-2010', '01-01-2014')
original_sky_status = detect_obj.df['sky_status']
detect_obj.scale_model()
detect_obj.calc_all_metrics()from sklearn import cluster# km = cluster.KMeans(n_clusters=8)
km = cluster.DBSCAN(eps=.25)
# km = cluster.SpectralClustering()
# km = cluster.AgglomerativeClustering(n_clusters=10)
# km = cluster.Birch(n_clusters=10, threshold=.5, branching_factor=)X = detect_obj.df[detect_obj.features_]from sklearn import preprocessingss = preprocessing.StandardScaler().fit(X)X_std = ss.transform(X)pred = km.fit_predict(X_std)detect_obj.df['cluster'] = predfor clstr, grp in detect_obj.df.groupby(detect_obj.df['cluster']):
    print(clstr)
    print('\tavg GHI/GHIcs mean: ', np.mean(grp['GHI/GHIcs mean']), ' avg GHI/GHIcs std: ', np.mean(grp['GHI/GHIcs std']))
    print('\tavg GHI-GHIcs mean: ', np.mean(grp['GHI-GHIcs mean']), ' avg GHI-GHIcs std: ', np.mean(grp['GHI-GHIcs mean']))
    print('\t', np.sum(grp['sky_status']) / len(grp))fig, ax = plt.subplots()

sample = detect_obj.df

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=.5)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)

nsrdb_mask = sample['sky_status'] == 1
cluster_mask = (sample['cluster'] == 53) # | (sample['cluster'] == 3)#   | (sample['cluster'] == 4) # cluster 8 is night time

ax.scatter(sample[(nsrdb_mask) & (~cluster_mask)].index, sample[(nsrdb_mask) & (~cluster_mask)]['GHI'], label='NSRDB only', zorder=100)
ax.scatter(sample[~nsrdb_mask & cluster_mask].index, sample[~nsrdb_mask & cluster_mask]['GHI'], label='Cluster only', zorder=100)
ax.scatter(sample[nsrdb_mask & cluster_mask].index, sample[nsrdb_mask & cluster_mask]['GHI'], label='Both', zorder=100)

_ = ax.legend()Clustering looks like the best and most consistent model yet.  NSRDB points generally look "wrong".  There are periods though where NSRDB labels points as clear "on the edge" of a large peak or valley.  The clustering-only points look to correct many 'mistakes' in the NSRDB labeling.  The number of clusters and how to choose them will be further investigated, but for this experiment two clusters were chosen out of a possible 10.  These clusters were chosen based on:

1. High average GHI/GHIcs (paired with low average GHI/GHIcs standard deviation).
2. Low average GHI-GHIcs (and associated standard deviation).

Cluster 1 captures most points.  Using the GHI/GHIcs ratio generally works for identifying periods of clarity.  It struggles when both GHI and GHIcs are very small, so small deviations lead to large ratios.  Cluster 2 captures these points by using the GHI-GHIcs criteria.
# 
