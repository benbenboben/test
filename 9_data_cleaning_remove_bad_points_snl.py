
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Investigate-input-data" data-toc-modified-id="Investigate-input-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Investigate input data</a></div><div class="lev2 toc-item"><a href="#ABQ" data-toc-modified-id="ABQ-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>ABQ</a></div><div class="lev3 toc-item"><a href="#Ratio" data-toc-modified-id="Ratio-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Ratio</a></div><div class="lev3 toc-item"><a href="#Difference" data-toc-modified-id="Difference-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Difference</a></div><div class="lev1 toc-item"><a href="#Train-on-default-data" data-toc-modified-id="Train-on-default-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train on default data</a></div><div class="lev1 toc-item"><a href="#Train-on-'cleaned'-data" data-toc-modified-id="Train-on-'cleaned'-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on 'cleaned' data</a></div><div class="lev1 toc-item"><a href="#Advanced-scoring" data-toc-modified-id="Advanced-scoring-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Advanced scoring</a></div><div class="lev2 toc-item"><a href="#set-up-model-trained-on-default-data-(scaled-only)" data-toc-modified-id="set-up-model-trained-on-default-data-(scaled-only)-41"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>set up model trained on default data (scaled only)</a></div><div class="lev2 toc-item"><a href="#set-up-model-trained-on-cleaned-data-(scaled-+-cutoffs-for-metrics)" data-toc-modified-id="set-up-model-trained-on-cleaned-data-(scaled-+-cutoffs-for-metrics)-42"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>set up model trained on cleaned data (scaled + cutoffs for metrics)</a></div><div class="lev2 toc-item"><a href="#scores" data-toc-modified-id="scores-43"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>scores</a></div><div class="lev3 toc-item"><a href="#Default-training,-default-testing" data-toc-modified-id="Default-training,-default-testing-431"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Default training, default testing</a></div><div class="lev3 toc-item"><a href="#Default-training,-cleaned-testing" data-toc-modified-id="Default-training,-cleaned-testing-432"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Default training, cleaned testing</a></div><div class="lev3 toc-item"><a href="#Cleaned-training,-default-testing" data-toc-modified-id="Cleaned-training,-default-testing-433"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>Cleaned training, default testing</a></div><div class="lev3 toc-item"><a href="#Cleaned-training,-cleaned-testing" data-toc-modified-id="Cleaned-training,-cleaned-testing-434"><span class="toc-item-num">4.3.4&nbsp;&nbsp;</span>Cleaned training, cleaned testing</a></div><div class="lev1 toc-item"><a href="#CV-scoring-and-model-selection-(2010-2015-only)" data-toc-modified-id="CV-scoring-and-model-selection-(2010-2015-only)-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>CV scoring and model selection (2010-2015 only)</a></div><div class="lev2 toc-item"><a href="#default" data-toc-modified-id="default-51"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>default</a></div><div class="lev2 toc-item"><a href="#cleand-(very-lax)" data-toc-modified-id="cleand-(very-lax)-52"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>cleand (very lax)</a></div><div class="lev2 toc-item"><a href="#cleaned-(lax)" data-toc-modified-id="cleaned-(lax)-53"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>cleaned (lax)</a></div><div class="lev2 toc-item"><a href="#cleaned" data-toc-modified-id="cleaned-54"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>cleaned</a></div><div class="lev2 toc-item"><a href="#cleaned-(aggressive)" data-toc-modified-id="cleaned-(aggressive)-55"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>cleaned (aggressive)</a></div><div class="lev2 toc-item"><a href="#cleaned-(very-aggressive)" data-toc-modified-id="cleaned-(very-aggressive)-56"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>cleaned (very aggressive)</a></div>

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
import cs_detection
import utils
import visualize_plotly as visualize
sns.set_style("white")

matplotlib.rcParams['figure.figsize'] = (20., 8.)

from IPython.display import Image

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()
init_notebook_mode(connected=True)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(precision=4)
get_ipython().magic('matplotlib inline')

get_ipython().magic("config InlineBackend.figure_format = 'retina'")

matplotlib.rcParams.update({'font.size': 16})

import warnings
warnings.filterwarnings(action='ignore')


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

# In[3]:


# filter off night times, will skew data
nsrdb_nonight = nsrdb[nsrdb['Clearsky GHI pvlib'] > 0]


# In[4]:


clear_pds = nsrdb_nonight[nsrdb_nonight['sky_status']]
cloudy_pds = nsrdb_nonight[~nsrdb_nonight['sky_status']]


# In[5]:


print(np.mean(clear_pds['GHI/GHIcs mean']), np.std(clear_pds['GHI/GHIcs mean']), np.median(clear_pds['GHI/GHIcs mean']))
print(np.mean(cloudy_pds['GHI/GHIcs mean']), np.std(cloudy_pds['GHI/GHIcs mean']), np.median(cloudy_pds['GHI/GHIcs mean']))


# In[6]:


fig, ax = plt.subplots()

sns.boxplot(data=[cloudy_pds['GHI/GHIcs mean'], clear_pds['GHI/GHIcs mean']], ax=ax)
_ = ax.set_ylabel('GHI / GHI$_\mathrm{CS}$ mean')
_ = ax.set_xticklabels(['cloudy', 'clear'])
_ = ax.set_xlabel('NSRDB labels')


# In[7]:


cloudy_pds['GHI/GHIcs mean'].describe()


# In[8]:


clear_pds['GHI/GHIcs mean'].describe()


# Clear periods have relatively tight distribution near a ratio of 1.  Some obvious low outliers need to be fixed/removed.  Cloudy periods show much more spread which is expected.  The mean and median of cloudy periods is quite a bit lower than clear periods  (mean and median of clear ~.95, cloudy mean and median is .66 and .74).  Based on the whiskers there might be mislabeled cloudy points, though these points could also have ratios near 1 but be in noisy periods.

# In[9]:


fig, ax = plt.subplots()

bins = np.linspace(0, 1.2, 25)
ax.hist(clear_pds['GHI/GHIcs mean'], bins, label='clear', color='C1', alpha=.75)
ax.hist(cloudy_pds['GHI/GHIcs mean'], bins, label='cloudy', color='C0', alpha=.75)
ax.set_xlabel('GHI / GHI$_\mathrm{CS}$ mean')
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

# In[10]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI/GHIcs mean'])
    cloudy_list.append(cloudy_pds['GHI/GHIcs mean'])
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

# In[11]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year, nsrdb_nonight.index.month // 3]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI/GHIcs mean'])
    cloudy_list.append(cloudy_pds['GHI/GHIcs mean'])
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

# In[12]:


clear_list, cloudy_list, tfn_list = [], [], []
for tfn, grp in nsrdb_nonight.groupby([nsrdb_nonight['abs(t-tnoon)']]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI/GHIcs mean'])
    cloudy_list.append(cloudy_pds['GHI/GHIcs mean'])
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

# In[13]:


sample = nsrdb[nsrdb.index >= '01-01-2012']


# In[14]:


fig, ax = plt.subplots(figsize=(16, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
a = ax.scatter(sample[sample['sky_status']].index, sample[sample['sky_status']]['GHI'], 
               c=sample[sample['sky_status']]['GHI/GHIcs mean'], label=None, zorder=10, cmap='seismic', s=10)

_ = ax.legend(loc='upper right')

_ = ax.set_title('NSRDB Clear periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI / GHI$_{\mathrm{CS}}$')


# For points labeled as clear by NSRDB, the early morning periods have a noticably lower ratio.  This is visual confirmation of the previous plot.  

# In[15]:


fig, ax = plt.subplots(figsize=(16, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=0.5)
a = ax.scatter(sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0].index, 
               sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI'], 
               c=sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI/GHIcs mean'], zorder=10, cmap='seismic', s=10, label=None)

_ = ax.legend(loc='upper right')

_ = ax.set_title('NSRDB cloudy periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI / GHI$_{\mathrm{CS}}$')


# In[16]:


fig, ax = plt.subplots(figsize=(16, 8))

subsample = sample[(sample.index >= '03-22-2012') & (sample.index < '03-26-2012')]

_ = ax.plot(subsample.index, subsample['GHI'], label='GHI')
_ = ax.plot(subsample.index, subsample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)

a = ax.scatter(subsample[(~subsample['sky_status']) & (subsample['Clearsky GHI pvlib'] > 0)].index, 
               subsample[(~subsample['sky_status']) & (subsample['Clearsky GHI pvlib'] > 0)]['GHI'], 
               c=subsample[(~subsample['sky_status']) & (subsample['Clearsky GHI pvlib'] > 0)]['GHI/GHIcs mean'], zorder=10, cmap='seismic', s=40, label=None)

_ = ax.legend(bbox_to_anchor=(.85, .85))

_ = ax.set_title('NSRDB cloudy periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI / GHI$_{\mathrm{CS}}$')


# There appear to be many clear points that are labeled as cloudy.  There are many cases where there is a ratio near 1 during a 'noisy' period, which should not be labeled clear.  We can't assume ratio is good enough.  This seems to be a much bigger problem than clear points having a low ratio.

# In[17]:


num_clear_good_ratio = len(nsrdb[(nsrdb['sky_status']) & (nsrdb['GHI/GHIcs mean'] >= .95) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_clear_bad_ratio = len(nsrdb[(nsrdb['sky_status']) & (nsrdb['GHI/GHIcs mean'] < .95) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_good_ratio = len(nsrdb[(~nsrdb['sky_status']) & (nsrdb['GHI/GHIcs mean'] >= .95) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_bad_ratio = len(nsrdb[(~nsrdb['sky_status']) & (nsrdb['GHI/GHIcs mean'] < .95) & (nsrdb['Clearsky GHI pvlib'] > 0)])


# In[18]:


print('Clear periods, good ratio: {}'.format(num_clear_good_ratio))
print('Clear periods, bad ratio: {}'.format(num_clear_bad_ratio))
print()
print('Cloudy periods, good ratio: {}'.format(num_cloudy_good_ratio))
print('Cloudy periods, good ratio: {}'.format(num_cloudy_bad_ratio))


# In[19]:


tmp = nsrdb[(nsrdb['sky_status']) & (nsrdb['GHI/GHIcs'] < .95)]

fig, ax = plt.subplots()

# bins = np.linspace(0, 1.2, 25)
ax.hist(tmp['abs(t-tnoon)'], label='clear', color='C1', alpha=.75)
# ax.hist(cloudy_pds['GHI/GHIcs'], bins, label='cloudy', color='C0', alpha=.75)
ax.set_xlabel('|t - t$_\mathrm{noon}$|')
ax.set_xlim((0, 450))
# _ = ax.legend()
_ = ax.set_ylabel('Frequency')
_ = ax.set_title('Clear with ratio < 0.95')


# For NSRDB labeled clear periods:
# * We see that clear periods with a good ratio (>= .9) has 81,000+ samples to 1,100+ samples with a bad ratio (< .9).  It's important to note that just be cause a point has a good ratio, does not necessarily clear.  This does indicate though that an overwhelming majority of clear periods are most likely labeled correctly.
# 
# For NSRDB labeled cloudy periods:
# * Cloudy periods have a much more 'even' ratio count.  There are 20,000+ cloudy points with a good ratio (>= .9) and 56,000+ cloudy points with a bad ratio (< .9).  From visual inspection, it appears that cloudy labels are much less reliable than clear labels provided by NSRDB.
# 
# It would be wise to score classifiers based on their recall score, as the vast majority clear labels from NSRDB appear to be correct (based on the $\mathrm{GHI/GHI_{CS}}$ ratio).  The no special method was used to choose the good/bad ratio cutoff of .9.

# ### Difference

# In[20]:


# filter off night times, will skew data
nsrdb_nonight = nsrdb[nsrdb['Clearsky GHI pvlib'] > 0]


# In[21]:


clear_pds = nsrdb_nonight[nsrdb_nonight['sky_status']]
cloudy_pds = nsrdb_nonight[~nsrdb_nonight['sky_status']]


# In[22]:


print(np.mean(clear_pds['GHI-GHIcs mean']), np.std(clear_pds['GHI-GHIcs mean']), np.median(clear_pds['GHI-GHIcs mean']))
print(np.mean(cloudy_pds['GHI-GHIcs mean']), np.std(cloudy_pds['GHI-GHIcs mean']), np.median(cloudy_pds['GHI-GHIcs mean']))

fig, ax = plt.subplots()

_ = ax.boxplot([clear_pds['ghi/ghics ratio'], cloudy_pds['ghi/ghics ratio']], showmeans=True, labels=['clear', 'cloudy'])
_ = ax.set_ylabel('GHI / GHIcs')
# In[23]:


fig, ax = plt.subplots()

sns.boxplot(data=[cloudy_pds['GHI-GHIcs mean'], clear_pds['GHI-GHIcs mean']], ax=ax)
_ = ax.set_ylabel('GHI - GHI$_\mathrm{CS}$ mean')
_ = ax.set_xticklabels(['cloudy', 'clear'])
_ = ax.set_xlabel('NSRDB label')


# In[24]:


pd.DataFrame(cloudy_pds['GHI-GHIcs mean'].describe())


# In[25]:


clear_pds['GHI-GHIcs mean'].describe()


# Clear periods have relatively tight distribution near a ratio of 1.  Some obvious low outliers need to be fixed/removed.  Cloudy periods show much more spread which is expected.  The mean and median of cloudy periods is quite a bit lower than clear periods  (mean and median of clear ~.95, cloudy mean and median is .66 and .74).  Based on the whiskers there might be mislabeled cloudy points, though these points could also have ratios near 1 but be in noisy periods.

# In[26]:


fig, ax = plt.subplots()

# sns.distplot(clear_pds['GHI/GHIcs'], label='clear', ax=ax, color='C1')
# sns.distplot(cloudy_pds['GHI/GHIcs'], label='cloudy', ax=ax, color='C0')
# bins = np.linspace(0, 1.2, 25)
ax.hist(clear_pds['GHI-GHIcs mean'], label='clear', color='C1', alpha=.75)
ax.hist(cloudy_pds['GHI-GHIcs mean'], label='cloudy', color='C0', alpha=.75)
ax.set_xlabel('GHI - GHI$_\mathrm{CS}$ mean')
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

# In[27]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI-GHIcs mean'])
    cloudy_list.append(cloudy_pds['GHI-GHIcs mean'])
    year_list.append(year)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
# _ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1], color='C0')
_ = ax[1].set_xticklabels(year_list, rotation=45)
_ = ax[1].set_title('Cloudy')
# _ = ax[1].set_ylim(0, 1.2)

ax[0].set_ylabel('GHI - GHI$_\mathrm{CS}$ mean')
ax[1].set_ylabel('GHI - GHI$_\mathrm{CS}$ mean')

_ = fig.tight_layout()


# Visually it appears that each year behaves similarly for both cloudy and clear data.

# In[28]:


clear_list, cloudy_list, year_list = [], [], []
for year, grp in nsrdb_nonight.groupby([nsrdb_nonight.index.year, nsrdb_nonight.index.month // 3]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI-GHIcs mean'])
    cloudy_list.append(cloudy_pds['GHI-GHIcs mean'])
    year_list.append(year)
    
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(16, 8))

sns.boxplot(data=clear_list, ax=ax[0], color='C1')
_ = ax[0].set_title('Clear')
# _ = ax[0].set_ylim(0, 1.2)
sns.boxplot(data=cloudy_list, ax=ax[1],  color='C0')
_ = ax[1].set_xticklabels(year_list, rotation=90)
_ = ax[1].set_title('Cloudy')
# _ = ax[1].set_ylim(0, 1.2)

ax[0].set_ylabel('GHI - GHI$_\mathrm{CS}$ mean')
ax[1].set_ylabel('GHI - GHI$_\mathrm{CS}$ mean')
    
_ = fig.tight_layout()


# A finer view of the year-on-year behavior doesn't show any striking seasonal trends.  During cloudy periods, we do notice that as time goes on, there are less 'outlier' points on the low end of the ratios.

# In[29]:


clear_list, cloudy_list, tfn_list = [], [], []
for tfn, grp in nsrdb_nonight.groupby([nsrdb_nonight['abs(t-tnoon)']]):
    clear_pds = grp[grp['sky_status']]
    cloudy_pds = grp[~grp['sky_status']]
    clear_list.append(clear_pds['GHI-GHIcs mean'])
    cloudy_list.append(cloudy_pds['GHI-GHIcs mean'])
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
_ = ax[0].set_ylabel('GHI - GHI$_{\mathrm{CS}}$ mean')
_ = ax[1].set_ylabel('GHI - GHI$_{\mathrm{CS}}$ mean')


_ = fig.tight_layout()


# As we move away from solar noon (defined as the point of maximum model irradiance), the clear labeled periods include more and more outliers with respect to the GHI/GHIcs ratio.

# In[30]:


sample = nsrdb[nsrdb.index >= '01-01-2012']
sample = nsrdb[(nsrdb.index >= '03-01-2012') & (nsrdb.index < '03-15-2012')]


# In[31]:


fig, ax = plt.subplots(figsize=(24, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
a = ax.scatter(sample[sample['sky_status']].index, sample[sample['sky_status']]['GHI'], 
               c=sample[sample['sky_status']]['GHI-GHIcs mean'], label=None, zorder=10, cmap='Reds', s=80)

_ = ax.legend(bbox_to_anchor=(1.25, .98))

_ = ax.set_title('NSRDB Clear periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI - GHI$_{\mathrm{CS}}$')


# For points labeled as clear by NSRDB, the early morning periods have a noticably lower ratio.  This is visual confirmation of the previous plot.  

# In[32]:


sample = nsrdb[(nsrdb.index >= '03-01-2012') & (nsrdb.index < '03-15-2012')]


# In[33]:


fig, ax = plt.subplots(figsize=(24, 8))

_ = ax.plot(sample.index, sample['GHI'], label='GHI')
_ = ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)
a = ax.scatter(sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0].index, 
               sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI'], 
               c=sample[~sample['sky_status'] & sample['Clearsky GHI pvlib'] > 0]['GHI-GHIcs mean'], zorder=10, cmap='Reds', s=80, label=None)

_ = ax.legend(bbox_to_anchor=(1.25, .98))

_ = ax.set_title('NSRDB cloudy periods')
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

cb = fig.colorbar(a)
cb.set_label('GHI - GHI$_{\mathrm{CS}}$')


# In[34]:


num_clear_good_diff = len(nsrdb[(nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs mean']) <= 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_clear_bad_diff = len(nsrdb[(nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs mean']) > 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_good_diff = len(nsrdb[(~nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs mean']) <= 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])
num_cloudy_bad_diff = len(nsrdb[(~nsrdb['sky_status']) & (np.abs(nsrdb['GHI-GHIcs mean']) > 50) & (nsrdb['Clearsky GHI pvlib'] > 0)])


# In[35]:


print('Clear periods, good diff: {}'.format(num_clear_good_diff))
print('Clear periods, bad diff: {}'.format(num_clear_bad_diff))
print()
print('Cloudy periods, good diff: {}'.format(num_cloudy_good_diff))
print('Cloudy periods, good diff: {}'.format(num_cloudy_bad_diff))


# There appear to be many clear points that are labeled as cloudy.  There are many cases where there is a ratio near 1 during a 'noisy' period, which should not be labeled clear.  We can't assume ratio is good enough.  This seems to be a much bigger problem than clear points having a low ratio.

# # Train on default data

# In[36]:


# nsrdb = pd.read_pickle('abq_nsrdb_1.pkl.gz')
detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')


# In[37]:


train_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates(None, '01-01-2015')
test_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)


# In[38]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[39]:


clf = train_obj.fit_model(clf)


# In[40]:


pred = test_obj.predict(clf)


# In[41]:


print(metrics.accuracy_score(test_obj.df['sky_status'], pred))


# In[42]:


print(metrics.recall_score(test_obj.df['sky_status'], pred))


# In[43]:


cm = metrics.confusion_matrix(test_obj.df['sky_status'], pred)


# In[44]:


visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))


# In[45]:


fig, ax = plt.subplots(figsize=(12, 8))

_ = ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
_ = ax.set_xticks(range(len(clf.feature_importances_)))
_ = ax.set_xticklabels(test_obj.features_, rotation=45)

_ = ax.set_ylabel('Importance')
_ = ax.set_xlabel('Feature')

_ = fig.tight_layout()


# In[46]:


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

# In[47]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')


# In[48]:


detect_obj.filter_labels()

fig, ax = plt.subplots(figsize=(24, 8))

sample = detect_obj.df[detect_obj.df.index.year == 2014]

sample = detect_obj.df[(detect_obj.df.index.year == 2014) & (detect_obj.df.index.week == 10)]

ax.plot(sample.index, sample['GHI'], label='GHI', alpha=1)
ax.plot(sample.index, sample['Clearsky GHI pvlib'], label='GHIcs', alpha=1)
ax.scatter(sample[sample['mask']].index, sample[sample['mask']]['GHI'], label='cleaned', s=80)
ax.scatter(sample[sample['sky_status']].index, sample[sample['sky_status']]['GHI'], label='clear', marker='x', s=80, color='k')
_ = ax.legend(bbox_to_anchor=(1.15, 1))
_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')

ax2 = ax.twinx()
ax2.plot(sample['GHI/GHIcs mean'].index, sample['GHI/GHIcs mean'], color='C3', label='ratio')
ax2.plot(sample['GHI-GHIcs mean'].index, np.abs(sample['GHI-GHIcs mean']) / 50, color='C4', label='diff')
ax2.axhline(0.95, color='k', linestyle='--')
_ = ax2.legend(bbox_to_anchor=(1.12, .78))
# In[49]:


train_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates(None, '01-01-2015')
test_obj = cs_detection.ClearskyDetection(detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)


# In[50]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[51]:


clf = train_obj.fit_model(clf, ratio_mean_val=0.95, diff_mean_val=50)


# In[52]:


pred = test_obj.predict(clf)


# In[53]:


test_obj.filter_labels(ratio_mean_val=0.95, diff_mean_val=50)


# In[54]:


print(metrics.accuracy_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))


# In[55]:


print(metrics.recall_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))


# In[56]:


print(metrics.recall_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))


# In[57]:


print(metrics.accuracy_score(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']]))


# In[58]:


cm = metrics.confusion_matrix(test_obj.df[test_obj.df['mask']]['sky_status'], pred[test_obj.df['mask']])


# In[59]:


visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'))


# In[60]:


fig, ax = plt.subplots(figsize=(12, 8))

_ = ax.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
_ = ax.set_xticks(range(len(clf.feature_importances_)))
_ = ax.set_xticklabels(test_obj.features_, rotation=45)

_ = ax.set_ylabel('Importance')
_ = ax.set_xlabel('Feature')

_ = fig.tight_layout()


# In[61]:


fig, ax = plt.subplots(figsize=(24, 8))

nsrdb_mask = test_obj.df['sky_status'].values
ax.plot(test_obj.df.index, test_obj.df['GHI'], label='GHI', alpha=.5)
ax.plot(test_obj.df.index, test_obj.df['Clearsky GHI pvlib'], label='GHIcs', alpha=.5)
ax.scatter(test_obj.df[nsrdb_mask & ~pred].index, test_obj.df[nsrdb_mask & ~pred]['GHI'], label='NSRDB only', zorder=10, s=50)
ax.scatter(test_obj.df[pred & ~nsrdb_mask].index, test_obj.df[pred & ~nsrdb_mask]['GHI'], label='RF only', zorder=10, s=50)
ax.scatter(test_obj.df[nsrdb_mask & pred].index, test_obj.df[nsrdb_mask & pred]['GHI'], label='Both', zorder=10, s=50)
_ = ax.legend(bbox_to_anchor=(1.25, 1))

_ = ax.set_xlabel('Date')
_ = ax.set_ylabel('GHI / Wm$^{-2}$')


# In[62]:


# fig, ax = plt.subplots(figsize=(12, 8))

nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers')
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers')
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers')
# _ = ax.legend(bbox_to_anchor=(1.25, 1))

# _ = ax.set_xlabel('Date')
# _ = ax.set_ylabel('GHI / Wm$^{-2}$')
iplot([trace1, trace2, trace3, trace4, trace5])


# In[63]:


print(len(test_obj.df[nsrdb_mask & ~pred]))


# Thus far, we have trained on default and cleaned data.  When scoring these methods, we have not cleaned the testing set.  This needs to be done to provide a fair comparison between cleaned and default data sets.  We will also score between default-trained/cleaned-testing and cleaned-trained/default-testing data sets.

# # Advanced scoring

# ## set up model trained on default data (scaled only)

# In[64]:


dflt_detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
dflt_detect_obj.df.index = dflt_detect_obj.df.index.tz_convert('MST')


# In[65]:


dflt_train_obj = cs_detection.ClearskyDetection(dflt_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
dflt_train_obj.trim_dates(None, '01-01-2015')
dflt_test_obj = cs_detection.ClearskyDetection(dflt_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
dflt_test_obj.trim_dates('01-01-2015', None)


# In[66]:


dflt_model = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[67]:


dflt_model = dflt_train_obj.fit_model(dflt_model)


# In[68]:


dflt_pred = dflt_test_obj.predict(dflt_model)


# In[69]:


np.bincount(dflt_test_obj.df['mask'])


# ## set up model trained on cleaned data (scaled + cutoffs for metrics)

# In[70]:


clean_detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
clean_detect_obj.df.index = clean_detect_obj.df.index.tz_convert('MST')


# In[71]:


clean_train_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
clean_train_obj.trim_dates(None, '01-01-2015')
clean_test_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
clean_test_obj.trim_dates('01-01-2015', None)


# In[72]:


clean_model = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[73]:


clean_model = clean_train_obj.fit_model(clean_model, ratio_mean_val=0.95, diff_mean_val=50)


# In[74]:


clean_pred = clean_test_obj.predict(clean_model)


# In[75]:


clean_test_obj.filter_labels(ratio_mean_val=0.95, diff_mean_val=50)


# In[76]:


np.bincount(clean_test_obj.df['mask'])


# ## scores

# ### Default training, default testing

# In[77]:


true = dflt_test_obj.df['sky_status']
pred = dflt_pred


# In[78]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[79]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[80]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[81]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[82]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on default trained and default scored')


# ### Default training, cleaned testing

# In[83]:


true = clean_test_obj.df[clean_test_obj.df['mask']]['sky_status']
pred = dflt_pred[clean_test_obj.df['mask']]


# In[84]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[85]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[86]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[87]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[88]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on default trained and cleaned scored')


# ### Cleaned training, default testing

# In[89]:


true = dflt_test_obj.df['sky_status']
pred = clean_pred


# In[90]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[91]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[92]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[93]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[94]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and default scored')


# ### Cleaned training, cleaned testing

# In[95]:


true = clean_test_obj.df[clean_test_obj.df['mask']]['sky_status']
pred = clean_pred[clean_test_obj.df['mask']]


# In[96]:


print('accuracy: {}'.format(metrics.accuracy_score(true, pred)))


# In[97]:


print('precision: {}'.format(metrics.precision_score(true, pred)))


# In[98]:


print('recall: {}'.format(metrics.recall_score(true, pred)))


# In[99]:


print('f1: {}'.format(metrics.f1_score(true, pred)))


# In[100]:


cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# # CV scoring and model selection (2010-2015 only)

# In[101]:


detect_obj = cs_detection.ClearskyDetection.read_pickle('abq_nsrdb_1.pkl.gz', 'GHI', 'Clearsky GHI pvlib', 'sky_status')
detect_obj.df.index = detect_obj.df.index.tz_convert('MST')

train_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
train_obj.trim_dates('01-01-2010', '01-01-2015')
test_obj = cs_detection.ClearskyDetection(clean_detect_obj.df, 'GHI', 'Clearsky GHI pvlib', 'sky_status')
test_obj.trim_dates('01-01-2015', None)


# ## default

# In[102]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=32, max_depth=10, random_state=42)


# In[103]:


scores = train_obj.cross_val_score(clf, scoring='f1')


# In[104]:


print('{} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[105]:


clf = train_obj.fit_model(clf)


# In[106]:


pred = test_obj.predict(clf)


# In[107]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[108]:


true = test_obj.df['sky_status']
cm = metrics.confusion_matrix(true, pred)
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# ## cleand (very lax)

# In[109]:


scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.90, 'diff_mean_val': 100}, filter_fit=True, filter_score=True)


# In[110]:


print('{} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[111]:


clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.90, 'diff_mean_val': 100})


# In[112]:


pred = test_obj.predict(clf)


# In[113]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[114]:


true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# ## cleaned (lax)

# In[115]:


scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.93, 'diff_mean_val': 70}, filter_fit=True, filter_score=True)


# In[116]:


print('{} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[117]:


clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.93, 'diff_mean_val': 70})


# In[118]:


pred = test_obj.predict(clf)


# In[119]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[120]:


true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# ## cleaned

# In[121]:


scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.95, 'diff_mean_val': 50}, filter_fit=True, filter_score=True)


# In[122]:


print('{} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[123]:


clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.95, 'diff_mean_val': 50})


# In[124]:


pred = test_obj.predict(clf)


# In[125]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[126]:


true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# ## cleaned (aggressive)

# In[127]:


scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.97, 'diff_mean_val': 30}, filter_fit=True, filter_score=True)


# In[128]:


print('{} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[129]:


clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.97, 'diff_mean_val': 30})


# In[130]:


pred = test_obj.predict(clf)


# In[131]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[132]:


true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# ## cleaned (very aggressive)

# In[133]:


scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.99, 'diff_mean_val': 10}, filter_fit=True, filter_score=True)


# In[134]:


print('{} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[135]:


clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.99, 'diff_mean_val': 10})


# In[136]:


pred = test_obj.predict(clf)


# In[137]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[138]:


true = test_obj.df['sky_status'][test_obj.df['mask']]
cm = metrics.confusion_matrix(true, pred[test_obj.df['mask']])
visualize.plot_confusion_matrix2(cm, ('cloudy', 'clear'), title='RF performance on clean trained and clean scored')


# In[139]:



for ml in [10, 20, 30]:
    for nest in [32]:
        clf = ensemble.RandomForestClassifier(n_jobs=-1, min_samples_leaf=ml, n_estimators=nest)
        scores = train_obj.cross_val_score(clf, scoring='f1', filter_kwargs={'ratio_mean_val': 0.99, 'diff_mean_val': 10}, filter_fit=True, filter_score=True)
        print('n_estimators: {}, min_samples_leaf: {}'.format(nest, ml))
        print('    {} +/- {}'.format(np.mean(scores), np.std(scores)))


# In[140]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, min_samples_leaf=20, n_estimators=32)
clf = train_obj.fit_model(clf, **{'ratio_mean_val': 0.99, 'diff_mean_val': 10})


# In[141]:


pred = test_obj.predict(clf)


# In[142]:


nsrdb_mask = test_obj.df['sky_status'].values

trace1 = go.Scatter(x=test_obj.df.index, y=test_obj.df['GHI'], name='GHI')
trace2 = go.Scatter(x=test_obj.df.index, y=test_obj.df['Clearsky GHI pvlib'], name='GHIcs')
trace3 = go.Scatter(x=test_obj.df[nsrdb_mask & ~pred].index, y=test_obj.df[nsrdb_mask & ~pred]['GHI'], name='NSRDB only', mode='markers', marker={'size': 10})
trace4 = go.Scatter(x=test_obj.df[pred & ~nsrdb_mask].index, y=test_obj.df[pred & ~nsrdb_mask]['GHI'], name='RF only', mode='markers', marker={'size': 10})
trace5 = go.Scatter(x=test_obj.df[nsrdb_mask & pred].index, y=test_obj.df[nsrdb_mask & pred]['GHI'], name='Both', mode='markers', marker={'size': 10})

iplot([trace1, trace2, trace3, trace4, trace5])


# In[ ]:




