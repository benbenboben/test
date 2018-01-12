
# coding: utf-8

# # Table of Contents
#  <p>

# In[14]:


import nsrdb_api
import pandas as pd
import numpy as np
import os


# In[15]:


city_data = pd.read_json('./cities.json')


# In[16]:


city_data


# In[17]:


lat_lon = (city_data['latitude'].astype(str) + ' '  + city_data['longitude'].astype(str))


# In[18]:


strings = []
for lat, lon in zip(city_data['latitude'], city_data['longitude']):
    strings.append(str(np.round(lon, 2)) + ' ' + str(np.round(lat, 2)))


# In[19]:


caller = nsrdb_call.NSRDBAPI(os.environ.get('NSRDB_API_KEY'), strings[:500])


# In[20]:


result = caller.site_count_call()


# In[21]:


result['outputs']['nsrdb_site_count']


# In[25]:


caller = nsrdb_api.NSRDPAPI(os.environ.get('NSRDB_API_KEY'), strings[:100], email='bhellis@lbl.gov', mailing_list=False, 
                                affiliation='LBL', full_name='ben', reason='research', leap_day=False, interval=30,
                                names=(1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015),
                                attributes=('ghi', 'clearsky_ghi', 'cloud_type', 'fill_flag'))


# In[28]:


caller.data_call()


# In[ ]:




