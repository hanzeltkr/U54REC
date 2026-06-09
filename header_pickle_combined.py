#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from pathlib import Path


# In[3]:


# Pickle file for 2d mapping
pkl2d = Path("release_to_header_mapping.pkl")


# In[4]:


mapping2d = pickle.load(open(pkl2d, "rb"))


# In[6]:


len(mapping2d)


# In[8]:


mapping2d["100000039159562031368112550794429920461"]


# In[10]:


ds1 = mapping2d["100000039159562031368112550794429920461"]
print(ds1.PatientID)
print(ds1.ExposureTime)


# In[ ]:


# Pickle file for 3d mapping
pkl3d = Path("release_to_header_mapping_3d.pkl")


# In[ ]:


mapping3d = pickle.load(open(pkl3d, "rb"))


# In[ ]:


len(mapping3d)


# In[ ]:


mapping3d["2.25.100003998065499037342635672099403856296"]


# In[ ]:


ds2 = mapping3d["2.25.100003998065499037342635672099403856296"]
print(ds2.PatientID)
print(ds2.Manufacturer)

