#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from pathlib import Path


# In[5]:


pkl = Path("release_to_header_mapping_3d.pkl")


# In[ ]:


mapping = pickle.load(open(pkl, "rb"))


# In[ ]:


len(mapping)


# In[ ]:


size_mb = pkl.stat().st_size / (1024 * 1024)
print(f"{size_mb:.2f} MB")


# In[6]:


mapping["2.25.100003998065499037342635672099403856296"]


# In[7]:


ds = mapping["2.25.100003998065499037342635672099403856296"]


# In[8]:


print(f"Total DICOM elements: {len(ds)}")


# In[9]:


print(ds.PatientID)


# In[11]:


print(ds.PixelPresentation)


# In[ ]:




