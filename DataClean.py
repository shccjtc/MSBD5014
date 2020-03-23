#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import gc


# In[229]:


train_data = pd.read_csv("./data/digsci2019 data/train_release.csv")
paper_data = pd.read_csv("./data/digsci2019 data/candidate_paper.csv")


# In[230]:


df = train_data.merge(paper_data,on='paper_id',how='left')[['description_text','abstract','title']]
df = df.dropna()


# In[231]:


df['description'] = df['description_text'] + '****####' + df['title']


# In[232]:


df['description'][0]


# In[233]:


def replaceCandidate(x):
    s = x.split('****####')
    return s[0].replace('[**##**]',s[1])


# In[234]:


df['description'] = df['description'].map(lambda x:replaceCandidate(str(x)))


# In[235]:


df2 = df.drop(['description_text','title'],axis=1)
df2['label'] = 1


# In[236]:


df2['description'][50]


# In[237]:


df3 = df2.sample(n=20001)


# In[238]:


df3['description'] = df3['description'].shift(1)


# In[239]:


df3 = df3.dropna()
df3['label'] = 0


# In[ ]:





# In[ ]:





# In[240]:


df4 = pd.concat([df2,df3],axis=0).sample(frac=1).reset_index(drop=True)


# In[245]:


df4.to_csv('./data/syn_sentence_pair.csv',index=False,encoding='utf_8_sig')


# In[246]:





# In[247]:





# In[ ]:




