#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
customers = pd.read_csv('customer_data.csv')
customers.head()


# In[3]:


points =customers.iloc[:, 3:5].values
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y, s=50, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')


# In[4]:


kmeans = KMeans(n_clusters=6,

random_state=0)
kmeans.fit(points)

predicted_cluster_indexes = kmeans.predict(points)
print(predicted_cluster_indexes)
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')
plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100)


# In[7]:


from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
text1 = "The data set given satisfies the requirement for model generation. This is used in Data Science Lab"
nltk.download('punkt')
print(sent_tokenize(text1))


# In[8]:


print(word_tokenize(text1))


# In[9]:


nltk.download('stopwords')
print(stopwords.words('english'))


# In[10]:


text = word_tokenize(text1)
text= [word for word in text if word not in stopwords.words('english')]
print(text)


# In[16]:


nltk.download('averaged_perceptron_tagger')
print(nltk.pos_tag(text))


# In[17]:


temp=zip(*[text[i:] for i in range(0,2)])
ans=[' '.join(ngram) for ngram in temp]
print(ans)


# In[18]:


temp=zip(*[text[i:] for i in range(0,4)])
ans=[' '.join(ngram) for ngram in temp]
print(ans)


# In[ ]:




