#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing, Standardization, and K-means clustering
# 
# We take our cleaned and formatted data and perform a number of steps to the data to preprocess and standardize before it is ready to use with our K-Means algorithm

# ## Import libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ## Import Data

# In[2]:


clean_data = pd.read_csv('Data/cleaned_data.csv')


# # Preprocessing
# 
# ## Label Encoding/One Hot Encoding
# 
# We need to have our data as numerical for the k-means clustering algorithm. Categorical data such as the column for "Study" pose a problem as the k-means algorithm cannot assign an appropriate value to a string. Two techniques that can solve this problem are label encoding and one hot encoding. Label encoding assigns a numerical value but this does not work for our dataset as it will assign values (1, 2, 3, etc.) which may imply a hierarchy that is not present in our data between each Study. We therefore explore the second option of one hot encoding which creates a new column for each of the 10 studies and then assigns the value 0 if it is not from this study or a 1 if it is. One hot encoding can cause problems when there are many categories but in this case we deemed it appropriate for use. 
# 
# We will use the label encoding technique to convert categorical variables to numbers and then use the one hot encoding toolkit from scikit-learn to convert these into binary values. Scikit-learn has a useful preprocessing module for this

# ### Label Encoder

# In[4]:


# create instance of labelencoder
labelencoder = LabelEncoder()
# Assign numerical values and store in a new column
clean_data['Study_no'] = labelencoder.fit_transform(clean_data['Study'])


# Display this new column

# In[6]:


clean_data


# ### One Hot Encoding

# In[7]:


# create instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# passing on Study_no column values
enc_df = pd.DataFrame(enc.fit_transform(clean_data[['Study_no']]).toarray())

# rename columns as Study names
enc_df = enc_df.rename(columns={0: 'Fridberg', 1: 'Horstmann', 2: 'Kjome', 3: 'Maia', 6: 'SteingroverInPrep', 4: 'Premkumar', 8: 'Wood', 9: 'Worthy', 5: 'Steingroever2011', 7: 'Wetzels'})

# reset index before concatenation
clean_data.reset_index(inplace=True)
enc = pd.concat([clean_data, enc_df], axis=1)
enc


# ## Drop columns

# In[12]:


df = enc.drop(columns=['level_0', 'index', 'Subj', 'Study', 'Study_no', 'Unnamed: 0'])
df


# ## Standardization
# 
# Generally speaking, learning algorithms perform better with standardized data. In our case there is a large variety in the 0 and 1 values for the studies and the total win and total loss values and so we have chosen to standardize the whole data set. Scitkit-learn again offers a way to do this using the preprocessing module.

# In[13]:


scaler = preprocessing.StandardScaler().fit(df)
X_scaled = scaler.transform(df)


# In[20]:


X_scaled


# In[16]:


X_scaled.std(axis=0)


# In[14]:


#rename columns to clearly represent decks
sd = pd.DataFrame(X_scaled, columns=['Deck_1', 'Deck_2', 'Deck_3', 'Deck_4', 'tot_win', 'tot_los', 'Fridberg', 'Horstmann', 'Kjome', 'Maia', 'Premkumar', 'Steingroever2011', 'SteingroverInPrep', 'Wetzels', 'Wood', 'Worthy'])
sd


# ## K- means clustering
# 
# Clustering involves grouping data points into groups that are similar to one another. It is an unsupervised technique using patterns within the data to decide upon the chosen groups. Within k means, the k represents the number of clusters we require. We choose this value for k and so the best way to determine the value of k is to run experiments for different values of k and then see which results in the least error. 
# 
# #### K means steps:
# 
# - The value of k must be decided
# - k random points are selected as the initial centroids
# - Each data point is assigned to a cluster based on their distance from the nearest initial centroid.
# - A new centroid is then calculated from each cluster
# - This process is iterated over until the data points settle in their appropriate cluster
# 

# In[24]:


kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)


# In[25]:


kmeans.fit(sd)


# In[27]:


# The lowest SSE value
kmeans.inertia_


# In[28]:


# Final locations of the centroid
kmeans.cluster_centers_


# In[29]:


# The number of iterations required to converge
kmeans.n_iter_


# In[30]:


kmeans.labels_[:5]


# ## Choosing the value for k
# 
# We will explore the elbow method and the silhoutte coefficient to determine our value for k

# In[33]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(sd)
    sse.append(kmeans.inertia_)


# In[34]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[35]:


# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(sd)
    score = silhouette_score(sd, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[36]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# In[ ]:




