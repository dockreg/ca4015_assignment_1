#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing, Standardization, and K-means clustering
# 
# We take our cleaned and formatted data and perform a number of steps to the data to preprocess and standardize before it is ready to use with our k-means clustering algorithm.

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
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure


# ## Import Data

# In[2]:


clean_data = pd.read_csv('../data/cleaned_data.csv')


# # Preprocessing
# 
# ## Label Encoding/One Hot Encoding
# 
# We want to have our data as numerical type for the k-means clustering algorithm. Categorical data such as the column for "Study" pose a problem as the k-means algorithm cannot assign an appropriate value to a string. Two techniques that can solve this problem are label encoding and one hot encoding. Label encoding assigns a numerical value but this does not work for our dataset as it will assign values (1, 2, 3, etc.) which may imply a hierarchy that is not present in our data between each Study. We therefore explore the second option of one hot encoding which creates a new column for each of the 10 studies and then assigns the value 0 if it is not from this study or a 1 if it is. One hot encoding can cause problems when there are many categories but in this case we deemed it appropriate for use. 
# 
# We will use the label encoding technique to convert categorical variables to numbers and then use the one hot encoding toolkit from scikit-learn to convert these into binary values. Scikit-learn has a useful preprocessing module for this.

# ### Label Encoder

# In[3]:


# create instance of labelencoder
labelencoder = LabelEncoder()
# Assign numerical values and store in a new column
clean_data['Study_no'] = labelencoder.fit_transform(clean_data['Study'])


# Display this new column

# In[4]:


clean_data


# ### One Hot Encoding

# In[5]:


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

# In[6]:


df = enc.drop(columns=['level_0', 'index', 'Subj', 'Study', 'Study_no', 'Unnamed: 0', "Unique_ID", "balance"])
df


# ## Standardization
# 
# Generally speaking, learning algorithms perform better with standardized data. In our case there is a large variety in the 0 and 1 values for the studies and the total win and total loss values which vary from 14,750 to -2,725 and so we have chosen to standardize the whole data set. Scitkit-learn again offers a way to do this using the preprocessing module.

# In[7]:


scaler = preprocessing.StandardScaler().fit(df)
X_scaled = scaler.transform(df)


# In[8]:


#rename columns to clearly represent decks
sd = pd.DataFrame(X_scaled, columns=['Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'tot_win', 'tot_los', 'Payoff', 'Fridberg', 'Horstmann', 'Kjome', 'Maia', 'Premkumar', 'Steingroever2011', 'SteingroverInPrep', 'Wetzels', 'Wood', 'Worthy'])
sd


# ## K- means clustering
# 
# Clustering involves grouping data points into groups based on their distance to the centroid of each cluster. It is an unsupervised technique using patterns within the data to decide upon the chosen groups. Within k-means, the k represents the number of clusters we require. We choose this value for k and so the best way to determine the most appropriate value of k is to run experiments for different values of k and then see which results in the least error. 
# 
# #### K means steps:
# 
# - The value of k must be decided
# - k random points are selected as the initial centroids
# - Each data point is assigned to a cluster based on their distance from the nearest initial centroid.
# - A new centroid is then calculated from each cluster
# - This process is iterated over until the data points settle in their appropriate cluster
# 
# This measure of distance d is the euclidean distance given by:
# 
# $$d = √((x1-y1)² + (x2-y2)²)$$
# 

# ## PCA dimensionality reduction
# 
# Principal component analysis is a method whereby we can reduce the dimensions used to represent our data. This can help with the curse of dimensionality which can often make clustering algorithms challenging with so many features. PCA helps us by taking all of these features and represents the data within a new set of features (the number of which we can decide). This can often help with the visualisation of cluster as we can reduce the features to 2 components which will healp to plot our points and therefore visualise the clustering algorithm and how well it represents our data.

# In[9]:


#select how many features we want
pca = PCA(2)
#transform data
df = pca.fit_transform(sd)


# ## Choosing the value for k
# 
# The first step in the k-means algorithm involves calculating what is the most appropriate value for k. We will explore the elbow method and the silhoutte coefficient to determine our value for k within this project.
# 
# From our data exploration thus far we are interested in whether the data should be partitioned in 10 groups (k=10) for each study we have, or whether clustering would be best in 3 groups(k=3) to reflect the 3 payoff schemes we have seen.
# 
# #### Elbow Method
# The elbow method involves calculating the sum of squared error (SSE) and deciding at what point the modelling of the data is most appropriate. Often we can add further clusters without much gain in SSE and so we must determine at which point any further returns are no longer worthwhile.
# 
# $$ SSE = \sum_{i=1}^n (y_i-ŷ_i)^2$$

# In[10]:


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
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# k=3 appears to be a good fit using the elbow method and SSE

# #### Silhouette Coefficient
# 
# The Silhouette coefficient is another measure of clustering and ranges between -1 and 1
# 
#  This value is calculated by the following equation:
#  
#  $$(y-x)/max(x,y)$$
# 
# Where:
# 
# x = average distance between each point in a cluster
# 
# y = average distance between all clusters

# In[11]:


# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# Again we see that k=3 returns the highest value for the silhouette coefficient confirming that payoff scheme may be the most defining feature in the data for the outcome of the Iowa Gambling task. We will proceed to cluster with k=3.

# In[12]:


kmeans = KMeans(n_clusters= 3)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
#Getting unique labels
u_labels = np.unique(label)
 
#plotting the results:
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, s=20)
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


# Interestingly this instance of k-means has split the data into three clusters which may not intuitively be the ideal clusters. This can often be a limiting factor when using k-means. We now add the Payoff labels back onto the data to visuakise the same points with each payoff scheme identified by a different color.

# In[13]:


tmp = pd.DataFrame(df, columns=['Component_1', 'Component_2'])


original_labels = pd.concat([tmp, clean_data[["Study", "Payoff"]]], axis=1)


# In[14]:


original_labels


# In[15]:


colors = {'Fridberg': 'blue','Horstmann': 'red','Kjome': 'green','Maia': 'yellow','SteingroverInPrep': 'black','Premkumar': 'orange','Wood': 'purple','Worthy': 'grey','Steingroever2011': 'brown','Wetzels': 'pink'}
#for i in range(618):
figure(figsize=(6, 4))
plt.scatter(original_labels['Component_1'], original_labels["Component_2"], c = original_labels["Study"].map(colors), s=20)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

for k in colors:
    print(k + ": " + colors[k])


# In[16]:


colors = {1: 'blue',2: 'brown',3:'red'}
#for i in range(618):
figure(figsize=(6, 4))
plt.scatter(original_labels["Component_1"], original_labels["Component_2"], c=original_labels["Payoff"].map(colors), s=20)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Clustering by Payoff")
plt.show()


# We can see that the payoff data clearly has a large impact on our clusters that we have created. All data points have been assigned to the same cluster as other data points in their payoff. 

# ## k=10 (without payoff data)
# 
# To explore the possibility that our data is impacted by the study in which it was recorded we drop the payoff column and then reduce the dimensionality before computing our clusters with k=10 

# In[17]:


#drop payoff column
w_out_payoff = sd.drop(columns=["Payoff"])

pca = PCA(2)
w_out = pca.fit_transform(w_out_payoff)

kmeans = KMeans(n_clusters= 10)
 
#predict the labels of clusters.
label = kmeans.fit_predict(w_out)
 
#Getting unique labels
u_labels = np.unique(label)
 
#plotting the results:
for i in u_labels:
    plt.scatter(w_out[label == i , 0] , w_out[label == i , 1] , label = i, s=20)
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


# We then plot the same data points but clustering by study using a unique color for each

# In[18]:


t = pd.DataFrame(w_out, columns=['Component_1', 'Component_2'])


t_labels = pd.concat([t, clean_data["Study"]], axis=1)

colors = {'Fridberg': 'blue','Horstmann': 'red','Kjome': 'green','Maia': 'yellow','SteingroverInPrep': 'black','Premkumar': 'orange','Wood': 'purple','Worthy': 'grey','Steingroever2011': 'brown','Wetzels': 'pink'}
#for i in range(618):
figure(figsize=(6, 4))
plt.scatter(t_labels['Component_1'], t_labels["Component_2"], c = t_labels["Study"].map(colors), s=20)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

#show study-colors
for k in colors:
    print(k + ": " + colors[k])


# ## Analysis
# 
# The two features we have identified, Payoff scheme and study, have clearly impacted the results of this clustering algorithm. The clusters for payoff show how strongly weighted this was within the clustering algorithm. We can agree from this that the appropriate value for k is 3 and that using k=10 can also tell us an interesting insight into how using study as a features may affect the final clustering result.
