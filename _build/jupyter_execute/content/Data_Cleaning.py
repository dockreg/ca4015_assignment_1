#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning
# We will load in the relevant files for inspection and carry out a number of data manipulations and processes to understand, analyse and format our data into a single source for our clustering algorithm.

# ## Importing required libraries

# In[1]:


import pandas as pd
import numpy as np


# ## Data Import
# 
# Our data is split across 12 files.
# These are split into trials of length 95, length 100 and length 150.
# The wi_95 file contains all wins from the 95 trials for each subject, lo_95 contains the losses, index_95 contains which study each subject relates to and choice_95 details what deck each subject chose across each trial. 

# In[3]:


wi_95 = pd.read_csv('Data/wi_95.csv')
wi_100 = pd.read_csv('Data/wi_100.csv')
wi_150 = pd.read_csv('Data/wi_150.csv')
lo_95 = pd.read_csv('Data/lo_95.csv')
lo_100 = pd.read_csv('Data/lo_100.csv')
lo_150 = pd.read_csv('Data/lo_150.csv')
index_95 = pd.read_csv('Data/index_95.csv')
index_100 = pd.read_csv('Data/index_100.csv')
index_150 = pd.read_csv('Data/index_150.csv')
choice_95 = pd.read_csv('Data/choice_95.csv')
choice_100 = pd.read_csv('Data/choice_100.csv')
choice_150 = pd.read_csv('Data/choice_150.csv')


# ## Inspection of the raw data

# In[6]:


wi_95.head()


# In[7]:


lo_95.head()


# In[8]:


index_95.head()


# In[10]:


choice_95.head()


# # Data manipulations
# 
# We want to consolidate the data into a single dataframe with certain features. The first step is to aggregate the choices for all subjects into a single columns for each card deck

# In[12]:


agg_choice_95 = choice_95.apply(pd.Series.value_counts, axis=1)

agg_choice_100 = choice_100.apply(pd.Series.value_counts, axis=1)

agg_choice_150 = choice_150.apply(pd.Series.value_counts, axis=1)


# In[13]:


agg_choice_95.head()


# We add columns labelled with the total wins and total losses for each subject 

# In[14]:


agg_choice_95["tot_win"] = wi_95.sum(axis=1)
agg_choice_95["tot_los"] = lo_95.sum(axis=1)

agg_choice_100["tot_win"] = wi_100.sum(axis=1)
agg_choice_100["tot_los"] = lo_100.sum(axis=1)

agg_choice_150["tot_win"] = wi_150.sum(axis=1)
agg_choice_150["tot_los"] = lo_150.sum(axis=1)

#resetting index for concatination in the next cell
agg_choice_95.reset_index(inplace=True)
agg_choice_100.reset_index(inplace=True)
agg_choice_150.reset_index(inplace=True)


# We then add the index dataframe so as to know which Study each subject is from

# In[16]:


final_95 = pd.concat([agg_choice_95, index_95], axis=1)
final_100 = pd.concat([agg_choice_100, index_100], axis=1)
final_150 = pd.concat([agg_choice_150, index_150], axis=1)


# We inspect the new dataframes to see if our aggregation has created any null values

# In[20]:


final_95.info()
final_100.info()
final_150.info()


# ## Note
# 
# Interestingly this step has shown that two participants chose the same deck for every single trial. It would be interesting to know whether this was a tactical choice to win the game or whether they were uninterested in the outcome of the game.

# We see that the decks with 0 choices are marked as Null so we replace this with 0 across the data

# In[21]:


final_150[1] = final_150[1].fillna(0)
final_150[2] = final_150[1].fillna(0)
final_150[4] = final_150[1].fillna(0)


# We change all columns to type int from type float

# In[23]:


final_150[1] = final_150[1].astype(int)
final_150[2] = final_150[2].astype(int)
final_150[3] = final_150[3].astype(int)
final_150[4] = final_150[4].astype(int)


# We then bring all three dataframes together

# In[25]:


temp = final_95.append(final_100)
final = temp.append(final_150)
final


# ## Export data
# 
# This final dataframe will be used as the input for the clustering 

# In[26]:


final.to_csv('Data/cleaned_data.csv')

