#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning
# 
# We load in the relevant files for inspection and carry out a number of data manipulations, visualisations and processes to understand and analyse our data before proceeding with the clustering algorithm. We also format our data into a single source for our clustering algorithm.

# ## Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ## Data Import
# 
# Our data is split across 12 files.
# These are split into trials of length 95, length 100 and length 150.
# The wi_95 file contains all wins from the 95 trials for each subject, lo_95 contains the losses, index_95 contains which study each subject relates to and choice_95 details what deck each subject chose across each trial.
# 
# These files contain data from 617 healthy subjects who have participated in the Iowa Gambling Task. These participants are split accross 10 independant studies with a variety of participants and length of trial:
# 
# | Study              | Participants | Trials |
# | :--                | ------------ | ------ |
# | Fridberg et al.	 |  15          |      95|	
# | Horstmannb         | 162          |     100|	
# | Kjome et al.	     |  19	        |     100|	
# | Maia & McClelland  |  40	        |     100|		
# | Premkumar et al. 	 |  25	        |     100|	
# | Steingroever et al.| 70	        |     100|	
# | Steingroever et al.| 57	        |     150|	
# | Wetzels et al. 	 | 41	        |     150|
# | Wood et al.	     |153	        |     100|	
# | Worthy et al.      | 35           |     100|

# In[2]:


#import raw data into dataframes
wi_95 = pd.read_csv('../data/wi_95.csv')
wi_100 = pd.read_csv('../data/wi_100.csv')
wi_150 = pd.read_csv('../data/wi_150.csv')
lo_95 = pd.read_csv('../data/lo_95.csv')
lo_100 = pd.read_csv('../data/lo_100.csv')
lo_150 = pd.read_csv('../data/lo_150.csv')
index_95 = pd.read_csv('../data/index_95.csv')
index_100 = pd.read_csv('../data/index_100.csv')
index_150 = pd.read_csv('../data/index_150.csv')
choice_95 = pd.read_csv('../data/choice_95.csv')
choice_100 = pd.read_csv('../data/choice_100.csv')
choice_150 = pd.read_csv('../data/choice_150.csv')


# ## Inspection of the raw data
# 
# We display the wins, losses, index and choices for the trials of length 95 to understand the current format of the data

# In[3]:


#Display wins
wi_95.head()


# In[4]:


#Display losses
lo_95.head()


# In[5]:


#Display index
index_95.head()


# In[6]:


#Display choices
choice_95.head()


# # Data manipulations
# 
# We want to consolidate the data into a single dataframe with certain features. The first step is to aggregate the choices for all subjects into single columns for each card deck.

# In[7]:


#Count values for each choice by deck
agg_choice_95 = choice_95.apply(pd.Series.value_counts, axis=1)

agg_choice_100 = choice_100.apply(pd.Series.value_counts, axis=1)

agg_choice_150 = choice_150.apply(pd.Series.value_counts, axis=1)


# In[8]:


#Display aggregated choice data
agg_choice_95.head()


# We add columns labelled with the total wins and total losses for each subject calculated from the wins and losses raw data we previously imported

# In[9]:


#calculate total wins and total losses for each subject
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


# We then add the index dataframe so as to know which Study each subject is from which will aid with our analysis and visualisations later on

# In[10]:


agg_choice_95.head()


# In[11]:


#Concatenate index to identify the study
final_95 = pd.concat([agg_choice_95, index_95], axis=1)
final_100 = pd.concat([agg_choice_100, index_100], axis=1)
final_150 = pd.concat([agg_choice_150, index_150], axis=1)


# In[12]:


final_95.head()


# We inspect the new dataframes to see if our aggregation has created any null values

# In[13]:


#Display summary info for our data types accross the three dataframes we have created
final_95.info()
final_100.info()
final_150.info()


# This has shown the presence of six null values within the final_150 dataframe. Further inspection of these null values may show us an insight into some participants choices in this task

# In[14]:


sample = final_150[final_150[1].isnull()]
sample.head(2)


# ## Note
# 
# Interestingly this step has shown that two participants, both from the Steingroever2011 study chose deck 3 for 150 consecutive trials. It would be interesting to know whether this was a tactical choice with prior knowledge of how the Iowa gambling task is set up - i.e. with decks 3 and 4 giving the best returns. This may have also been a tactic to quickly finish the game from a participant who was uninterested in the game itself. Either way this slightly varies from the expected results of the Iowa Gambling task in which we expect subjects to show a variety of choices but ultimately learn which of the decks are more rewarding.

# We replace these cells marked as Null with 0 across the data

# In[15]:


#replace all NaN values with '0'
final_150[1] = final_150[1].fillna(0)
final_150[2] = final_150[2].fillna(0)
final_150[4] = final_150[4].fillna(0)


# We change all columns to type int from type float

# In[16]:


#convert float to int
final_150[1] = final_150[1].astype(int)
final_150[2] = final_150[2].astype(int)
final_150[3] = final_150[3].astype(int)
final_150[4] = final_150[4].astype(int)


# We then bring all three dataframes together into a single dataframe named 'final'. This is comprised of:
# 
# - Trials of length 95
# - Trials of length 100
# - Trials of length 150
# 
# We can then do manipulations across all of this data all at once

# In[17]:


#creation of a single dataframe with all 617 subjects accross all studies
temp = final_95.append(final_100)
final = temp.append(final_150)
final


# As can be seen from the above output - the dataframe contains 617 rows (one for each of the the participants). However we dont have a unique identifier for each subject. We also have the issue that there is Subject 1 for each of the three length trials data. We add a column with a unique ID for each of the 617 participants to overcome this issue. We also calculate the balance for each participant after the completion of their trials which is the total wins added to the total losses.

# $$wins + losses = balance$$

# In[18]:


#calculate balance and add Unique ID for participants
final['Unique_ID'] = (np.arange(len(final))+1)
final["balance"] = final["tot_win"] + final["tot_los"]

final = final.rename(columns={1:'Deck_A', 2:'Deck_B', 3:'Deck_C', 4:'Deck_D'})
final


# ## Add Payoff to Dataframe
# 
# We are interested in the payoffs used in the various trials for the Iowa Gambling task study. There were three different kinds of payoff used in all the studies.
# 
# #### Payoff 1
# This scheme gives +250 result from 10 choices of the good decks (C&D) while giving a -250 result from 10 choices of the bad decks (A&B). Payoff 1 also has a feature in which decks A&C result in frequent losses while decks B&D result in infrequent losses. Another feature of payoff 1 is that deck C has varialble loss between -25, -50 or -75. The final feature of Payoff 1 is that wins and losses are in a fixed sequence
# 
# #### Payoff 2
# Payoff 2 is a variant on the first payoff with a number of changes. The first is that the loss from Deck C is constant at -50. The second change involves a random sequence of wins and losses through the trials.
# 
# #### Payoff 3
# Payoff 3 is another variety which still contains the base feature of bad decks A&B and good decks C&D with A&C having frequent losses and B&D having infrequent losses. This scheme differs by having decks that change every 10 trials. This means the bad decks (A&B) produce a -250 result from 10 trials at the start but this decreases by 150 for every 10 trials resulting in a final result of -1000 for 10 card chosen fromn the bad decks. Similarly the good decks start with a +250 result and increase by 25 with every 10 trials finishing with a reward of 375 for every 10 trials. This means both choices are magnified as the trials progress but also the punishment for choosing a bad deck increases far more than the reward for choosing a good deck. The final feature of payoff scheme 3 is that there is a difference in wins between decks and the sequence of wins and losses is fixed.
# 
# This variety in reward/punishment from payoff to payoff is an interesting factor in the outcomes of our project and so we will further explore how much of an impact this variety has within our clustering analysis.

# In[19]:


#payoff data
data = [['Fridberg', 1],['Horstmann', 2],['Kjome', 3],['Maia', 1],['SteingroverInPrep', 2],['Premkumar', 3],['Wood', 3],['Worthy', 1],['Steingroever2011', 2],['Wetzels', 2]]
 
# Create the pandas DataFrame
payoff = pd.DataFrame(data, columns = ['Study', 'Payoff'])
 
# print dataframe.
payoff


# We join the final dataframe with the dataframe containing the payoff values

# In[20]:


final = final.join(payoff.set_index('Study'), on='Study')


# ## Export data
# 
# This final dataframe is in the appropriate format for clustering and for some initial visualisations and analysis. We export the data for use in the clustering notebook and then proceed with our exploration.

# In[21]:


final.to_csv('../data/cleaned_data.csv')


# ## Exploration of the data
# 
# We want to further understand the data set to see if there are any insights or trends that we can explore during our clustering. A number of methods and graphs can help us to understand the make up of this data and what questions we want to answer within this project.

# In[22]:


final.describe()


# We can see that mean choices for decks B, C & D are higher than Deck A showing a clear trend of participants away from A and towards the good decks C&D. Interestingly the mean total wins is less than the mean average losses and hence participants mean balance was -156.83.
# 
# ## Analysis by Individual

# In[23]:


#create histogram for wins
plt.hist(final["tot_win"], bins=100, color="green")
plt.title("Distribution of Wins")
plt.xlabel("Wins Value")
plt.ylabel("Count")
plt.show()

#create histogram for losses
plt.hist(final["tot_los"], bins=100, color="red")
plt.title("Distribution of Losses")
plt.xlabel("Losses Value")
plt.ylabel("Count")
plt.show()


# We can see wins has a right skew distribution with a small number of subjects with wins above 10,000. The losses is similarly distributed but with a left skew distribution. The presence of trials of length 150 in this data alongside the data of trials with length 95/100 would be partly the cause of this.

# In[24]:


#create histogram for wins

to_keep= ['Fridberg','Horstmann','Kjome','Maia','SteingroverInPrep','Premkumar','Wood','Worthy']
filtered_df = final[final['Study'].isin(to_keep)]
plt.hist(filtered_df["tot_win"], bins=100, color="green")
plt.title("Distribution of Wins for trials of length 95/100")
plt.xlabel("Wins Value")
plt.ylabel("Count")
plt.show()

#create histogram for losses
plt.hist(filtered_df["tot_los"], bins=100, color="red")
plt.title("Distribution of Losses for trials of length 95/100")
plt.xlabel("Losses Value")
plt.ylabel("Count")
plt.show()

filtered_df.describe()


# We can see once we remove the trials of length 150 the level of right skew in the wins histogram reduces as does the level of left skew in the losses histogram.

#  ## Analysis by Payoff
# 
# We now look at the distribution of the balance across the different payoff schemes.

# In[25]:


i = 1
while i < 4:
    pay1 = final['Payoff'] == i
    temp = final[pay1]
    plt.hist(temp['balance'], bins = 100)
    plt.title("Payoff "+ str(i) + " Balance Distribution")
    plt.xlabel("Balance")
    plt.ylabel('Count')
    plt.show()
    i += 1
    temp.describe()


# We can see payoff 2 has distribution that centers just above 0 while payoff 1 and 3 have a negative center - payoff 3 showing the worst results. This may be due to the sharp increase in penalty that occurs with every 10 cards chosen in payoff 3.

# In[26]:


#Show wins and losses by Payoff
plt.scatter(final["Payoff"], final["tot_win"], label = "Total Wins", color="g")
plt.scatter(final["Payoff"], final["tot_los"], label = "Total Losses", color="r")

plt.legend(ncol=1, loc='upper left')
plt.title("Wins/Losses by Payoff")
plt.xlabel("Payoff")
plt.ylabel("Value")
plt.xticks([1, 2, 3])
plt.show()


# Payoff 2 shows the widest distribution across wins and losses while payoff 3 again shows a very tight distribution of wins but a wide variety of losses due to this evolving reward/punishment system every 10 cards.

# ## Analysis by Study
# 
# Another theory we have is that the study in which the subject is part of may affect the participants results. We will explore this data with each study in miond to see if any obvious trends emerge. This may also be a useful clustering excercise to see if subjects will cluster with others in their study

# In[27]:


#Show wins and losses by Study
plt.scatter(final["Study"], final["tot_win"], label = "Total Wins", color="g")
plt.scatter(final["Study"], final["tot_los"], label = "Total Losses", color="r")

plt.xticks(rotation=90)
plt.legend(ncol=2)
plt.title("Wins/Losses by Study")
plt.xlabel("Study")
plt.ylabel("Value")
plt.show()


# Steingroever2011 and Wetzels, as the only studies with trials of length 150, show the greatest variety in wins and losses across their participants. The Fridberg study group shows limited wins but limited losses also. This group has 15 participants so this may be a factor although Kjome shows a wider variety of wins and losses with 19 participants.  

# In[28]:


#show balance by study
plt.scatter(final["Study"], final["balance"], label = "Balance")
plt.xticks(rotation=90)
plt.legend()
plt.title("Balance by Study")
plt.show()


# Steingroever2011, Wood and Premkumar show the widest variety in terms of positive and negative balance of its participants. Fridberg and Kjome again vary despite their similar small group size. We can also see that the two participants from Steingroever2011 which chose deck 3 150 times had the greatest profit of 3,750 amongst all participants.
# 
# ## Analysis of deck choices
# 
# One of the features we have included in our dataframe is the number of choices of each deck. We know there are good decks and bad decks with varying rewards and punishments so this will definitely be a feature that will affect our clustering algorithm.

# In[29]:


plt.scatter(final["Unique_ID"], final['Deck_D'], label = "Deck D", s=20)
plt.scatter(final["Unique_ID"], final['Deck_C'], label = "Deck C", s=20)
plt.scatter(final["Unique_ID"], final['Deck_B'], label = "Deck B", s=20)
plt.scatter(final["Unique_ID"], final['Deck_A'], label = "Deck A", s=20)


plt.title("Deck choices by Subject")
plt.xlabel("Subjects")
plt.ylabel("Count")
plt.legend()
plt.show()


# We can see above the tendency of participants to choice Decks 2/3/4 more than 60 times where deck 1 was never chosen this many times by any participant. We can also see the trend of a number of participants who have chose the same deck close to 100% of the trials. The large increase in choices from Subject 500 onwards is due to the trials of length 150.
# 
# There seems to be a unnatural distribution of choices between subjects 300-480 with a consistent number of choices being exactly 60. We will explore this data in more detail:

# In[30]:


filter1 = final["Unique_ID"]>300
filter2 = final["Unique_ID"]<480
final_zoom = final.where(filter1&filter2, inplace=False)
plt.scatter(final_zoom["Unique_ID"], final_zoom['Deck_D'], label = "Deck D", s=30)
plt.scatter(final_zoom["Unique_ID"], final_zoom['Deck_C'], label = "Deck C", s=30)
plt.scatter(final_zoom["Unique_ID"], final_zoom['Deck_B'], label = "Deck B", s=30)
plt.scatter(final_zoom["Unique_ID"], final_zoom['Deck_A'], label = "Deck A", s=30)

plt.title("Deck choices by Subject")
plt.xlabel("Subjects")
plt.ylabel("Choice")
plt.legend()
plt.show()


# Further inspection on a subset of participants shows an unusual trend to choose a specific deck exactly 60 times. This is across decks 2/3/4 but again deck 1 is neglected in this. This may have been a stipulation of the study that no deck was allowed to be chosen more than 60 times.

# In[31]:


plt.scatter(final["Study"], final['Deck_D'], label = "Deck D")
plt.scatter(final["Study"], final['Deck_C'], label = "Deck C")
plt.scatter(final["Study"], final['Deck_B'], label = "Deck B")
plt.scatter(final["Study"], final['Deck_A'], label = "Deck A")
plt.xticks(rotation=90)
plt.legend()
plt.title("Deck Choice by Study")
plt.xlabel("Study")
plt.ylabel("No. of Choices")
plt.show()


# Steingroever2011 had the two most successful participants and also had the two highest choices of a single deck with two members selecting deck C 150/150 trials. There were also a number of other participants who chose a single deck more than 100 times in their study.

# In[32]:


plt.scatter(final["Payoff"], final['Deck_D'], label = "Deck D")
plt.scatter(final["Payoff"], final['Deck_C'], label = "Deck C")
plt.scatter(final["Payoff"], final['Deck_B'], label = "Deck B")
plt.scatter(final["Payoff"], final['Deck_A'], label = "Deck A")
plt.legend()
plt.xticks([1, 2, 3])
plt.title("Deck Choice by Payoff")
plt.xlabel("Payoff")
plt.ylabel("No. of Choices")
plt.show()


# Interestingly payoff 2 shows a much higher selection of choices compared to payoff 1 or payoff 3. The distribution across these three is clearly quite different and so it will be interesting to see if clustering will indicate this also.
