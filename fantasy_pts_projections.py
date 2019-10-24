#!/usr/bin/env python
# coding: utf-8

# In[189]:


import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)  


# In[247]:


# Get stats and combine into position dataframes

# Function to find positional files, read them, add the week of the game, and return as one single dataframe
def position_df(file):
    l = []
    i = 1
    files = glob.glob(file)
    files = sorted(files)
    print(files)
    for file in files:
        df = pd.read_csv(file)
        df['week'] = i
        l.append(df)
        i += 1
    l = pd.concat(l)
    return(l)


# In[312]:


# Get dataframe for each position
passing = position_df('stats_19/fantasy-stats-passing*.csv')
rec_rush = position_df('stats_19/fantasy-stats-rec*.csv')
dst = position_df('stats_19/fantasy-stats-dst*.csv')
idps = position_df('stats_19/fantasy-stats-defense*.csv')
#kickers = position_df('stats_19/fantasy-stats-kicking*.csv')

# Create schedule dataframe, used to add in opponents
schedule = pd.read_csv('schedule.csv')


# In[250]:


# Create dict of team abvs and full names, fix errors in dict
team_long = np.unique(list([schedule['Winner/tie'],schedule['Loser/tie']]))
team_abv = passing['team'].unique()
team_abv = np.sort(team_abv)
teams_dict = dict(zip(team_long, team_abv))
chargers = teams_dict['Los Angeles Rams']
teams_dict['Los Angeles Rams'] = teams_dict['Los Angeles Chargers']
teams_dict['Los Angeles Chargers'] = chargers
niners = teams_dict['Seattle Seahawks']
teams_dict['Seattle Seahawks'] = teams_dict['San Francisco 49ers']
teams_dict['San Francisco 49ers'] = niners


# In[252]:


# create list of lists of opponenets for each row of the passing df by week
q =[]
for i in range(1,passing['week'].max()+1):
    b = []
    week = schedule[schedule['Week'] == i]
    winners = week['Winner/tie']
    losers = week['Loser/tie']
    matchups = dict(zip(winners,losers))
    matchups2 = dict(zip(losers,winners))
    matchups.update(matchups2)
    for k,v in matchups.items():
        c = []
        c.append(teams_dict[k])
        c.append(teams_dict[v])
        b.append(c)
    b = dict(b)
    z = []
    passing_wk = passing[passing['week'] == i]
    for i in passing_wk['team']:
        z.append(b[i])
    q.append(z)


# In[279]:


def create(df):
    q =[]
    for i in range(1,df['week'].max()+1):
        b = []
        week = schedule[schedule['Week'] == i]
        winners = week['Winner/tie']
        losers = week['Loser/tie']
        matchups = dict(zip(winners,losers))
        matchups2 = dict(zip(losers,winners))
        matchups.update(matchups2)
        for k,v in matchups.items():
            c = []
            c.append(teams_dict[k])
            c.append(teams_dict[v])
            b.append(c)
        b = dict(b)
        z = []
        df_wk = df[df['week'] == i]
        for i in df_wk['team']:
            z.append(b[i])
        q.append(z)
    # Flattend list and append new column to passing df    
    flattened  = [val for sublist in q for val in sublist]
    df['opp'] = flattened
    return df


# In[339]:


rec_rush_df = create(rec_rush)


# In[340]:


rec_rush_df


# In[334]:


rec_rush_df.sort_values('player', inplace=True)


# In[355]:


y = rec_rush_df[['player','fantasyPts']]


# In[336]:


def prep(df):
    df2 = df.drop('team', axis=1)
    df2 = df2.drop('games', axis=1)
    y = df2['fantasyPts']
    df2 = df2.drop('fantasyPts', axis=1)
    df2 = df2.drop('week', axis=1)
    df2 = df2.sort_values('player')
    opp = df2['opp']
    df2 = df2.drop('opp', axis=1)
    df2 = df2.fillna(0)
    df_avg = df.groupby('player').agg('mean')
    df_avg = df_avg.drop('games', axis=1)
    df_avg = df_avg.drop('fantasyPts', axis=1)
    df_avg = df_avg.drop('week', axis=1)
    df_avg = df_avg.fillna(0)
    # Set index for update, update each row with each avg, and add the opponents
    df2.set_index('player', inplace=True)
    df2.update(df_avg)
    df2['opp'] = opp.values
    #Replace oppoenets with dummies for ML algorithm
    df3 = pd.get_dummies(df2, drop_first=True)
    return(df3)


# In[342]:


X = prep(rec_rush_df)


# In[ ]:





# In[367]:


# ML algorithm


# In[343]:


kfold = KFold(n_splits=10, random_state=7)
model = Ridge()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
scoring = 'neg_mean_squared_error'
ridge = Ridge(alpha = 0.5, normalize=True).fit(X_train, y_train)
ridge.score(X_test, y_test)


# In[346]:


ridge.predict(X_test)


# In[347]:


ridge = Ridge(alpha = 0.5, normalize=True).fit(X_train, y_train)


# In[348]:


ridge.predict(X_test)


# In[349]:


y_test


# In[350]:


X_scaled = scale(X)
#passing3 = np.array(passing2)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
ridge = Ridge(alpha = 0.5, normalize=True).fit(X_train, y_train)
ridge.score(X_test, y_test)


# In[351]:


ridge.predict(X_test)


# In[274]:


y_test


# In[ ]:


# Calculate points per game by player


# In[ ]:


# Add defense rank to the player row


# In[ ]:


# Add metric to value players ability (EPA?)


# In[ ]:


# Use metric and defense to train model, test accuracy


# In[ ]:


# Add in value for DFS leagues


# In[ ]:


# Search and optimize lineups for the best value / most points per lineup


# In[ ]:




