#!/usr/bin/env python
# coding: utf-8

# # Assignment 2<br>
# 
# Jennifer Pei-Ling Wu<br>
# Shirley Xinye Gong<br>
# Tracy Yu-Tung Huang<br>
# Wei-Te Fang<br>

# In[19]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


df = pd.read_csv('transactions_n100000.csv', parse_dates=['order_timestamp'])


# In[24]:


df.head()


# In[25]:


# check to see if we need any cleaning
df.info()


# In[26]:


# convert location into categorical
df['location'] = df['location'].astype('str')


# **Reshape and Feature Engineering**
# 
# Since we want to do customer segmentation based on purchasing behaviour, we should aggregate the orders by ticket_id. Thus, we need to reshape our data:

# In[27]:


# Reshape our data:

df1 = df.pivot(index = 'ticket_id',columns ='item_name',values='item_count').fillna(0) 
merged = pd.merge(right = df1, left = df.drop_duplicates(subset=['ticket_id']),right_index=True,left_on='ticket_id',how = 'inner')
merged.reset_index(drop=True,inplace=True) 
merged.drop(['item_name', 'item_count'],axis=1,inplace=True)


# In[28]:


# inspect the merged data
merged.head()


# In[29]:


# save id for further use
id_col = merged['ticket_id']


# We would like to narrow the time of purchase to Day of Week and Time of Day (Hour)

# In[30]:


# Hour and day of week may be influential
merged['order_timestamp'] = pd.to_datetime(merged['order_timestamp'], format="%Y/%m/%d, %H:%M:%S")
merged['Hour'] = merged['order_timestamp'].dt.hour
merged['Day of Week'] = merged['order_timestamp'].dt.dayofweek


# In[31]:


# clean up the dataframe and get rid of some columns 
merged.drop(['ticket_id','order_timestamp','lat','long'],axis=1,inplace=True)


# In[32]:


# inspect again
merged.head()


# In[33]:


# change the type of hour and day of week into string for further processing 
merged['Hour'] = merged['Hour'].astype(str)
merged['Day of Week'] = merged['Day of Week'].astype(str)


# To make further sense of our data, we changed time values to Lunch/Dinner/Late Night categories, and Day of Week to Weekend/Weekdays.

# In[34]:


# create dining window labels - timeframe from visualization 
def get_dining_time(hour):
    hour = int(hour)
    if hour >= 1 and hour <= 15:
        return('Lunch')
    elif hour >= 16 and hour <= 21:
        return('Dinner')
    else: return("Late Night")


# In[35]:


# create weekend weekday labels
def weekday(day):
    day = int(day)
    if day in [5,6]:
        return("Weekend")
    else: return("Weekday")


# In[36]:


# create our new columns 
merged['Meal'] = merged['Hour'].apply(get_dining_time)
merged['Day of Week'] =merged['Day of Week'].apply(weekday)


# In[37]:


# get rid of the hour column, we only need the meal col
merged.drop('Hour',axis=1,inplace=True)


# In[38]:


# check out the meal column
merged['Meal'].value_counts(dropna=False)


# In[39]:


merged.head()


# In order to prepare our data for K-Means Clustering, we create dummy variables

# In[40]:


# transform categorical variables into dummy columns 
df_new = pd.get_dummies(merged)


# In[41]:


df_new.head()


# In order to check the stability of our clusters, we split the dataset into two groups - train and test. We will create the unsupervised clustering algorithm using the training group.

# In[42]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df_new, test_size=0.33, random_state=42)


# In[43]:


# scale data after splitting (fit using training set)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X1 = scaler.fit_transform(train)
X2 = scaler.transform(test)

X = scaler.transform(df_new)


# Training the model

# In[44]:


from sklearn.cluster import KMeans
# use the elbow method

# calculate distortion for a range of number of cluster
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X1)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[45]:


# use our optimal k
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y1 = km.fit_predict(X1)
y2 = km.predict(X2)
y_km = km.predict(X)


# In[46]:


# assign the split dataframes back to dataframe
train = pd.DataFrame(train,columns=df_new.columns)
test = pd.DataFrame(test,columns=df_new.columns)


# In[47]:



# assign results back to the dataframes
train['Cluster'] = y1
test['Cluster'] = y2


# ### Evaluate Results
# 
# Having created the clusters of both our train and test groups, we evaluate the results to evaluate if the clustering algorithm is stable, and if the two clusters created from both groups behave similarly.

# In[48]:


# check out the train clusters

train.groupby(by='Cluster').mean()


# In[49]:


# check out the test clusters

test.groupby(by='Cluster').mean()


# In[50]:


# check cluster proportions - train 
for i in [0,1,2]:
    print("Cluster {} is {}%".format(i,train['Cluster'].value_counts()[i]/670))


# In[51]:


# check cluster proportions - test
for i in [0,1,2]:
    print("Cluster {} is {}%".format(i,test['Cluster'].value_counts()[i]/330))


# It seems like the clusters are quite stable between train and test sets.

# In[52]:



df_new['Cluster'] = y_km # predict on whole dataset

# check out the clusters - whole dataset 

df_new.groupby(by='Cluster').mean()


# In[53]:


# assign ticket id back to the dataset for further visualization
df_new = pd.concat([df_new,id_col],axis=1)


# In[54]:



# save the result on the whole dataset to file 
df_new.to_csv('Result.csv')


# In[55]:


df_new.head()


# In[ ]:




