#!/usr/bin/env python
# coding: utf-8

# # Problem Statement : Given is a dataset containing the data about orders placed on zomato for the restaurants in Banglore. We have to predict the overall rating of the restaurant based on various factors.
# 
# Column Description:
# 
# url: contains the url of the restaurant in the zomato website
# 
# Address: contains the address of the restaurant in Bengaluru
# 
# Name: contains the name of the restaurant.
# 
# Online_order: whether online ordering is available in the restaurant or not.
# 
# Book_table: table book option available or not.
# 
# Rate: contains the overall rating of the restaurant out of 5.
# 
# Votes: contains total number of rating for the restaurant as of the above mentioned date.
# 
# Phone: contains the phone number of the restaurant.
# Location: contains the neighborhood in which the restaurant is located.
# 
# Rest_type: restaurant type.
# Dish_liked: dishes people liked in the restaurant.
# 
# Cuisines: food styles, separated by comma.
# 
# Approx_cost(for two people): contains the approximate cost for meal for two people.
# 
# Reviews_list: list of tuples containing reviews for the restaurant, each tuple.
# 
# Menu_item: contains list of menus available in the restaurant.
# Listed_in(type): type of meal.
# 
# Listed_in(city): contains the neighborhood in which the restaurant is listed.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


order_data=pd.read_csv('zomato.csv')


# In[64]:


order_data.head(2)


# ## Checking the size of DataSet 

# In[4]:


order_data['listed_in(type)']


# Total 51717 Rows and 17 Columns in data set.

# In[5]:


order_data.columns.to_list()


# In[6]:


order_data.shape


# # Renaming some column names

# In[7]:


order_data.rename(columns={'approx_cost(for two people)':'approx_cost','listed_in(type)':'Type_of_Restaurant','listed_in(city)':'Area'},inplace=True)


# In[8]:


order_data.head(1)


# # Data Cleaning

# This step contains dealing with irrelevent data,missing value and dropping of irrelevent columns.

# # Dropping irrelevent columns

# In[9]:


order_data.drop(['url','address','phone','dish_liked','menu_item','Type_of_Restaurant'],axis=1,inplace=True)
order_data.drop(['reviews_list'],axis=1,inplace=True)


# In[10]:


order_data.head(5)


# # checking and handling datatypes

# In[11]:


order_data.info()


# From above result, we can see that the columns { rate, location, rest_type, approx_cost, cuisines } either have null values or wrong datatypes.

# In[12]:


#order_data['name'].isnull().sum()
#order_data['online_order'].isnull().sum()
#order_data['book_table'].isnull().sum()
#order_data['rate'].isnull().sum()
#order_data['votes'].isnull().sum()
#order_data['location'].isnull().sum()
#order_data['rest_type'].isnull().sum()
#order_data['cuisines'].isnull().sum()
#order_data['approx_cost'].isnull().sum()


# In[13]:


order_data['rate'].isnull().sum()
#here Rate column is having null values


# In[14]:


#order_data['rate'].value_counts()
order_data['rate']


# In[15]:


# Changing the Datatype of Rate column from object to string as we got error while replacing null values.
order_data['rate'] = order_data['rate'].astype(str)


# Here we can see there are many NEW and Null values.

# ### Replacing all the null values and garbage values and making it to be converted into numbers.

# In[16]:


order_data['rate']=order_data['rate'].str.replace("/5", "")
order_data['rate']=order_data['rate'].str.replace("NEW","Nan")
order_data['rate']=order_data['rate'].str.replace("nan","Nan")
order_data['rate']=order_data['rate'].str.replace("-","Nan")
order_data['rate']=order_data['rate'].str.replace(" /5", "")
order_data['rate']=order_data['rate'].fillna(np.nan)
order_data['rate']=order_data['rate'].str.replace(" ", "")


# In[17]:


# Verifying the results
order_data['rate'].unique()


# In[18]:


# Changing the Datatype of Rate column from str to float
order_data['rate'] = order_data['rate'].astype(float)


# In[19]:


order_data['rate'].value_counts()


# # checking Approx Cost Column.

# In[20]:


order_data['approx_cost'].isnull().sum()


# In[21]:


order_data['approx_cost'].unique()


# In[22]:


order_data['approx_cost'].value_counts()


# In[23]:


#Replacing the null values and make it able to convert
order_data['approx_cost'] =  order_data['approx_cost'].str.replace("nan", "NaN")
order_data['approx_cost'] =  order_data['approx_cost'].fillna('NaN')
order_data['approx_cost'] =  order_data['approx_cost'].str.replace(",", "")


# In[24]:


order_data['approx_cost'].unique()


# In[25]:


# Changing the Datatype of the column from Object to Float
order_data['approx_cost'] = order_data['approx_cost'].astype(float)


# In[26]:


#Checking the data types of the column to verify.
order_data.info()


# In[27]:


#Checking for null values
order_data.isnull().sum().sum()


# In[28]:


# Checking of Percentage of Null values in Each Column
(order_data.isnull().sum()/order_data.shape[0])*100


# In[29]:


## Here we can see the rate columns is having higher missing value as compared to other columns like location,rest_type,cuisines and approx_cost
# so lets drop null values from the columns having lesser null values.
order_data=order_data[order_data['location'].notnull()]
order_data=order_data[order_data['rest_type'].notnull()]
order_data=order_data[order_data['cuisines'].notnull()]
order_data=order_data[order_data['approx_cost'].notnull()]


# In[30]:


(order_data.isnull().sum()/order_data.shape[0])*100


# In[31]:


order_data.isnull().sum()


# # Handling the null value in Rate columns.

# As the Rate columns having higher number of null values so instead of dropping it. We will impute their values either with mean median or mode.

# In[32]:


# For imputing the null values, We will check for outliers.
q1=order_data['rate'].quantile(0.25)
q3=order_data['rate'].quantile(0.75)

iqr=q3-q1

lower_range=q1-(1.5*iqr)
upper_range=q3+(1.5*iqr)

outliers = len(order_data[(order_data['rate'] < lower_range) | (order_data['rate'] > upper_range)])
print("Number of Rows having Outliers : ", outliers)




# Here, we can see that 183 rows have outliers, but as we know that rating are always in the range from 1.0 to 5.0. So, using the mean is not appropriate. Therefore, we will impute with median.

# In[33]:


# Imputing the null values with median of rate column
order_data['rate'] = order_data['rate'].fillna(order_data['rate'].median())


# In[34]:


order_data.isnull().sum()


# In[35]:


order_data.head()


# # Data Visualization

# ### Lets visualize our data to get some insight and relationship between them.

# #### 1) Number of orders having online orders

# In[36]:


data=order_data[['name', 'online_order']].drop_duplicates()
data


# In[37]:


data.duplicated().sum()
#all duplicates value removed


# In[38]:


plt.figure(figsize=(10,5))
ax = sns.countplot(x="online_order", data=data).set_title('Number of Restaurant having Online Order Facility',fontsize = 15)
plt.xlabel('data')
plt.show()


# # No. of Orders vs Restaurant.

# In[39]:


# Restaurant having higher number of orders.
plt.figure(figsize = (14,5))
data=order_data['name'].value_counts()[:41]
data.plot(kind='bar')
plt.xlabel('Name of Restaurants', size = 14)
plt.ylabel('No. of Orders', size = 14)
plt.title("Restaurants with Maximum No. of Orders", fontsize=15)
plt.show()


# Above Bar chart shows that the Cafe Coffee Day and Onesta got slightly higher number of orders as compared to others restaurants while others got approximately same number of orders.

# # Top Locations got Higher Number of Orders.

# In[40]:


# Checking top Locations got Higher Number of Orders
plt.figure(figsize = (12,5))
data = order_data.location.value_counts()[0:25]
data.plot(kind='bar')
plt.xlabel('Locations')
plt.ylabel('No. of orders')
plt.title ('Top Locations got Higher Number of Orders', size = 15)
plt.show()


# # Number of restaurants having Prebooking Table Facility.

# In[41]:


# Checking the number of restaurants having Prebooking Table Facility
data = order_data[['name', 'book_table']].drop_duplicates()

plt.figure(figsize = (8,6))
ax = sns.countplot(x="book_table", data=data).set_title('Pre Booking Facility', fontsize = 15)
plt.show()


# From above chart, we can see that Most of the restaurants in Banglore doesn't have Pre Booking Facility which is about 8011 restaurants and only 820 restaurants having these facility.

# # Most Common Ratings for orders.

# In[42]:


# Checking the most Common ratings for orders
data = order_data.rate.value_counts().reset_index()[0:20]

plt.figure(figsize = (15,6))
sns.barplot(x = data['index'], y = data['rate'])
plt.xlabel('Ratings')
plt.ylabel('No. of orders')
plt.title ('Most Common Ratings', size = 15)
plt.show()


# # Common Approximate Costs for meal of two people.

# In[43]:


data = pd.DataFrame(order_data["approx_cost"].value_counts().reset_index()[:20])
data.columns = ['approx_cost', 'count']
data = data.set_index('approx_cost')


# In[44]:


data


# In[45]:


plt.figure(figsize = (15, 5))
sns.barplot(x = data.index, y = data['count'])
plt.xlabel('Rate for Two People', size=15)
plt.ylabel('No. of orders', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.title('Top preferred costs for two people', size = 20)
plt.show()


# # Most famous Cusines among People.

# In[46]:


order_data.head(1)


# In[47]:


plt.figure(figsize=(16,6))
Cusines=order_data.cuisines.value_counts()[:30].plot(kind='bar')
plt.title("Most famous Cusines among People",fontsize=15)
plt.xlabel("Cuisines")
plt.ylabel ("No. of Orders")
plt.show()


# In[48]:


order_data.head(5)


# # Encoading Columns.
# ## Encoding online_order column.

# In[49]:


order_data['online_order']=order_data['online_order'].replace({'Yes':1,'No':0})
order_data.head(5)


# In[50]:


order_data['book_table']=order_data['book_table'].replace({'Yes':1,'No':0})
order_data.head()


# In[51]:


order_data['location'].unique()


# # Rest Type Column.

# In[52]:


rest_df = pd.DataFrame(order_data['rest_type'])
rts = list(rest_df.rest_type.str.split(", "))
unique_rts = list(set([rt for sub_list in rts for rt in sub_list]))

for rt in unique_rts:
    rest_df[rt] = int(0)
    
rest_df.head()


# In[53]:


count = 0
for i in rts:
    rest_df.loc[count, i] = int(1)     
    count+=1


# In[54]:


rest_df.head()


# In[60]:


rest_df.head()


# # Encoding Cuisine Column.

# In[62]:


order_data.head()


# In[57]:


order_data['rest_type'] = order_data['rest_type'].str.replace(',' , '') 
order_data['rest_type'] = order_data['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
order_data['rest_type'].value_counts().head()


# In[65]:


order_data.head()


# In[66]:


cuisines_df = pd.DataFrame(order_data['cuisines'])
cuisines = list(cuisines_df.cuisines.str.split(", "))
unique_cuisines = list(set([cs for sub_list in cuisines for cs in sub_list]))
unique_cuisines.remove("Cafe") 
unique_cuisines.remove("Bakery")

for cs in unique_cuisines:
    cuisines_df[cs] = 0   
    
cuisines_df.head()


# In[67]:


corrected_cuisines = []
for i in cuisines:
    if "Cafe" in i:
        i.remove("Cafe")
        
    if "Bakery" in i:
        i.remove("Bakery")
        
    corrected_cuisines.append(list(set(i)))


# In[68]:


count = 0
for i in corrected_cuisines:
    cuisines_df.loc[count, i] = int(1)
    count+=1


# In[70]:


cuisines_df.head(3)


# In[71]:


cuisines_df.drop(['cuisines'], axis = 1, inplace=True)
cuisines_df.head()


# # Location Column.

# In[75]:


location_df = order_data['location']
location_df = pd.get_dummies(location_df)
location_df.head()


# # Area Column.

# In[74]:


order_data.head()


# In[78]:


area_df = order_data['Area']
area_df = pd.get_dummies(area_df)
area_df.head()


# In[80]:


data_with_location = pd.concat([order_data, rest_df, cuisines_df, location_df], axis = 1)
data_with_location.head()


# In[81]:


data_with_area = pd.concat([order_data, rest_df, cuisines_df, area_df], axis = 1)
data_with_area.head()


# In[82]:


print(data_with_location.columns[200:])


# In[ ]:




