#!/usr/bin/env python
# coding: utf-8

# ### Let's Fetch Data from Excel File

# In[1]:


import pandas as pd

# Load the first three sheets (2018, 2019, 2020) into one DataFrame
df_2018_2020 = pd.concat([
    pd.read_excel(r"C:\Users\61403\Desktop\Hotel_revenue.xlsx", sheet_name="2018", engine="openpyxl"),
    pd.read_excel(r"C:\Users\61403\Desktop\Hotel_revenue.xlsx", sheet_name="2019", engine="openpyxl"),
    pd.read_excel(r"C:\Users\61403\Desktop\Hotel_revenue.xlsx", sheet_name="2020", engine="openpyxl")
], ignore_index=True)

print("First three sheets combined:")
print(df_2018_2020.head())


# In[2]:


df_2018_2020


# In[3]:


df = df_2018_2020


# In[4]:


df


# In[5]:


print(df.isnull().sum())


# In[6]:


df.describe().T


# ### Identified Outliers  
# 
# - The `adr` (average daily rate) has a negative value, which is not logically valid.  
# - Both `children` and `babies` have values of 10, which seems unrealistic and may indicate data entry errors.  
# 

# In[7]:


len(df.columns)


# ### Handling Missing Values  
# 
# - The `children` column has only 8 missing values, so we will replace them with the mean, which is 0.  
# - The `country` column has 625 missing values. Since Portugal is the most frequently occurring country in the dataset, we will replace the missing values with "Portugal."
# - The `company` column has around 90% missing values, so we will drop the column.  
# - The `agent` column contains missing values, which we will replace with `-1` to flag them.  
# 
# 
# 

# In[8]:


df.loc[:, 'children'] = df['children'].fillna(0)


# In[9]:


df.loc[:, 'country'] = df['country'].fillna('PRT')


# In[10]:


df = df.drop(columns=['company'])


# In[11]:


df.loc[:, 'agent'] = df['agent'].fillna(-1)


# # Handling Outliers

# ## Lead Time

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew
from scipy.stats.mstats import winsorize


# In[13]:


sns.boxplot(x=df['lead_time'])


# In[14]:


# Create the histogram
n, bins, patches = plt.hist(df['lead_time'], bins=10, edgecolor='black')

plt.title('Histogram for Lead Time')
plt.xlabel('Lead Time')
plt.ylabel('Frequency')

for i in range(len(n)):
    plt.text(bins[i], n[i], str(int(n[i])), va='bottom', ha='center')

plt.show()


# In[15]:


# Log Transformation
df['lead_time_log'] = np.log(df['lead_time'] + 1)

# Square Root Transformation
df['lead_time_sqrt'] = np.sqrt(df['lead_time'])

# Box-Cox Transformation
df['lead_time_boxcox'], _ = boxcox(df['lead_time'] + 1)  # Add 1 to handle zeros

# Capping Outliers
df['lead_time_capped'] = df['lead_time'].clip(upper=300)

# Binning
df['lead_time_binned'] = pd.cut(df['lead_time'], bins=10, labels=False)

# Winsorization
df['lead_time_winsorized'] = winsorize(df['lead_time'], limits=[0.05, 0.05])


# In[16]:


# List of columns to plot
columns = [
    'lead_time_log', 
    'lead_time_sqrt', 
    'lead_time_boxcox', 
    'lead_time_capped', 
    'lead_time_winsorized'
]

for col in columns:
    n, bins, patches = plt.hist(df[col], bins=10, edgecolor='black')
    
    plt.title(f'Histogram for {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    for i in range(len(n)):
        plt.text(bins[i], n[i], str(int(n[i])), va='bottom', ha='center')
    
    plt.show()


# ### Handling Skewness in Lead Time  
# 
# After analyzing the distribution of the `lead_time` column using a histogram, we observed a highly skewed distribution with a long right tail. To address this, we applied various transformations, including logarithmic transformation, binning, clipping, square-root transformation, Box-Cox transformation, and winsorization.  
# 
# After visualizing the results of each transformation, we found that the **Box-Cox transformation** produced the most normalized distribution, resembling a bell curve. Therefore, we selected this transformation and updated the original DataFrame accordingly.  
# 

# In[17]:


df['lead_time'] = df['lead_time_boxcox']


# ## Average Daily Rate

# In[18]:


df['adr'].describe()


# In[19]:


sns.boxplot(x=df['adr'])


# In[20]:


bin_edges = range(80, 500, 20)
plt.hist(df['adr'], bins=bin_edges, edgecolor='black')
plt.title('Histogram for ADR')
plt.xlabel('ADR (Average Daily Rate)')
plt.ylabel('Frequency')
plt.show()


# In[21]:


# Replacing negative values in 'adr' column with the median of 'adr'
df['adr'] = df['adr'].apply(lambda x: df['adr'].median() if x < 0 else x)


# ### Handling '0' Values in Average Daily Rate and Market Segments  
# 
# We began by checking the count of '0' values in the `average_daily_rate` and `market_segments` columns. A value of '0' is valid for complementary bookings; however, for other segments, we will replace '0' with the median of non-complementary bookings.
# 
# Additionally, we will create a new column, `is_complementary`, to flag whether a booking is complementary with a binary value:  
# - `0` for non-complementary bookings  
# - `1` for complementary bookings  
# 

# In[22]:


zero_adr_count = df[df['adr'] == 0].shape[0]
print(f"Number of zero ADR values: {zero_adr_count}")


# In[23]:


zero_adr_bookings = df[df['adr'] == 0]
print(zero_adr_bookings['market_segment'].value_counts())


# In[24]:


df['is_complementary'] = df['market_segment'] == 'Complementary'

median_of_non_complementary = df[df['market_segment'] != 'Complementary']['adr'].median()

df.loc[(df['market_segment'] != 'Complementary') & (df['adr'] == 0), 'adr'] = median_of_non_complementary


# ### Handling Outliers and Transformations  
# 
# - **Clipping**: I will clip the data using the 1st and 99th percentiles to remove outliers.
# - **Visualization**: After clipping, I will visualize the data to confirm the removal of outliers.
# - **Transformation**: Finally, I will apply transformations to the normalized data to ensure it is properly scaled and ready for analysis.
# 

# In[25]:


# Calculate the 1st and 99th percentiles
lower_threshold = df['adr'].quantile(0.01)
upper_threshold = df['adr'].quantile(0.99)

# Cap/Floor the outliers
df['adr'] = df['adr'].clip(lower=lower_threshold, upper=upper_threshold)


# In[26]:


from scipy.stats import skew
skewness = skew(df['adr'])
print(f"Skewness: {skewness}")


# In[27]:


#applying transformation logic

# Log Transformation
df['adr_log'] = np.log(df['adr'] + 1)

# Square Root Transformation
df['adr_sqrt'] = np.sqrt(df['adr'])

# Box-Cox Transformation
df['adr_boxcox'], _ = boxcox(df['adr'] + 1) 


# In[28]:


columns = [
    'adr_log', 
    'adr_sqrt', 
    'adr_boxcox'
]

for col in columns:
    n, bins, patches = plt.hist(df[col], bins=10, edgecolor='black')
    
    plt.title(f'Histogram for {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    for i in range(len(n)):
        plt.text(bins[i], n[i], str(int(n[i])), va='bottom', ha='center')
    
    plt.show()


# In[29]:


df['adr'] = df['adr_sqrt']


# ### Arrival Date Year

# In[30]:


year_counts = df['arrival_date_year'].value_counts().sort_index()

plt.bar(year_counts.index, year_counts.values, color='blue')


# In[31]:


df['arrival_date_year'].describe(include = object ).T


# ### Creating New Features  
# 
# - I created the `total_stay_duration` column, which is the sum of `stays_in_weekend_nights` and `stays_in_week_nights`.  
# - I also created the `weekend_stay_ratio` column, which is calculated by dividing `stays_in_weekend_nights` by the `total_stay_duration` (with 1 added to avoid division by zero).  
# 
# These new features utilize the `stays_in_weekend_nights` and `stays_in_week_nights` columns for further analysis.
# 

# In[32]:


df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']


# In[33]:


df['weekend_stay_ratio'] = df['stays_in_weekend_nights'] / (df['total_stay_duration'] + 1)  # Add 1 to avoid division by zero


# ### Adults

# In[34]:


df['adults'].describe()


# In[35]:


zero_adult = df[df['adults'] == 0].value_counts()
print(zero_adult.shape[0])


# In[36]:


df['adults'] = df['adults'].clip(upper=df['adults'].quantile(0.99))


# ### Handling the `adults` Column  
# 
# - There are 428 rows where the `adults` column has a value of 0. These could represent bookings for children, so we will only remove the rows where `adults`, `children`, and `babies` are all 0.
# - There is an outlier with 55 adults, so we will cap this value by the 99th percentile to handle the extreme value and prevent it from skewing the data.
# - Additionally, we are creating a new feature column, `total_people`, which will sum the values of `adults`, `children`, and `babies` to represent the total number of people in the booking.
# 

# In[37]:


zero_adult_children_babies = df[(df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0)][['adults', 'children', 'babies']]
print(zero_adult_children_babies)


# In[38]:


# Creating the 'total_people'
df.loc[:,'total_people'] = df['adults'] + df['children'] + df['babies']


# In[39]:


# Removing rows where adults, children, and babies are all 0, modifying the original df
df = df[~((df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0))]


# ## Previous Cancellations , Booking Changes , Days In Waiting List
# 
# ### Handling Outliers and Applying Box-Cox Transformation  
# 
# - We applied clipping at the 99th percentile to handle outliers for the `previous_cancellations`, `booking_changes`, and `days_in_waiting_list` columns.  
# - After clipping the `days_in_waiting_list` column, we noticed that the maximum value was 77 days. While this is an outlier compared to the normal values of 2-3 days, bookings can sometimes be made 77 days ahead in business scenarios. Thus, we decided to keep it.
# - The `days_in_waiting_list` column has a skewness of 6.32, indicating it is highly skewed. To address this, we applied the Box-Cox transformation to normalize the data.  
# 
# 
# 

# In[40]:


df['previous_cancellations'].describe()


# In[41]:


df.loc[:, 'previous_cancellations'] = df['previous_cancellations'].clip(upper=df['previous_cancellations'].quantile(0.99))


# In[42]:


df['booking_changes'].describe()


# In[43]:


df.loc[:, 'booking_changes'] = df['booking_changes'].clip(upper=df['booking_changes'].quantile(0.99))


# In[44]:


df['days_in_waiting_list'].describe()


# In[45]:


df.loc[:, 'days_in_waiting_list'] = df['days_in_waiting_list'].clip(upper=df['days_in_waiting_list'].quantile(0.99))


# In[46]:


bin_edges = range(0, int(df['days_in_waiting_list'].max()) + 10, 10)

plt.hist(df['days_in_waiting_list'], bins=bin_edges, edgecolor='black')
plt.title('Histogram for Days_in_waiting_list')
plt.xlabel('Days_in_waiting_list')
plt.ylabel('Frequency')
plt.show()


# In[47]:


# Clip values above the 99th percentile using .loc
df.loc[:, 'days_in_waiting_list'] = df['days_in_waiting_list'].clip(upper=df['days_in_waiting_list'].quantile(0.99))

# Check the statistics again
print(df['days_in_waiting_list'].describe())


# In[48]:


skewness = df['days_in_waiting_list'].skew()
print(f"Skewness after clipping: {skewness}")


# In[49]:


# Ensure that all values are positive (adding 1 to handle any zeros)
df['days_in_waiting_list_positive'] = df['days_in_waiting_list'] + 1

# Apply Box-Cox transformation
df['boxcox_days_in_waiting_list'], lam = boxcox(df['days_in_waiting_list_positive'])

# Check skewness after Box-Cox transformation
new_skewness = df['boxcox_days_in_waiting_list'].skew()
print(f"Skewness after Box-Cox transformation: {new_skewness}")


# In[50]:


df['boxcox_days_in_waiting_list'].describe()


# In[51]:


df.loc[:, 'days_in_waiting_list'] = df['boxcox_days_in_waiting_list']


# ### Required Car Parking Spaces

# In[52]:


df['required_car_parking_spaces'].describe().T


# # Categorical Columns

# In[53]:


categorical_columns = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type', 'reservation_status'
]

plt.figure(figsize=(15, 20))

for i, col in enumerate(categorical_columns, 1):
    plt.subplot(4, 3, i)
    sns.countplot(data=df, y=col, order=df[col].value_counts().index, palette="viridis")
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# In[54]:


sns.countplot(data=df, x='hotel')
plt.title('Distribution of Hotel Types')
plt.show()


# In[55]:


categorical_columns = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type', 'reservation_status'
]

for col in categorical_columns:
    print(f"\nDistribution of {col}:\n")
    print(df[col].value_counts(normalize=True) * 100)  # Convert to percentage
    print("-" * 50)


# In[56]:


sns.countplot(data=df, x='hotel', hue='reservation_status')
plt.title('Cancellation Rates by Hotel Type')
plt.show()


# In[57]:


pd.crosstab(df['arrival_date_month'], df['reservation_status']).plot(kind='bar', stacked=True)
plt.title('Cancellation Rates by Arrival Month')
plt.show()


# ### Encoding of Categorical Features
# 
# - **Hotel**: This is a **nominal** categorical variable, where:
#   - `0` represents **City Hotel**
#   - `1` represents **Resort Hotel**
#   
#   Since this is a nominal variable, we applied **label encoding** to convert it into numerical format.
# 
# - **Arrival_Date_Month**: This is an **ordinal** categorical variable, where:
#   - `1` represents **January**
#   - `2` represents **February**
#   - `3` represents **March**, and so on.
# 
#   As this variable has a natural ordering (months of the year), we applied **label encoding** based on this ordinal relationship.

# In[58]:


df.loc[:,'hotel'] = df['hotel'].map({"City Hotel" : 0 , "Resort Hotel" : 1})
df.loc[:,'arrival_date_month'] = df['arrival_date_month'].map({"January" : 1 , "February" : 2 , "March" : 3 , "April" : 4 , "May" : 5 , "June" : 6 , "July" : 7 , "August" : 8 , "September" : 9 , "October" : 10 ,  "November" : 11 , "December" : 12 })


# ### Encoding of Nominal Features
# 
# - **Meal**: This is a **nominal** categorical variable. We performed **numerical encoding** as we are using binary classification algorithms, which will treat these numbers as nominal rather than ordinal. The encoding is as follows:
#   - `BB` = `0`
#   - `HB` = `1`
#   - `SC` = `2`
#   - `Others` = `3` (We combined the undefined and `FB` categories into `Others` as they have negligible values.)
# 
# - **Country**: We applied the same strategy for the **Country** column. We merged small countries with negligible values into an `Others` category, and for the rest of the countries, we assigned numerical labels (`0`, `1`, `2`, etc.).
# 
# - **Market Segment**: Similarly, we merged smaller market segments with negligible values into an `Others` category, and the remaining segments were assigned numerical labels (`0`, `1`, `2`, etc.).
# 
# This encoding approach allows these nominal variables to be processed appropriately by classification algorithms.
# 

# In[59]:


df['meal'].value_counts()


# In[60]:


df.loc[:,'meal'] = df['meal'].apply(lambda x:'others' if x in ['Undefined','FB'] else x)
df.loc[:,'meal'] = df['meal'].map({'BB' : 0 , "HB" : 1 , "SC" : 2 , "others" : 3})


# In[61]:


df['meal'].value_counts()


# In[62]:


df['meal'].isna().sum()


# In[63]:


Top_countries = ['PRT' , 'GBR' , 'FRA' , 'ESP' , 'DEU']

df.loc[:,'country'] = df['country'].apply( lambda x: x if x in Top_countries else 'Others')
df.loc[:,'country'] = df['country'].map({'PRT' : 0 , 'GBR' : 1 , 'FRA':2,'ESP':3,'DEU':4,'Others':5})


# In[64]:


segment = ['Online TA','Offline TA/TO','Groups','Direct','Corporate']

df.loc[:,'market_segment'] = df['market_segment'].apply(lambda x:x if x in segment else 'others')
df.loc[:,'market_segment'] = df['market_segment'].map({'Online TA':0,'Offline TA/TO':1,'Groups':2,'Direct':3,'Corporate':4,'others':5})


# In[65]:


top_categories = ['TA/TO', 'Direct', 'Corporate']

df.loc[:, 'distribution_channel'] = df['distribution_channel'].apply(lambda x: x if x in top_categories else 'Others')

encoding_map = {
    'TA/TO': 0,
    'Direct': 1,
    'Corporate': 2,
    'Others': 3
}

df.loc[:, 'distribution_channel'] = df['distribution_channel'].map(encoding_map)


# ### Reserved room type and assigned room type have a discrepeancy

# In[66]:


#df['reserved_room_type'].nunique()

reserved_room_type_count = df['reserved_room_type'].value_counts()

reserved_room_type_count


# In[67]:


#df['assigned_room_type'].nunique()

assigned_room_type_count = df['assigned_room_type'].value_counts()

assigned_room_type_count


# ### Room Type Encoding
# 
# In the **Assigned Room Type** column, we observed two extra values: `'I'` and `'K'`. These values are not clearly defined in the dataset. To maintain **data consistency** and handle this ambiguity, we will classify these two values as **'Others'**. This approach will help avoid inconsistencies in the dataset.
# 
# Additionally, the frequency discrepancy observed in these values is likely due to **room upgrades**. However, since we don't have records for room upgrades, we will assume that these values represent an upgrade and group them accordingly under the `Others` category.
# 
# This step will ensure we handle all values in a consistent manner.
# 

# In[68]:


df.loc[:, 'assigned_room_type'] = df['assigned_room_type'].apply(
    lambda x: x if x in {'A', 'D', 'E', 'F', 'G', 'C', 'B', 'H', 'L'} else 'others'
)


# In[69]:


#df['assigned_room_type'].nunique()

assigned_room_type_transformed = df['assigned_room_type'].value_counts()

assigned_room_type_transformed


# In[70]:


# Check for null values
print("Null values in 'assigned_room_type':", df['assigned_room_type'].isnull().sum())

# Inspect unique values
print("Unique values in 'assigned_room_type':", df['assigned_room_type'].unique())


# In[71]:


# Define the mapping
labelling_assigned = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'others': 9,
    'L': 10
}

labelling_reserved = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'L': 10
}

# Apply the mapping
df.loc[:,'assigned_room_type'] = df['assigned_room_type'].map(labelling_assigned)

# Apply the mapping
df.loc[:,'reserved_room_type'] = df['reserved_room_type'].map(labelling_reserved)


# ### Adding Room Upgrade Feature
# 
# We are introducing a new feature called **`room_upgrade`**, which indicates whether a room upgrade has occurred. This feature is based on the comparison between the **reserved room type** and the **assigned room type**.
# 
# - If the **reserved room type** is different from the **assigned room type**, we will consider it as a **room upgrade** (set to `1`).
# - If the **reserved room type** is the same as the **assigned room type**, it means no upgrade occurred (set to `0`).
# 
# This new feature will help us understand and model the potential impact of room upgrades in our data.
# 

# In[72]:


# Adding 'room_upgrade' feature by comparing 'reserved_room_type' and 'assigned_room_type'
df.loc[:, 'room_upgrade'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)


# ### Deposit type , Customer Type
# 

# In[73]:


deposit = {
    'No Deposit' : 0,
    'Non Refund' : 1,
    'Refundable' : 2
}

df.loc[:, 'deposit_type'] = df['deposit_type'].map(deposit)


# In[74]:


df['customer_type'].value_counts()


# In[75]:


mapping = {
    'Transient': 0,
    'Transient-Party': 1,
    'Contract': 2,
    'Group': 3
}

df.loc[:,'customer_type'] = df['customer_type'].map(mapping)


# ### Feature Selection

# In[76]:


features = [
    'hotel',
    'lead_time',
    'arrival_date_month',
    'total_stay_duration',
    'weekend_stay_ratio',
    'adults',
    'children',
    'babies',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'reserved_room_type',
    'assigned_room_type',
    'booking_changes',
    'deposit_type',
    'days_in_waiting_list',
    'customer_type',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests',
    'is_complementary',
    'room_upgrade'
]


# In[77]:


X = df[features]
y = df['is_canceled']


# In[78]:


X.describe()


# In[79]:


X.describe(include = object).T


# In[80]:


df[features].isna().sum()


# ### Splitting Dataset

# In[81]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# In[82]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


# In[83]:


from sklearn.metrics import classification_report, roc_auc_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC-AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))


# ### Classification Report:
# 
# The classification report provides several important evaluation metrics that help assess the performance of a model. The key metrics in the report are **Precision**, **Recall**, **F1-Score**, and **Support**.
# 
# 1. **Precision**:  
#    Precision is the ratio of correctly predicted positive observations to the total predicted positives. It is a measure of how many of the items we predicted as positive are actually positive. For class 0, the precision is 0.91, and for class 1, it is 0.90.
# 
# 2. **Recall**:  
#    Recall, also known as sensitivity or true positive rate, is the ratio of correctly predicted positive observations to all observations in the actual class. It tells us how many actual positives were correctly identified. For class 0, recall is 0.94, and for class 1, recall is 0.85.
# 
# 3. **F1-Score**:  
#    The F1-Score is the harmonic mean of Precision and Recall. It is particularly useful when the class distribution is imbalanced. For class 0, the F1-Score is 0.93, and for class 1, it is 0.87.
# 
# 4. **Support**:  
#    Support is the number of actual occurrences of the class in the dataset. In this case, class 0 has 17,785 samples, and class 1 has 10,563 samples.
# 
# Additionally, the report provides the following averages:
# - **Macro Average**: This averages the performance metrics for all classes without considering class imbalance. In this case, the macro average for F1-Score is 0.90, precision is 0.91, and recall is 0.89.
# - **Weighted Average**: This averages the performance metrics while considering the support for each class. The weighted averages are:
#   - F1-Score: 0.91
#   - Precision: 0.91
#   - Recall: 0.91
# 
# ### ROC-AUC Score:
# 
# The **ROC-AUC score** is 0.9675, which indicates a high performance of the model in distinguishing between classes. A score close to 1.0 suggests that the model is doing a great job at correctly predicting the positive and negative classes. In this case, the ROC-AUC score of 0.9675 shows that the model has a high ability to differentiate between class 0 and class 1.
# 

# In[84]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


feature_importances = model.feature_importances_


if isinstance(X_train, pd.DataFrame):
    feature_names = X_train.columns
else:
    feature_names = [f"Feature {i}" for i in range(len(feature_importances))] 


sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 5))
plt.barh(range(len(feature_importances)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(feature_importances)), np.array(feature_names)[sorted_idx])  
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in RandomForest")
plt.gca().invert_yaxis() 
plt.show()


# In[ ]:




