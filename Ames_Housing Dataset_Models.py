#!/usr/bin/env python
# coding: utf-8

# # Ames house price prediction

# In[1]:


#Import  necessary libraries
#Data Analysis 
import numpy as np
import pandas as pd

# For data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#For Mathematical operations and styling
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.columns",None)
import math

#For Advanced statistical functions
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

#Inferential Analysis
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

#Data Encoding
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from category_encoders import TargetEncoder

#Data Splitting
from sklearn.model_selection import train_test_split

# Machine learning Algorithms
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor

#Evaluation Metrics 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Read the house price CSV file
df_Ames = pd.read_csv('Ames_HousingDataset.csv')


# In[2]:


#Display total number of rows and columns in the dataset

print("The Dataset has ",df_Ames.shape[0],"Records/Rows and ",df_Ames.shape[1],"attributes/columns.")


# In[3]:


df_Ames.head(5)


# In[4]:


#Calculate the number of categorical columns and Numeric columns
numeric_cols = df_Ames.select_dtypes(include=['int', 'float']).columns
num_numeric_cols = len(numeric_cols)

categorical_cols = df_Ames.select_dtypes(include=['object']).columns
num_categorical_cols = len(categorical_cols)

print("Total number of rows:", df_Ames.shape[0])
print("Total number of columns:", df_Ames.shape[1])

print("Number of numeric columns:", num_numeric_cols)
print("Number of categorical columns:", num_categorical_cols)


# In[5]:


#To display discrete and continouse numerical/categorical features in the dataset

# Separate numerical features from the DataFrame
numerical_features = df_Ames.select_dtypes(include=['int64', 'float64'])
# Identify discrete and continuous numerical features
discrete_numerical_features = []
continuous_numerical_features = []

for feature in numerical_features.columns:
    unique_count = numerical_features[feature].nunique()
    if unique_count <= 10:  
        discrete_numerical_features.append(feature)
    else:
        continuous_numerical_features.append(feature)

print("Discrete Numerical Features:")
print(discrete_numerical_features)

print("\nContinuous Numerical Features:")
print(continuous_numerical_features)

# Separate categorical features from the DataFrame
categorical_features = df_Ames.select_dtypes(include=['object'])

# Identify categorical features with high /low cardinality
high_cardinality_categorical_features = []
low_cardinality_categorical_features = []

for feature in categorical_features.columns:
    unique_count = categorical_features[feature].nunique()
    if unique_count > 10: 
        high_cardinality_categorical_features.append(feature)
    else:
        low_cardinality_categorical_features.append(feature)

# Display categorical features with high & low cardinality
print("\nCategorical Features with High Cardinality:")
print(high_cardinality_categorical_features)

print("\nCategorical Features with Low Cardinality:")
print(low_cardinality_categorical_features)


# # Missing Values

# In[6]:


#Calculate % of missing values in the dataset and display

null_df = df_Ames.isnull().sum()[df_Ames.isnull().sum()>0].sort_values().to_frame().rename(columns={0:"Total Missing values"})
null_df["% of Missing Values"] = round(null_df["Total Missing values"]/len(df_Ames)*100,2)
null_df["Feature Data Type"] = df_Ames[null_df.index.tolist()].dtypes
null_df


# In[7]:


#Plot a bar grpah showing top 20 missing count

#Calculate missing percentage 
total = df_Ames.isnull().sum().sort_values(ascending=False)
percent = (df_Ames.isnull().sum()/df_Ames.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# Select the top 20 variables with missing values
top_missing = missing_data.head(20)

# Create the bar plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top_missing.index, top_missing['Percent'] * 100)
ax.set_xlabel('Missing Percentage')
ax.set_ylabel('Feature')
ax.set_title('Missing Data Percentage')
for i, v in enumerate(top_missing['Percent'] * 100):
    ax.text(v + 0.5, i, f'{v:.2f}%', va='center')

plt.show()


# In[8]:


# Handling missing values as per the data description file

# Update MiscVal to zero where MiscFeature is None
df_Ames.loc[df_Ames['MiscFeature'].isnull(), 'MiscVal'] = 0


# In[9]:


#update Garage features based on GarageType, if GarageType NA , there is no Garage, hence  all Garage features updates to'NoGrg'

# List of  Garage categorical columns
categorical_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

# List of  Garage numerical columns
numerical_cols = ['GarageYrBlt', 'GarageCars', 'GarageArea']

# Update Garage categorical columns where GarageType is null
df_Ames.loc[df_Ames['GarageType'].isnull(), categorical_cols] = 'NoGrge'

# Update Garage numerical columns where GarageType is null
df_Ames.loc[df_Ames['GarageType'].isnull(), numerical_cols] = 0


# In[10]:


#Dealing with numerical basement features

# Fill  missing values with zero
columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
df_Ames[columns] = df_Ames[columns].fillna(0)

# Calculate TotalBsmtSF as the sum of BsmtFinSF1, BsmtFinSF2, and BsmtUnfSF
df_Ames['TotalBsmtSF'] = df_Ames['BsmtFinSF1'] + df_Ames['BsmtFinSF2'] + df_Ames['BsmtUnfSF']


# In[11]:


#dealing with basement Categorical variables 

# List all categorical basment columns
categorical_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

# Check if all categorical columns are null in a row
mask_all_null = df_Ames[categorical_cols].isnull().all(axis=1)

# Update rows with all null categorical values with 'NoBsmt'
df_Ames.loc[mask_all_null, categorical_cols] = 'NoBsmt'


# In[12]:


#handling MasVnr features

# Calculate the mode of MasVnrType
most_common_type = df_Ames['MasVnrType'].mode().iloc[0]

# Fill missing values in MasVnrType with mode
df_Ames['MasVnrType'].fillna(most_common_type, inplace=True)

# Fill MasVnrArea with zero where MasVnrType is 'None'
df_Ames.loc[df_Ames['MasVnrType'] == 'None', 'MasVnrArea'] = 0

# Calculate the median of MasVnrArea for each MasVnrType group
median_vnr_area_by_type = df_Ames.groupby('MasVnrType')['MasVnrArea'].median()

# Fill missing values in MasVnrArea with the median value specific to each MasVnrType group
df_Ames['MasVnrArea'] = df_Ames.apply(lambda row: median_vnr_area_by_type[row['MasVnrType']] 
                                      if pd.isnull(row['MasVnrArea']) else row['MasVnrArea'], axis=1)


# In[13]:


#Dealing with  MS Zoning  & Lotfrontage  

# Calculate the mode values for each 'Neighborhood' in the 'MSZoning' column
mszoning_mode = df_Ames.groupby('Neighborhood')['MSZoning'].apply(lambda x: x.mode().iloc[0])

# Map the mode values to the missing values in 'MSZoning' based on 'Neighborhood'
df_Ames['MSZoning'] = df_Ames.apply(
    lambda row: row['MSZoning'] if pd.notna(row['MSZoning']) else mszoning_mode[row['Neighborhood']],
    axis=1
)

#LotFrontage , replace with median value as its not possible to have no street infront of lot
df_Ames['LotFrontage'] = df_Ames.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# In[14]:


#Exterior features

# Group by 'Exterior1st' and 'Exterior2nd', and get the most frequent pair
most_frequent_pair = df_Ames.groupby(['Exterior1st', 'Exterior2nd']).size().idxmax()

# Fill missing values in 'Exterior1st' and 'Exterior2nd' with the most frequent values
df_Ames['Exterior1st'].fillna(most_frequent_pair[0], inplace=True)
df_Ames['Exterior2nd'].fillna(most_frequent_pair[1], inplace=True)


# In[15]:


#Dealing with Sale Type and Electrical features

# Calculate the mode for each column
electrical_mode = df_Ames['Electrical'].mode().iloc[0]
sale_type_mode = df_Ames['SaleType'].mode().iloc[0]

# Fill missing values with mode value
df_Ames['Electrical'].fillna(electrical_mode, inplace=True)
df_Ames['SaleType'].fillna(sale_type_mode, inplace=True)


# In[16]:


#Dealing with KitchenQual, update missing values with mode

KitchenQual_mode= df_Ames['KitchenQual'].mode().iloc[0]
df_Ames['KitchenQual'].fillna(KitchenQual_mode, inplace=True)


# In[17]:


# Dealing with PoolQC column 

# Find the mode of 'PoolQC' where 'PoolArea' is not 0
mode_poolqc = df_Ames.loc[df_Ames['PoolArea'] != 0, 'PoolQC'].mode().iloc[0]

# Fill the 'PoolQC' column with 'NoPool' where 'PoolQC' is null and 'PoolArea' is 0
df_Ames.loc[(df_Ames['PoolQC'].isnull()) & (df_Ames['PoolArea'] == 0), 'PoolQC'] = 'NoPool'

# Fill the 'PoolQC' column with the mode where 'PoolQC' is null and 'PoolArea' is not 0
df_Ames.loc[(df_Ames['PoolQC'].isnull()) & (df_Ames['PoolArea'] != 0), 'PoolQC'] = mode_poolqc


# In[18]:


#create a list to handle NA values in below features,these missing values were filled based on data description file

Impute_list = [
    ('MiscFeature', 'None'),
    ('Alley', 'NoAlley'),
    ('Fence','NoFen'),
    ('FireplaceQu', 'NoFireP'),
    ('BsmtHalfBath', 0),
    ('BsmtFullBath', 0),
    ('Utilities', 'AllPub'),
    ('Functional', 'Typ'),
]

# Iterate over the impute_list
for column, value in Impute_list:
    
    # Impute missing values in the specified column with the above value
    df_Ames[column].fillna(value, inplace=True)


# In[19]:


#Calculate % of missing values in the dataset to see the pending rows to handle
null_df = df_Ames.isnull().sum()[df_Ames.isnull().sum()>0].sort_values().to_frame().rename(columns={0:"Total Missing values"})
null_df["% of Missing Values"] = round(null_df["Total Missing values"]/len(df_Ames)*100,2)
null_df["Feature Data Type"] = df_Ames[null_df.index.tolist()].dtypes
null_df


# In[20]:


#Fill GarageYrBlt to 0 if house doesnt have Garage
df_Ames.loc[df_Ames['GarageType'] == 'NoGrge', 'GarageYrBlt'] = 0


# In[21]:


# Remaining rows in GarageYrBlt handled based on below;

# Calculate the median of GarageYrBlt for rows where GarageType is 'Detchd'
median_garageyrblt_dtchd = df_Ames.loc[df_Ames['GarageType'] == 'Detchd', 'GarageYrBlt'].median()

# Fill missing values in 'GarageYrBlt'  with median where GarageType is 'Detchd'
df_Ames.loc[(df_Ames['GarageType'] == 'Detchd') & (df_Ames['GarageYrBlt'].isnull()), 'GarageYrBlt'] = median_garageyrblt_dtchd

# Calculate the mode of GarageFinish for rows where GarageType is 'Detchd'
mode_garagefinish_dtchd = df_Ames.loc[df_Ames['GarageType'] == 'Detchd', 'GarageFinish'].mode().iloc[0]

# Fill missing values in 'GarageFinish' where GarageType is 'Detchd' with the mode value
df_Ames.loc[(df_Ames['GarageType'] == 'Detchd') & (df_Ames['GarageFinish'].isnull()), 'GarageFinish'] = mode_garagefinish_dtchd

# Calculate the mode of GarageQual for rows where GarageType is 'Detchd'
mode_garagequal_dtchd = df_Ames.loc[df_Ames['GarageType'] == 'Detchd', 'GarageQual'].mode().iloc[0]

# Fill missing values in 'GarageQual' where GarageType is 'Detchd' with the mode value
df_Ames.loc[(df_Ames['GarageType'] == 'Detchd') & (df_Ames['GarageQual'].isnull()), 'GarageQual'] = mode_garagequal_dtchd

# Calculate the mode of GarageCond for rows where GarageType is 'Detchd'
mode_garagecond_dtchd = df_Ames.loc[df_Ames['GarageType'] == 'Detchd', 'GarageCond'].mode().iloc[0]

# Fill missing values in 'GarageCond' where GarageType is 'Detchd' with the mode value
df_Ames.loc[(df_Ames['GarageType'] == 'Detchd') & (df_Ames['GarageCond'].isnull()), 'GarageCond'] = mode_garagecond_dtchd


# In[22]:


#Calculate % of missing values in the dataset to see the pending rows to handle
null_df = df_Ames.isnull().sum()[df_Ames.isnull().sum()>0].sort_values().to_frame().rename(columns={0:"Total Missing values"})
null_df["% of Missing Values"] = round(null_df["Total Missing values"]/len(df_Ames)*100,2)
null_df["Feature Data Type"] = df_Ames[null_df.index.tolist()].dtypes
null_df


# In[23]:


#Dealing with Garage Cars 

# Calculate the median of GarageCars for rows where GarageType is 'Detchd'
median_garagecars= df_Ames.loc[df_Ames['GarageType'] == 'Detchd', 'GarageCars'].median()

# Fill missing values in 'GarageCars' where GarageType is 'Detchd' with the median value
df_Ames.loc[(df_Ames['GarageType'] == 'Detchd') & (df_Ames['GarageCars'].isnull()), 'GarageCars'] = median_garagecars

# Calculate the median of GarageArea for the rows where GarageType is 'Detchd'
median_garagearea= df_Ames.loc[df_Ames['GarageType'] == 'Detchd', 'GarageArea'].median()

# Fill missing values in 'GarageArea' where GarageType is 'Detchd' with the median value
df_Ames.loc[(df_Ames['GarageType'] == 'Detchd') & (df_Ames['GarageArea'].isnull()), 'GarageArea'] = median_garagearea


# In[24]:


#Calculate % of missing values in the dataset to see the pending rows to handle
null_df = df_Ames.isnull().sum()[df_Ames.isnull().sum()>0].sort_values().to_frame().rename(columns={0:"Total Missing values"})
null_df["% of Missing Values"] = round(null_df["Total Missing values"]/len(df_Ames)*100,2)
null_df["Feature Data Type"] = df_Ames[null_df.index.tolist()].dtypes
null_df


# In[25]:


#Fill remaining missing numerical columns with zero

df_Ames['BsmtFinType2'] = df_Ames['BsmtFinType2'].fillna(0)
#BsmtExposure
df_Ames['BsmtExposure'].fillna(df_Ames['BsmtExposure'].mode().iloc[0], inplace=True)
#BsmtQual
df_Ames['BsmtQual'].fillna(df_Ames['BsmtQual'].mode().iloc[0], inplace=True)
#BsmtCond
df_Ames['BsmtCond'].fillna(df_Ames['BsmtCond'].mode().iloc[0], inplace=True)


# In[26]:


#Checking the missing value again
print("Total Missing Values Left is:",df_Ames.isnull().sum().sum())


# # Descriptive  Data Analysis

# In[27]:


df_Ames.describe()


# # Inferential Data Analysis

# In[28]:


#Hypothesis Testing - T Test whether there is a significant difference in the average SalePrice 
#for houses with and without a garage.

# Extract SalePrice for houses with and without a garage
price_with_garage = df_Ames[df_Ames["GarageType"] != "NoGrge"]["SalePrice"]
price_without_garage = df_Ames[df_Ames["GarageType"] == "NoGrge"]["SalePrice"]

# Perform a two-sample t-test for means
t_statistic, p_value = ttest_ind(price_with_garage, price_without_garage)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in SalePrice between houses with and without a garage.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in SalePrice between houses with and without a garage.")
    
print("P-value:", p_value)
print("T-statistic:", t_statistic)


# In[29]:


#ANOVA Test to check if there is any significant difference in average sale price among neighborhood

# Group SalePrice by Neighborhood
neighborhoods = df_Ames["Neighborhood"].unique()
neighborhood_groups = [df_Ames[df_Ames["Neighborhood"] == neighborhood]["SalePrice"] for neighborhood in neighborhoods]

# Perform ANOVA
f_statistic, p_value = f_oneway(*neighborhood_groups)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in average SalePrice among different neighborhoods.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in average SalePrice among different neighborhoods.")
    
print("P-value:", p_value)
print("F-statistic:", f_statistic)


# # Exploratory data analysis (EDA)

# In[30]:


#checking the distribution of 'SalePrice'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,6))

sns.histplot(x='SalePrice', data=df_Ames, kde=True, ax=ax1)
ax1.set(ylabel = 'frequency')
ax1.set(xlabel = 'SalePrice')
ax1.set(title = 'SalePrice Distribution');

# Compute the empirical cumulative distribution function (ECDF)
sorted_prices = np.sort(df_Ames['SalePrice'])
ecdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)

# Create the ECDF plot
ax2.plot(sorted_prices, ecdf, marker='.', linestyle='none')
ax2.set(xlabel='Sales Price')
ax2.set(ylabel='ECDF')
ax2.set (title='Empirical Cumulative Distribution Function');

#skewness and kurtosis
print("Skewness: %f" % df_Ames['SalePrice'].skew())
print("Kurtosis: %f" % df_Ames['SalePrice'].kurt())


# In[31]:


#Checking the location of the houses sold
df_Ames.groupby(['Neighborhood']).Id.count().sort_values().plot(kind='bar',figsize=(10,4))
plt.title('Location of the properties')
plt.show()


# In[32]:


#Lets look at the trend in House prices

#check whether there is a relation between year the house is sold and the sales price
median_prices = df_Ames.groupby('YrSold')['SalePrice'].median()
plt.plot(median_prices.index, median_prices.values, marker='o', linestyle='-')
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs Year Sold")
plt.show()


# In[33]:


#Display Min,Max and Median house price with respect to month and year

median_prices_by_year_month = df_Ames.groupby(['YrSold', 'MoSold'])['SalePrice'].median()
max_prices_by_year_month = df_Ames.groupby(['YrSold', 'MoSold'])['SalePrice'].max()
min_prices_by_year_month = df_Ames.groupby(['YrSold', 'MoSold'])['SalePrice'].min()

year_month_strings = [f'{year}-{month:02d}' for year, month in median_prices_by_year_month.index]
plt.figure(figsize=(15, 8))

plt.plot(year_month_strings, median_prices_by_year_month.values, marker='o', linestyle='-', color='#8A2BE2', label='Median Price')
plt.plot(year_month_strings, max_prices_by_year_month.values, marker='x', linestyle='-', color='#FF5733', label='Max Price')
plt.plot(year_month_strings, min_prices_by_year_month.values, marker='x', linestyle='-', color='g', label='Min Price')
plt.xlabel('Year Sold - Month Sold')
plt.ylabel('House Price')
plt.title("House Price vs Year Sold - Month Sold")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout() 
plt.show()


# In[34]:


#Visualise numerical features distribution in the dataset
numerical_attributes = df_Ames.select_dtypes(include=['int64', 'float64'])
num_features = numerical_attributes.shape[1]
num_rows = (num_features - 1) // 4 + 1
num_cols = min(num_features, 4)

plt.figure(figsize=(25, 5 * num_rows))
for index, column in enumerate(numerical_attributes.columns):
    plt.subplot(num_rows, num_cols, index + 1)
    sns.histplot(df_Ames[column], bins=10, kde=True)
    plt.title(f"{column} Distribution", fontweight="black", size=20, pad=10)
    plt.tight_layout()

plt.savefig('numerical_features_distribution.jpg')
plt.show()


# In[35]:


#Visualizing Categorical Features Vs SalePrice.
cat_cols = df_Ames.select_dtypes(include="object").columns.tolist()
def boxplot(col_list):
    plt.figure(figsize=(22,12))
    for index,column in enumerate(col_list):
        plt.subplot(2,4,index+1)
        sns.boxplot(x=column, y="SalePrice", data=df_Ames)
        plt.title(f"{column} vs SalePrice",fontweight="black",pad=10,size=20)
        plt.xticks(rotation=90)
        plt.tight_layout()


# In[36]:


boxplot(cat_cols[0:8])


# In[37]:


boxplot(cat_cols[8:16])


# In[38]:


boxplot(cat_cols[16:24])


# In[39]:


boxplot(cat_cols[24:32])


# In[40]:


plt.figure(figsize=(22,12))
for index,column in enumerate(cat_cols[32:]):
    plt.subplot(4,4,index+1)
    sns.boxplot(x=column, y="SalePrice", data=df_Ames)
    plt.title(f"{column} vs SalePrice",fontweight="black",pad=10,size=20)
    plt.xticks(rotation=90)
    plt.tight_layout()


# In[41]:


#Dropping Street ,Utilities & RoofMtl Since it has lot of imbalances as seen in categorical visualisation
df_Ames.drop(columns=["Utilities","Street"],inplace=True)
df_Ames.drop(columns=["RoofMatl"],inplace=True)


# In[42]:


#Visualizing Discrete Numerical Features Vs Average "SalePrice"
dis_cols = ["OverallQual","OverallCond","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr",
            "KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","MoSold"]

plt.figure(figsize=(22,14))
for index,column in enumerate(dis_cols):
    data = df_Ames.groupby(column)["SalePrice"].mean()
    plt.subplot(3,4,index+1)
    sns.barplot(x=data.index,y= data)
    plt.title(f"{column} vs Avg. Sale Price",fontweight="black",size=15,pad=10)
    plt.tight_layout()


# # Outlier analysis

# In[43]:


#scatter plot GrLivArea/SalePrice
#Houses with Large living area where sold at lower price, so removed houses which has living area more than 4000SF
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_Ames)
plt.title('Living Area vs SalePrice');


# In[44]:


#scatter plot TotalBsmtSF/SalePrice
#Remove records having TotalBsmtSF >4000SF
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df_Ames)
plt.title('Total Basement Area vs SalePrice');


# In[45]:


# Dropping outliers after analysing the scatter plots

df_Ames = df_Ames.drop(df_Ames[(df_Ames['TotalBsmtSF']>=4000) & (df_Ames['SalePrice']<300000)].index)

df_Ames = df_Ames.drop(df_Ames[(df_Ames['GrLivArea']>4000) & (df_Ames['SalePrice']<300000)].index)


# # Feature Engineering

# In[46]:


df_Ames['Remodeled'] = (df_Ames['YearBuilt'] != df_Ames['YearRemodAdd']).astype(int)


# In[47]:


df_Ames['Age'] = df_Ames['YrSold'] - df_Ames['YearBuilt']


# In[48]:


#Create bathroom features
df_Ames["Total_BathRMs"] = (df_Ames["FullBath"] + (0.5 * df_Ames["HalfBath"]) + 
                               df_Ames["BsmtFullBath"] + (0.5 * df_Ames["BsmtHalfBath"]))


# In[49]:


#Create porch features
df_Ames['Total_Porch_SF'] = (df_Ames['OpenPorchSF'] + df_Ames['3SsnPorch'] +df_Ames['EnclosedPorch'] +
                              df_Ames['ScreenPorch'] + df_Ames['WoodDeckSF'])


# In[50]:


#delete porch related columns since we have total porch square feet
cols = ["OpenPorchSF","3SsnPorch","EnclosedPorch","ScreenPorch","WoodDeckSF"]
df_Ames.drop(columns=cols,inplace=True)


# In[51]:


# Creating total sqaure footage
df_Ames['Total_SF_Footage']=(df_Ames['BsmtFinSF1']+df_Ames['BsmtFinSF2']+df_Ames['1stFlrSF']+df_Ames['2ndFlrSF'])


# In[52]:


df_Ames['TotalBsmtFinSF'] = df_Ames['BsmtFinSF1'] + df_Ames['BsmtFinSF2']


# In[53]:


#dropped BsmtFinSF1&BsmtFinSF2

cols = ["BsmtFinSF2","BsmtFinSF1"]
df_Ames.drop(columns=cols,inplace=True)


# In[54]:


df_Ames['YearsSinceRemodel'] = df_Ames['YrSold'] - df_Ames['YearRemodAdd']


# In[55]:


df_Ames['BedroomToRoomsRatio'] = df_Ames['BedroomAbvGr'] / df_Ames['TotRmsAbvGrd']


# In[56]:


df_Ames['BathRoomToRoomsRatio'] = df_Ames['Total_BathRMs'] / df_Ames['TotRmsAbvGrd']


# In[57]:


df_Ames['PricePerSF'] = df_Ames['SalePrice'] / df_Ames['GrLivArea']

df_Ames['PricePerRoom'] = df_Ames['SalePrice'] / df_Ames['TotRmsAbvGrd']

df_Ames['TotalArea'] = df_Ames['LotArea'] + df_Ames['TotalBsmtSF'] + df_Ames['GrLivArea'] + df_Ames['GarageArea']

df_Ames['GarageAge'] = df_Ames['YrSold'] - df_Ames['GarageYrBlt']


# In[58]:


#Create new feature for condition by combining condition 1 and condition 2
#Norm means normal which indicates there's no second condition, hence Normal in condition 2 replaced with empty string
Proximity_condition = []
df_Ames["Condition2"] = df_Ames["Condition2"].replace({"Norm": ""})

for val1, val2 in zip(df_Ames["Condition1"], df_Ames["Condition2"]):
    if val2 == "":
        Proximity_condition.append(val1)
    elif val1 == val2:
        Proximity_condition.append(val1)
    else:
        Proximity_condition.append(val1 + val2)

df_Ames["Proximity_condition"] = Proximity_condition


# In[59]:


df_Ames.drop(columns=["Condition1","Condition2"],inplace=True)


# In[60]:


df_Ames["HeatingQuality"] = df_Ames["Heating"] + "-" + df_Ames["HeatingQC"]


# In[61]:


df_Ames.drop(columns=["Heating","HeatingQC"],inplace=True)


# In[62]:


#Create boolean features
def boolean_feature(df):
    df["Has2ndFloor"] = (df['2ndFlrSF'] != 0).astype(int)
    df["HasGarage"]  = (df["GarageArea"] !=0).astype(int)
    df["HasBsmt"]    = (df["TotalBsmtSF"]!=0).astype(int)
    df["HasFirePlace"] = (df["Fireplaces"]!=0).astype(int) 
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFence'] = (df['Fence'] != 'NoFen').astype(int)
    df['HasMiscFeature'] = (df['MiscFeature'] != 'None').astype(int)
    df["HasPorch"]=(df["Total_Porch_SF"]!=0).astype(int)
    df["Has_Normal_Proximity"] = (df["Proximity_condition"] == "Norm").astype(int)


# In[63]:


boolean_feature(df_Ames)


# In[64]:


#Visualisng all these boolean features
plt.figure(figsize=(22,15))
for index,column in enumerate(["Has2ndFloor","HasGarage","HasBsmt","HasFirePlace","HasPool",
                               "HasPorch","HasFence","HasMiscFeature","Has_Normal_Proximity","Remodeled"]):
    plt.subplot(3,4,index+1)
    sns.boxenplot(x=column, y="SalePrice", data=df_Ames, palette="Set3")
    plt.title(f"{column} vs SalePrice",pad=15,size=20)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('Booleanfeatures.jpg')


# In[65]:


#Display only numerical variables 
numerical_features = df_Ames.select_dtypes(include=['int64', 'float64'])
numerical_features.info()



# In[66]:


#Dropping Id as it is no longer needed
df_Ames.drop(columns="Id",inplace=True)


# In[67]:


# Visualising new features created

New_cols = ["Age", "Total_BathRMs", "Total_Porch_SF", "Total_SF_Footage",
            "TotalBsmtFinSF", "YearsSinceRemodel", "BedroomToRoomsRatio", "BathRoomToRoomsRatio",
            "PricePerSF", "PricePerRoom", "TotalArea", "GarageAge"]

target_variable = 'SalePrice'
num_cols = len(New_cols)
num_rows = math.ceil(num_cols / 4) 
fig, axes = plt.subplots(num_rows, 4, figsize=(20, 14))  

for index, column in enumerate(New_cols):
    row = index // 4
    col = index % 4
    ax = axes[row, col]
    
    data = df_Ames.groupby(column)[target_variable].mean()
    ax.scatter(x=data.index, y=data)
    ax.set_title(f"{column} vs Avg. Sale Price", fontweight="black", size=15, pad=10)
    
# Remove any remaining empty subplots
for i in range(num_cols, num_rows * 4):
    ax = axes.flatten()[i]
    ax.axis('off')

plt.tight_layout()
plt.savefig('Newfeatures.jpg')
plt.show()


# # Feature Selection

# In[68]:


#Correlation Analysis
# Display numerical correlations (pearson) between features on heatmap.
Numeric_features=df_Ames.select_dtypes(include=[np.number])
sns.set(font_scale=1.1)
correlation= Numeric_features.corr()
mask = np.triu(correlation.corr())
plt.figure(figsize=(20, 20))
heatmap=sns.heatmap(correlation,
            annot=True,
            fmt='.2f',
            cmap='PuBu',
            square=True,
            mask=mask,
            linewidths=1,
            cbar=False)

plt.show()
heatmap.get_figure().savefig('heatmap_correlation.png',bbox_inches='tight')


# In[69]:


# Highlight features which has correlation greater than 0.8
plt.subplots(figsize=(20, 20))
sns.heatmap(correlation>=0.8, annot=True, square=True, cmap="PuBu",  mask=mask,linewidth='.1')


# In[70]:


#Visualisation of highly correlated features with Sale Price
cols = ["TotRmsAbvGrd", "GrLivArea", "GarageArea", "GarageCars","TotalArea","LotArea","Has2ndFloor","2ndFlrSF",
        "HasGarage","GarageYrBlt","HasFirePlace","Fireplaces","HasPool","PoolArea"]

plt.figure(figsize=(22, 12))
for index, column in enumerate(cols):
    plt.subplot(4, 4, index + 1)
    sns.regplot(x=df_Ames[column], y=df_Ames["SalePrice"], scatter_kws={'s': 70, 'alpha': 0.5}, 
                line_kws={'color': 'black', 'lw': 3})
    corr = round(df_Ames[[column, "SalePrice"]].corr()["SalePrice"][0], 2)
    plt.title(f"{column}\nCorrelation value with SalePrice: {corr}", pad=10, size=15)
    plt.tight_layout()
plt.savefig('correlationwithSalePrice.jpg',bbox_inches='tight')
plt.show()


# In[71]:


#Dropping columns as part of correlation Matrix
columns_to_drop = ["GarageArea", "TotRmsAbvGrd","LotArea","HasGarage","HasFirePlace","Has2ndFloor","HasPool"]
df_Ames.drop(columns=columns_to_drop, inplace=True)


# # Feature Transformation

# In[72]:


#Display the skewness of the target variable.

skewness_saleprice = df_Ames['SalePrice'].skew()

print("Skewness of 'SalePrice':", skewness_saleprice)


# In[73]:


#Since the SalePrice is positively skewed, apply Box-Cox transformation
transformed_data, lambda_param = stats.boxcox(df_Ames['SalePrice'])
df_Ames['SalePrice'] = transformed_data
df_Ames['SalePrice'].skew()


# In[74]:


#Plot sales price distribution graph after box-cox transformation
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
sns.histplot(df_Ames["SalePrice"],kde=True)
plt.title("SalePrice after  transformation ",size=20,pad=10)


# In[75]:


#checkeing skewness of numerical variables
numerical_features = df_Ames.select_dtypes(include=['int64', 'float64'])
numerical_features.skew().sort_values()


# In[76]:


skewness = numerical_features.skew().sort_values()

plt.figure(figsize=(18,8))
sns.barplot(x=skewness.index, y=skewness)
for i, v in enumerate(skewness):
    plt.text(i, v, f"{v:.1f}",size=15)

plt.ylabel("Skewness")
plt.xlabel("features")
plt.xticks(rotation=90)
plt.title("Skewness of Numerical features",fontweight="black",size=15,pad=10)
plt.savefig('SkewnessNumerical.jpg',bbox_inches='tight')
plt.tight_layout()
plt.show()


# In[77]:


#dropping features exhibiting high skewness
cols = ["LowQualFinSF","PoolArea","MiscVal"]
df_Ames.drop(columns=cols, inplace=True)


# In[78]:


df_Ames.head(10)


# In[79]:


#Performing Target Encoding on Categorical Features with High Cardinality

cols = ["Neighborhood", "Exterior1st", "Exterior2nd", "HeatingQuality"]

for column in cols:
    encoder = TargetEncoder(cols=[column], min_samples_leaf=20, smoothing=10)
    df_Ames[column] = encoder.fit_transform(df_Ames[column], df_Ames["SalePrice"])

# Display the resulting encoded DataFrame
print(df_Ames[cols])


# In[80]:


#Performing Label Encoding on below Features

cols = ["HouseStyle","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","Electrical","KitchenQual",
        "GarageQual","GarageCond","FireplaceQu","Proximity_condition"]

encoder = LabelEncoder()

df_Ames[cols] = df_Ames[cols].apply(encoder.fit_transform)


# In[81]:


#Apply  One-Hot Encoding on Nominal Categorical Columns
cols = df_Ames.select_dtypes(include="object").columns

df_Ames_copy=df_Ames

# Apply one-hot encoding to the categorical columns
df_encoded = pd.get_dummies(df_Ames, columns=cols, prefix=cols, prefix_sep='_')


# In[82]:


df_Ames_copy.info()


# In[83]:


df_encoded.shape


# In[84]:


df_Ames=df_encoded


# In[85]:


X = df_Ames.drop(columns=["SalePrice"])
y = df_Ames["SalePrice"]


# In[86]:


# Use MinMaxScaler to scale the features (X) only
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[87]:


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=50)


# In[88]:


print("Dimension of x_train:=>",x_train.shape)
print("Dimension of x_test:=>",x_test.shape)
print("Dimension of y_train:=>",y_train.shape)
print("Dimension of y_test:=>",y_test.shape)


# # Modeling

# In[89]:


# Create List of Model metrics
R2_value = []
MAE_value = []
MSE_value = []
RMSE_value = []


# In[90]:


#Apply-Model function for learning and prediction
def apply_model(model):
    model.fit(x_train, y_train)
    y_train_pred= model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Metrics Calculation.
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    MAE_value.append(mae)
    MSE_value.append(mse)
    RMSE_value.append(rmse)
    R2_value.append(r2)

    print(f"R2 Score of the {model} model is=>",r2)
    print(f"MAE of {model} model is=>",mae)
    print(f"MSE of {model} model is=>",mse)
    print(f"RMSE of {model} model is=>",rmse)

    # Scatter plot of true values Vs actual values.
    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, color='black', label='Train')
    plt.scatter(y_test, y_test_pred, color='blue', label='Test')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.legend()
    plt.title('True values Vs Predicted values', size=20, pad=10)
    plt.show()


# In[91]:


apply_model(LinearRegression())


# In[92]:


apply_model(RandomForestRegressor())


# In[93]:


from sklearn.tree import DecisionTreeRegressor
apply_model(DecisionTreeRegressor())


# In[94]:


apply_model(SVR())


# In[95]:


apply_model(AdaBoostRegressor())


# In[96]:


apply_model(CatBoostRegressor(verbose=False))


# # Summary of the Models

# In[97]:


Models = [" Multiple Linear Regression","Random Forest Regresor","Decision Tree Regressor","Support Vector Regressor",
         "AdaBoostRegressor","CatBoostRegressor"]

results_df = pd.DataFrame({"Model":Models,"R2_Score":R2_value,
                       "MAE":MAE_value,"MSE":MSE_value,"RMSE":RMSE_value})
results_df


# In[98]:


# Create a line graph for each metric
plt.figure(figsize=(10, 6))

plt.plot(results_df["Model"], results_df["R2_Score"], marker='o', label='R2 Score')
plt.plot(results_df["Model"], results_df["MAE"], marker='o', label='MAE')
plt.plot(results_df["Model"], results_df["MSE"], marker='o', label='MSE')
plt.plot(results_df["Model"], results_df["RMSE"], marker='o', label='RMSE')

plt.xlabel("Models")
plt.ylabel("Metric Value")
plt.title("Model Evaluation Metrics")
plt.xticks(rotation=45)
plt.legend()
plt.savefig('ModelEvaluationResults.jpg',bbox_inches='tight')
plt.tight_layout()
plt.show()



# In[99]:


# Plot the most correlated variables to Sale Price as a matrix
Numeric_features=df_Ames.select_dtypes(include=[np.number])
correlation= Numeric_features.corr()
imp_ftr = correlation['SalePrice'].sort_values(ascending=False).head(35).to_frame()
plt.subplots(figsize=(5,8))
plt.title('SalePrice Correlation Matrix')
sns.heatmap(imp_ftr, vmax=0.9, annot=True, fmt='.2f', cmap="PuBu", linewidth='.1')


# In[ ]:




