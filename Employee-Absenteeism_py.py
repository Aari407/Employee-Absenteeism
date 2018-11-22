
# coding: utf-8

# In[396]:


import os
from fancyimpute import KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


# In[397]:


os.chdir("C:/Project")#setting working directory
os.getcwd()#checking the current working directory


# In[398]:


xl=pd.ExcelFile('Absenteeism_at_work_Project.xls', sep=',')#Importing excel file
print(xl.sheet_names)#printing the sheetnames in excel file
df_train=xl.parse(0)#Parsing the first sheetname 'Absenteeism_at_work' to a dataframe


# In[399]:


df_train.info()


# In[400]:


df_train.head()#taking a look at first 5 rows of dataframe


# In[401]:


#Dividing the columns into categorical and numerical data
cat_list=['ID', 'Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 'Disciplinary failure', 'Education', 'Social drinker', 'Social smoker', 'Pet', 'Son']
num_list=['Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']


# In[402]:


#missing value
missing_val = pd.DataFrame(df_train.isnull().sum()).reset_index()#creating data frame with missing values of each column
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})#renaming the column
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(df_train))*100#calculating the percentage of missing values
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)#sorting in descending order


# In[403]:


#df_train['Distance from Residence to Work'].iloc[6]=np.nan
# Actual value = 52

#df_train['Distance from Residence to Work'] = df_train['Distance from Residence to Work'].fillna(df_train['Distance from Residence to Work'].mean())
#Mean method: 29.637228260869566

#df_train['Distance from Residence to Work'] = df_train['Distance from Residence to Work'].fillna(df_train['Distance from Residence to Work'].median())
#Median method: 26.0

#Knn imputation
#KNN method: 50.98273352398267
df_train=pd.DataFrame(KNN(k=3).complete(df_train), columns=df_train.columns)


# In[404]:


#Rounding the values
cols_name=df_train.columns.tolist()
df_train[cols_name]=df_train[cols_name].apply(lambda x: round(x))


# In[405]:


df_train.info()


# In[406]:


#Outlier Analysis
#Box pplo for all numerical variables
for i in num_list:
    plt.figure(figsize=(5,5))
    plt.boxplot(df_train[i])
    plt.xlabel(i)
    plt.show()


# In[407]:


#Replacing outliers with NAN and imputing with KNN
for i in num_list:
    q75, q25= np.percentile(df_train.loc[:,i], [75, 25])
    iqr=q75-q25
    min=q25 - (iqr*1.5)
    max=q75 + (iqr*1.5)
    df_train.loc[df_train[i]< min,i] = np.nan
    df_train.loc[df_train[i]> max,i] = np.nan
df_train=pd.DataFrame(KNN(k=3).complete(df_train), columns=df_train.columns)


# In[408]:


#Visual EDA
#Univariate analysis
for i in cat_list:
    plt.figure(figsize=(15,5))
    sns.countplot(x=i, data=df_train)
    plt.show()


# In[409]:


#Bivariate analysis for categorical columns
for i in cat_list:
    plt.figure(figsize=(10, 5))
    df_train.groupby(i)['Absenteeism time in hours'].sum().plot(kind='bar')
    plt.show()


# In[410]:


#Bivariate analysis for numerical columns
for i in num_list:
    df_train.groupby(i)['Absenteeism time in hours'].sum().plot(kind='bar')
    plt.show()


# In[411]:


#Normality check
for i in num_list:
    sns.distplot(df_train[i])
    plt.show()


# In[412]:


#Feature Scaling using Normalization
for i in num_list:
    df_train[i]= (df_train[i]-np.min(df_train[i]))/(np.max(df_train[i])-np.min(df_train[i]))


# In[413]:


#Featuere selction
f, ax = plt.subplots(figsize=(15, 15))
corr_matrix = df_train.corr()

#Plot using seaborn library
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 50, as_cmap=True),
            square=True, ax=ax, annot = True)
plt.plot()

df_train=df_train.drop('Weight', axis=1)


# In[414]:


#getting dummies
df_train_cat = pd.get_dummies(data = df_train, columns = cat_list)
df1 = df_train_cat.copy()


# In[415]:


#X_train, X_test, y_train, y_test = train_test_split( df_train_cat.iloc[:, df_train_cat.columns != 'Absenteeism time in hours'], df_train_cat['Absenteeism time in hours'], test_size = 0.20)


# In[416]:


#Decision Tree
#model= DecisionTreeRegressor(max_depth = 2, random_state=25)#initiate model
#model.fit(X_train, y_train)#fit to train data
#pred= model.predict(X_test)#predict on test data
#rmse=np.sqrt(mean_squared_error(y_test, pred))#Root mean square error
#print("Root Mean Squared Error For Decision Tree= "+str(rmse))


# In[417]:


# Random forest
#model_rf = RandomForestRegressor(n_estimators = 500, random_state=2)#Intiate model
#model_rf.fit(X_train, y_train)#fit on training ata
#pred_rf = model_rf.predict(X_test)#predict on test data
#rmse_rf = np.sqrt(mean_squared_error(y_test,pred_rf))#Root mean square error
#print("Root Mean Squared Error For Random Forest = "+str(rmse_rf))


# In[418]:


#Linear Regression
#model_lr = LinearRegression()#Initiate model
#model_lr.fit(X_train , y_train)#fit on training data
#pred_lr = model_lr.predict(X_test)#predict on test data
#rmse_lr =np.sqrt(mean_squared_error(y_test,pred_lr))#Root mean square eror
#print("Root Mean Squared Error For Linear Regression = "+str(rmse_lr))


# In[419]:


#PCA
target = df_train_cat['Absenteeism time in hours']
df_train_cat.drop(['Absenteeism time in hours'], inplace = True, axis=1)
df_train_cat.shape


# In[420]:


# Converting data to numpy array
features=df1.values

# Data has 116 variables so no of components of PCA = 116
pca = PCA(n_components=116)
pca.fit(features)

# The amount of variance that each PC explains
var= pca.explained_variance_ratio_

# Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()


# In[421]:


pca = PCA(n_components=50)
pca.fit(features)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.20)


# In[422]:


#Decision Tree
np.random.seed(121)
model= DecisionTreeRegressor(max_depth = 2, random_state=25)#initiate model
model.fit(features_train, target_train)#fit to train data
pred= model.predict(features_test)#predict on test data
rmse=np.sqrt(mean_squared_error(target_test, pred))#Root mean square error
print("Root Mean Squared Error For Decision Tree= "+str(rmse))
print("R^2 Score = "+str(r2_score(target_test,pred)))


# In[423]:


#Root Mean Squared Error For Decision Tree= 0.04419942802838929
#R^2 Score = 0.9536500781893115


# In[424]:


# Random forest
model_rf = RandomForestRegressor(n_estimators = 500, random_state=2)#Intiate model
model_rf.fit(features_train, target_train)#fit on training ata
pred_rf = model_rf.predict(features_test)#predict on test data
rmse_rf = np.sqrt(mean_squared_error(target_test,pred_rf))#Root mean square error

print("Root Mean Squared Error For Random Forest = "+str(rmse_rf))
print("R^2 Score = "+str(r2_score(target_test,pred_rf)))


# In[425]:


#Root Mean Squared Error For Random Forest = 0.007627809966417199
#R^2 Score = 0.9986195666691634


# In[426]:


#Linear Regression
model_lr = LinearRegression()#Initiate model
model_lr.fit(features_train , target_train)#fit on training data
pred_lr = model_lr.predict(features_test)#predict on test data
rmse_lr =np.sqrt(mean_squared_error(target_test,pred_lr))#Root mean square eror
print("Root Mean Squared Error For Linear Regression = "+str(rmse_lr))
print("R^2 Score = "+str(r2_score(target_test,pred_lr)))


# In[427]:


#Root Mean Squared Error For Linear Regression = 0.00947782685153262
#R^2 Score = 0.9978687556351598

