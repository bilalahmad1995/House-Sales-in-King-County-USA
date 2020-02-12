#!/usr/bin/env python
# coding: utf-8

# <h2 align='center'> Peer-graded Assignment: House Sales in King County, USA</h2>

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('kc_house_data.csv')


# Question 1) Display the data types of each column using the attribute dtype, then take a screenshot and submit it, include your code in the image.

# In[5]:


df.dtypes


# Question 2) Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to True. 

# In[8]:


df.drop('id', axis=1, inplace=True)
df.describe()
#there is no column named 'Unnamed: 0 in the dataset I downloaded'


# Question 3) use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.

# In[12]:


df['floors'].value_counts().to_frame()


# Question 4) use the function boxplot in the seaborn library to produce a plot that can be used to determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# In[15]:


sns.boxplot('waterfront', 'price', data=df)
plt.show()


# Question 4) Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.

# In[17]:


sns.regplot('sqft_above', 'price', data=df)
plt.show()


# In[18]:


df[['sqft_above', 'price']].corr()


# Fit a linear regression model to predict the price using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2.

# In[19]:


from sklearn.linear_model import LinearRegression


# In[21]:


LM = LinearRegression()


# In[29]:


LM.fit(df[['sqft_living']], df['price'])


# In[30]:


yhat = LM.predict(df[['sqft_living']])
yhat[0:6]


# In[31]:


print('The value of R^2 is: ', LM.score(df[['sqft_living']], df['price']))


# Fit a linear regression model to predict the 'price' using the list of features:
# "floors"
# "waterfront"
# "lat"
# "bedrooms"
# "sqft_basement"
# "view"
# "bathrooms"
# "sqft_living15"
# "sqft_above"
# "grade"
# "sqft_living"

# In[34]:


X = df[['floors', 
        'waterfront', 
        'lat',
        'bedrooms',
        'sqft_basement',
        'view',
        'bathrooms',
        'sqft_living15',
        'sqft_above',
        'grade',
        'sqft_living']]
Y = df['price']


# In[35]:


LM.fit(X, Y)


# In[36]:


Yhat = LM.predict(X)


# In[37]:


print('The value of R^2 is: ', LM.score(X, Y))


# Create a pipeline object that scales the data performs a polynomial transform and fits a linear regression model. Fit the object using the features in the question above, then fit the model and calculate the R^2

# In[39]:


from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# In[42]:


#Creating a pipeline object
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
Pipe = Pipeline(Input)


# In[53]:


# For Polynomial transformation
PF = PolynomialFeatures()
_yhat = PF.fit_transform(X, Y)


# In[59]:


#Fitting the object using the features in in question above
Pipe.fit(X, Y)


# In[64]:


#Fitting the model and calculate the R^2
_Yhat = LM.fit(X, Y)
print('The value of R^2 is: ', Pipe.score(X, Y))


# Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1 and calculate the R^2 using the test data.

# In[96]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge


# In[90]:


X = df.drop(['price', 'date'], axis=1)
Y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[91]:


RigeModel = Ridge(alpha=0.1)


# In[92]:


RigeModel.fit(x_train, y_train)


# In[98]:


print('The value of R^2 using the test data is: ', RigeModel.score(x_test, y_test))


# Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, setting the regularisation parameter to 0.1. Calculate the R^2 utilising the test data provided

# In[94]:


PR = PolynomialFeatures(degree=2)


# In[101]:


X = df.drop(['price', 'date'], axis=1)
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

train_pr = PR.fit_transform(x_train, y_train)
test_pr = PR.fit_transform(x_test, y_test)

RigeModel2 = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)


# In[102]:


print('The value of R^2 using the test data is: ', RigeModel.score(x_test, y_test))


# In[ ]:




