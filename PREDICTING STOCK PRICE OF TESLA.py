#!/usr/bin/env python
# coding: utf-8

# # PREDICTING STOCK PRICE OF TESLA BY APPLYING A REGRESSION ALGORITHM

# Stock price analysis has been a critical area of research and is one of the top applications of machine learning. Stock Price Prediction using machine learning helps to discover what the future value of company's stock and other financial assets traded on an exchange look like. The main idea behind stock price prediction is to obtain a picture of how the stocks will perform, and how to maximize the prospects of that company's stocks. Predicting how the stock market will perform is a hard task to do. 
# 
# There are many factors one must consider when predicting stock prices. Factors such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. 
# 
# In this project, I am developing and evaluating the performance and the predictive power of a model trained and tested on data collected from Tesla Stock Prices.

# In[55]:


#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# In[5]:


Tesla_df = pd.read_csv('TSLA.csv')
Tesla_df.head()


# In[6]:


Tesla_df.shape


# In[7]:


Tesla_df.describe()


# In[6]:


Tesla_df.isnull().sum()


# In[8]:


corr = Tesla_df.corr(method= 'pearson')
corr


# In[9]:


sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,
 cmap='RdBu_r', annot=True, linewidth=0.5)


# In[45]:


#Visualizing the Adjusted closing price history of Tesla Stocks
plt.figure(figsize=(16,8))
plt.title('Tesla Stock Closing Price History')
plt.plot(Tesla_df_new['Adj Close'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('Stock Close Price $', fontsize=16)
plt.show()


# #### Visualize the Dependent variable with Independent Features

# In[20]:


#Plot 
Tesla_df[['Close','Open','Adj Close','High','Low']].head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# #### Model Training and Testing

# In[24]:


Tesla_df['Date'] = pd.to_datetime(Tesla_df['Date'], errors='coerce')
Tesla_df['Year']=Tesla_df['Date'].dt.year
Tesla_df['Month']=Tesla_df['Date'].dt.month
Tesla_df['Day']=Tesla_df['Date'].dt.day


# In[26]:


Tesla_df_new=Tesla_df[['Day','Month','Year','High','Open','Low','Close']]
Tesla_df_new.head(10)


# In[29]:


#separate Independent and dependent variable
X = Tesla_df_new.iloc[:,Tesla_df_new.columns !='Close']
Y= Tesla_df_new.iloc[:, 5]
print(X.shape) 
print(Y.shape)


# In[33]:


from sklearn.model_selection import train_test_split
# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=.3)
#x_train,x_test,y_train,y_test= train_test_split
# Test set
print(X_test.shape)

# Training set
print(X_train.shape)


# In[56]:


from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()
lin_model.fit(X_train,y_train)


# In[47]:


# Use model to make predictions
y_pred=model.predict(X_test)


# In[57]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Printout relevant metrics
print("Model Coefficients:", lin_model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))


# In[58]:


from sklearn import model_selection
from sklearn.model_selection import KFold
kfold = model_selection.KFold(n_splits=20, shuffle=True)
results_kfold = model_selection.cross_val_score(lin_model, X_test, y_test.astype('int'), cv=kfold)
print("Accuracy: ", results_kfold.mean()*100)


# In[59]:


plot_df=pd.DataFrame({'Actual':y_test,'Pred':y_pred})
plot_df.head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




