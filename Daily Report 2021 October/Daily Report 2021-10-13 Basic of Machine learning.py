#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[3]:


np.random.seed(1013)
x=np.arange(1, 30, 1, dtype=np.int16)

y=2*x+1
y_random = y+np.random.normal(loc=0,scale=8,size=29)


# In[4]:


len(y)


# In[5]:


plt.plot(x,y,".",markersize=20)
plt.plot(x,y_random,".", markersize=20)
plt.show()


# #Parameter estimation
# 

# In[9]:


Sxy = (x-np.mean(x))*(y_random-np.mean(y))
Sxx = (x-np.mean(x))**2

Sxy = Sxy.sum()
Sxx = Sxx.sum()

beta_1 = Sxy/Sxx
beta_0=np.mean(y_random)-beta_1*np.mean(x)
print("beta_1 = {}".format(beta_1.round(3)))
print("beta_0 = {}".format(beta_0.round(3)))


# In[10]:


#추정량

y_hat = beta_1*x+beta_0
print(y_hat)


# In[12]:


plt.plot(x,y,".",markersize=20)
plt.plot(x, y_random,".",markersize=20)
plt.plot(x,y_hat,"-",markersize=10,color="r")
plt.show()


# In[14]:


## sklearn import
from sklearn.linear_model import LinearRegression


# In[20]:


lr=LinearRegression()
x_2d=x.reshape(-1,1)

lr.fit(x_2d, y_random)
print(lr.coef_, lr.intercept_)


# In[22]:


print("x = ",np.shape(x))
print("x_2d = ",np.shape(x_2d))


# In[ ]:




