#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )


# In[3]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
perch_length, perch_weight)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)


# In[4]:


from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors =3)
knr.fit(train_input, train_target)


# In[5]:


print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))


# In[6]:


knr.predict([[50]])


# In[7]:


dist, indx = knr.kneighbors([[50]])
print(indx)


# In[8]:


plt.scatter(train_input, train_target)
plt.scatter(50, 1033, marker="D")
plt.scatter(train_input[indx],train_input[indx],marker='^', s=100)
plt.show()


# In[9]:


#선형회귀
from sklearn.linear_model import LinearRegression


# In[10]:


lr = LinearRegression()
lr.fit(train_input, train_target)


# In[11]:


print(lr.score(train_input, train_target))   #연속형이면 R제곱값
print(lr.score(test_input, test_target))


# In[12]:


lr.predict([[50]])


# In[13]:


print(lr.coef_,lr.intercept_)


# In[14]:


x_new = np.arange(12, 60)
y_new = x_new*lr.coef_+lr.intercept_

plt.scatter(train_input, train_target)
plt.scatter(50, 1177, marker='D')
plt.plot(x_new,y_new,marker='_')

plt.show()


# In[15]:


print(train_input[:10])
print(train_input.shape)


# In[16]:


train_poly = np.column_stack((train_input,train_input**2))
train_poly[:10]
test_poly =np.column_stack((test_input, test_input**2))
test_poly[:10]


# In[17]:


lrp =LinearRegression()
lrp.fit(train_poly, train_target)


# In[18]:


print(lrp.coef_,lrp.intercept_) #왼쪽부터 세타1 세타2 세타0, 세타2는 x의 제곱임


# ### 위의 것을 식으로 세우면 y=-23.4x+1.03x^2+145.16 이 된다

# In[19]:


print(lrp.score(train_poly, train_target))
print(lrp.score(test_poly, test_target))


# In[20]:


x_new = np.arange(10,50)
y_new = lrp.intercept_+lrp.coef_[0]*x_new +lrp.coef_[1]*x_new**2

plt.scatter(train_input, train_target)
plt.plot(x_new,y_new)
plt.show()


# In[21]:


import pandas as pd


# In[22]:


df = pd.read_csv('https://bit.ly/perch_csv')
print(type(df))
#df.head()
#df.shape
perch_full = df.to_numpy()
print(type(perch_full))
print(np.shape(perch_full))
print(perch_full)


# In[23]:


perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


# In[24]:


train_input, test_input, train_target, test_target = train_test_split(
perch_full, perch_weight, random_state=42)


# In[25]:


#polynomial transform
from sklearn.preprocessing import PolynomialFeatures


# In[26]:


poly = PolynomialFeatures(degree = 3) #3차항으로 지정
poly.fit([[2, 3]])
poly.transform([[2,3]])


# In[27]:


poly = PolynomialFeatures(degree = 2, include_bias=False)

poly.fit(train_input)

train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)


# In[28]:


#print(train_poly[:10])

poly.get_feature_names() #위의 데이터를 기준으로 만들어진 다항식의 형태


# In[29]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(train_poly,  train_target)


# In[30]:


print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))


# In[31]:


poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)

train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)


# In[ ]:





# In[32]:


lr=LinearRegression()
lr.fit(train_poly,train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))


# In[33]:


#Ridge regression
from sklearn.preprocessing import StandardScaler

ss= StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)


# In[34]:


from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)


# In[35]:


print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))


# In[36]:


# Ridge lambda plot
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha = alpha)  #alpha_list에 있는 요소들을 alpha에 넣는다
    ridge.fit(train_scaled, train_target)
    
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))


# In[37]:


print(train_score)
print(test_score)


# In[38]:


plt.plot(np.log10(alpha_list), train_score, label="Train")
plt.plot(np.log10(alpha_list), test_score, label='Test')
plt.xlabel("log10(alpha)")
plt.ylabel("R^2")
plt.legend()
plt.show()


# ### 위의 경우 -1이 최적의 람다 값인데 -1=log10(10)이므로 log10x=-1이고 즉 x=0.1로 0.1의 람다가 최적이다

# In[39]:


from sklearn.linear_model import Lasso


# In[40]:


lasso = Lasso()
lasso.fit(train_scaled, train_target)


# In[41]:


print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target)) #쉬프트 텝을 눌러보면 람다값이 1인것을 알 수 있는데 이것이 최적값인지는 그래프를 그려봐야 확실히 알 수 있다.


# In[52]:


train_score = []
test_score= []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    #라쏘모델을 만든다
    lasso = Lasso(alpha = alpha, max_iter=10000)  #alpha_list에 있는 요소들을 alpha에 넣는다
    #라쏘 모델을 훈련한다
    lasso.fit(train_scaled, train_target)
    #훈련 점수와 테스트 점수를 저장한다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))


# In[55]:


plt.plot(train_score, alpha_list)
plt.plot(test_score, alpha_list)
plt.show()


# In[ ]:


lasso=Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print(lasso.scorec(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))


# In[ ]:


np.sum(lasso.coef_ != 0)

