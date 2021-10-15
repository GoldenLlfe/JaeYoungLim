#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  #행렬 연산
import pandas as pd  #엑셀
import matplotlib.pyplot as plt


# In[2]:


bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 
                32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 
                35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 
                500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 
                620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
print(type(bream_length))
print(np.shape(bream_length))


# In[6]:


plt.scatter(bream_length,bream_weight)

plt.title("Bream lenght vs Weight")
plt.xlabel("Length(cm)")
plt.ylabel("weight(g)")

plt.show()


# In[7]:


smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


# In[8]:


plt.scatter(bream_length, bream_weight,label="Bream")
plt.scatter(smelt_length,smelt_weight,label="Smelt")

plt.title("Bream and Smelt")
plt.xlabel("Length(cm)")
plt.ylabel("weight(g)")
plt.legend()

plt.show()


# # knn을 이용한 분류

# In[10]:


length = bream_length + smelt_length
weight = bream_weight + smelt_weight

print(np.shape(length))
print(length)


# In[11]:


fish_data = [[l,w] for l, w in zip(length,weight)]

print(type(fish_data))
print(np.shape(fish_data))


# In[12]:


fish_target = [1]*35 + [0]*14
print(fish_target)


# In[14]:


#scikitlean knn module import

from sklearn.neighbors import KNeighborsClassifier

##객체 생성

kn= KNeighborsClassifier(n_neighbors =5)


# In[15]:


#knn 모델 fitting
kn.fit(fish_data, fish_target)


# In[16]:


# knn model metrics
kn.score(fish_data,fish_target)


# In[18]:


#prediction
kn.predict([[30, 600]])


# In[21]:


plt.scatter(bream_length, bream_weight, s=100, label="bream=1")
plt.scatter(smelt_length, smelt_weight, s=100, label="bream=0")
plt.scatter(30, 600, s=100, label="new")

plt.title("Bream vs Smelt",pad=30)
plt.xlabel("Length")
plt.ylabel("Weight")
plt.legend()

plt.show


# In[24]:


kn._fit_X
kn._y


# In[28]:


fish_data=np.column_stack((length, weight))
fish_target = [1]*35 + [0]*14

print(np.shape(fish_data))


# In[31]:


train_input = fish_data[:35]
test_input = fish_data[35:]

train_target = fish_target[:35]
test_target = fish_target[35:]


# #  샘플링 편향

# In[32]:


from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()


# In[33]:


kn.fit(train_input,train_target)
kn.score(test_input,test_target)
#0.0이 나왔는데 그 이유는 빙어 데이터가 안 들어와서 전혀 러닝이 되질 않았다


# In[37]:


print(type(fish_data))
print(type(fish_target))
fish_target=np.array(fish_target)
print(type(fish_target))


# In[44]:


#np.arange(1, 11, 2) #시작 값을 1로주고 11까지 범위를 지정하고 2씩 띄어서 값을 출력

index = np.arange(49)

print(index)

np.random.shuffle(index)

print(index)


# In[55]:


train_input = fish_data[index[:35]]   #처음부터 35번째 인덱스까지 = 35개의 데이터
train_target = fish_target[index[:35]]

test_input = fish_data[index[35:]]    #36번째부터 42번째까지 = 14개의 데이ㅓ
test_target = fish_target[index[35:]]

train_input[:5]


# In[56]:


plt.scatter(train_input[:,0],train_input[:,1])  #train_input의 전체에서 0번째 컬럼을 지정
plt.scatter(test_input[:,0],test_input[:,1])
plt.show()


# In[57]:


kn.fit(train_input, train_target)
kn.score(test_input, test_target)


# In[67]:


length = bream_length+smelt_length
weight = bream_weight+smelt_weight


# In[68]:


fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)


# In[74]:


fish_data = np.column_stack((length,weight))
print(np.shape(fish_data))

fish_target = np.concatenate((np.ones(35)),(np.zeros(14)))
fish_target


# In[66]:


from sklearn.model_selection import train_test_split


# In[75]:


train_input, test_input, train_target, test_target = train_test_split(
fish_data, fish_target, stratify=fish_target)


# In[76]:


from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()


# In[77]:


kn.fit(train_input, train_target)  #fit은 트레인데이터로
kn.score(test_input,test_target)   #score는 테스트데이터로


# In[79]:


kn.predict([[25, 150]])


# In[83]:


distances, inds = kn.kneighbors([[25,150]])

print(distances)
print(inds)


# In[85]:


plt.scatter(train_input[:, 0],train_input[:,1])
plt.scatter(25,150,marker = '^')
plt.scatter(train_input[inds,0], train_input[inds,1], marker='D')

plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()


# In[88]:


train_input[inds]
train_target[inds]


# In[91]:


# feature scaling
mean = np.mean(train_input,axis=0)
std=np.std(train_input, axis=0)

print("mean = ",mean.round(3))
print("std = ",std.round(3))
print(train_input[:5])


# In[92]:


#표준화
train_scaled = (train_input - mean)/std
train_scaled[:5]


# In[93]:


new = ([25,150]-mean)/std  #새로운 데이터를 표준화 안해주면 그 데이터 혼자만 다른 세상 데이터처럼 나온다
print(new)


# In[95]:


plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker='^')

plt.xlabel("std of weight")
plt.ylabel("std of length")
plt.show()


# In[96]:


kn=KNeighborsClassifier()
kn.fit(train_scaled, train_target)


# In[101]:


#mean_t=np.mean(test_input, axis=0)
#std_t=np.std(test_input, axis=0)

test_scaled=(test_input-mean)/std


# In[102]:


kn.score(test_scaled,test_target)


# In[103]:


kn.predict([new])


# In[104]:


dist,indx=kn.kneighbors([new])
print(indx)


# In[106]:


plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indx,0],train_scaled[indx,1],marker='D')
plt.show()
#결과를 보면 표준화 전의 그래프와 다른 점이 이번 것은 제일 가까운 이웃을 도미로 꼽았고 전의 것은 빙어도 꼽았었다
#이것은 순전히 표준화로 길이와 무게의 단위를 맞춰서 나온 결과이다


# # KNN 회귀

# In[107]:


#농어
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


# In[108]:


np.shape(perch_length)


# In[109]:


np.shape(perch_weight)


# In[110]:


plt.scatter(perch_length,perch_weight)

plt.xlabel('length')
plt.ylabel('weight')

plt.show()


# In[123]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=18
)

print(np.shape(train_input))
print(np.shape(test_input))


# In[125]:


train_input = train_input.reshape(-1,1)  # [1,2,3]이걸-1열3행을 3열1행으로 바꿔준다 -1을 하면 자기가 알아서 모든 열을 해주고 아니면 정확히 42를 써야한다
test_input = test_input.reshape(-1,1)

print(np.shape(train_input))


# In[114]:


from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()


# In[126]:


knr.fit(train_input,train_target)


# In[127]:


knr.score(test_input,test_target)  #회귀 문제는 R제곱값이 나온다, 위에서 분류의 문제를 했을 때는 혼동행렬의 accuracy값이 나왔다 R제곱은 '우리의 모델이' 데이터의 99%를 설명한다는 뜻이다


# In[117]:


#R^2

from sklearn.metrics import mean_absolute_error   #평균의 절대값의 차이를 보는 것


# In[119]:


test_prediction = knr.predict(test_input)

print(test_prediction)
print(test_target)


# In[121]:


mae = mean_absolute_error(test_target,test_prediction)
print(mae)


# In[122]:


from sklearn.metrics import mean_squared_error

mse =mean_squared_error(test_target,test_prediction)

print(mse)


# In[130]:



knr = KNeighborsRegressor()
r2_train = []
r2_test = []
neighbors_n = []

for n in range(1,21):  #1부터 20까지
    knr.n_neighbors = n
    knr.fit(train_input,train_target)
    r2_train.append(knr.score(train_input, train_target))
    r2_test.append(knr.score(test_input, test_target))
    neighbors_n.append(n)

    
print(r2_train)
print(r2_test)
print(neighbors_n)


# In[131]:


plt.scatter(neighbors_n, r2_train, label='train')
plt.scatter(neighbors_n, r2_test, label='test')

plt.xlabel('num of neighbors')
plt.ylabel('R^2')
plt.legend()
plt.show()


# ## 위의 트레인 데이터는 점점더 좋아지는 Overfit이 된다
# ## 테스트는 좋아지다가 어느 순간 안좋아지기 시작하는데
# ## 위의 그래프를 기준으로 peak가 5.0과 7.5 사이에서 n을 결정해서 
# ## 최적값을 결정한다 - 이것이 하이퍼파라메타

# In[ ]:




