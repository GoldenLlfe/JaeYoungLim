#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish["Species"].value_counts()


# In[4]:


fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()


# In[5]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)


# In[6]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# In[7]:


bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]


# In[8]:


print(target_bream_smelt.shape)
print(train_bream_smelt.shape)


# In[9]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


# In[12]:


print(lr.fit(train_scaled, train_target))
print(lr.score(train_scaled,train_target))


# In[13]:


print(lr.predict(train_scaled[:5]))
print(train_target[:5])


# In[15]:


print(lr.predict_proba(train_scaled[:5]).round(3))


# In[17]:


wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()
wine.info()


# In[18]:


data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()


# In[20]:


data[:10]
target[:10]


# In[21]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
data, target, stratify=target)

print(train_input.shape)
print(test_input.shape)


# In[36]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

dt = DecisionTreeClassifier()
dt.fit(train_input, train_target)


# In[37]:


plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


# In[38]:


plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


# In[39]:


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))  #트리에서 오버피빙팅을 방지하는 법은 따로 regulazation 말고 튜닝이라고 부른다


# In[40]:


plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,feature_names=['alcoho','sugar','pH'])
plt.show()


# In[41]:


dt.feature_importances_   #정확도가 높은 피쳐 고르는것
#feature_names = ['alcohol', 'sugar','pH'] 이 중에 피쳐를 뺐을 때 


# In[43]:


data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

print(data.shape)


# In[44]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
data, target, stratify=target, random_state=42)


# In[45]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)  #여기서 랜덤 스테이터는 data를 나눌때 쓰는것, 부트스트래핑할 때 쓴다.


# In[51]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

spliter = StratifiedKFold(n_splits=10,shuffle=True, random_state=42)  #cv가 10이니 스플릿도 10개, 이 코드의 목적은 데이터의 비율을 맞추기 위해서임
score = cross_validate(dt, train_input, train_target, cv=10) #cv는 k-fold수하지만 테스트와 트레인의 비율은 랜덤이다


# In[52]:


score['test_score']  #10개가 나온이유는 k-fold수가 10이어서


# In[55]:


score['test_score'].mean()  #이 값과 비슷해야한다, 이 값이 너무 않좋으면 DecisionTreeClassifier에서 랜덤 스테이트 앞에 max_iter를 넣으면 된다


# In[54]:


print(score)


# In[64]:


from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier()

papam = {'max_depth': np.arange(4, 20, 1),
        'min_impurity_decrease': np.arange(0.0001, 0.001, 0.001)}

gs = GridSearchCV(dt, param_grid=papam, cv=5,
                 n_jobs=-1)  #컴퓨터가 가진 자원을 다쓴다는 말이다 아니면 프로세서 1개만 쓴다


# In[65]:


gs.fit(train_input ,train_target)


# In[66]:


gs.cv_results_['mean_test_score']  #4번째, 측 depth가 5일 때 정확도가 피크이다


# In[67]:


dt = gs.best_estimator_  #교차 검증 값이 가장 좋은 걸 가져다 쓴다
print(dt.score(train_input, train_target))


# In[68]:


gs.best_params_


# In[69]:


from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,
                           criterion='gini',
                           n_jobs=-1)

scores = cross_validate(rf, train_input, train_target, cv=10,
                       return_train_score=True) #교차 검증된 값이 나온다


# In[73]:


print(np.mean(scores['test_score']))  #이 경우 랜덤포래스트가 굉장히 잘 맞는 경우다
print(np.mean(scores['train_score']))


# In[75]:


rf.fit(train_input, train_target)


# In[76]:


print(rf.feature_importances_)


# In[78]:


wine.columns


# ### 위의 feature_importances_를 봤을 때 가장 높은 것이 0.495인데 이때 와인데이터의 컬럼들을 보면 2번째가 설탕이기 때문에 가장 중요한 피처가 설탕이다

# In[81]:


get_ipython().system(' python -m wget https://bit.ly/fruits_300 -o fruits_300.npy')


# In[80]:


get_ipython().system(' pip install wget')


# In[82]:


import numpy as np
import matplotlib.pyplot as plt


# In[84]:


fruits = np.load('./fruits_300.npy')
fruits[0]


# In[85]:


plt.imshow(fruits[0], cmap="gray")
plt.show()


# In[86]:


plt.imshow(fruits[0], cmap="gray_r")
plt.show()


# In[88]:


plt.imshow(fruits[100], cmap="gray_r")
plt.show()


# In[89]:


plt.imshow(fruits[200], cmap="gray_r")
plt.show()


# In[92]:


plt.imshow(fruits[299], cmap="gray_r")
plt.show()
print(fruits.shape)


# In[94]:


fruits_2d = fruits.reshape(-1, 10000) #그림의 차원이 300 x 100 x 100의 3차원으로 이렇게 하면 300 x 10000짜리 2차원으로 바뀐다
print(fruits_2d.shape)


# In[99]:


fig, axs = plt.subplots(1,3) #1열에 3장짜리 그림들을 그리는것
axs[0].imshow(fruits[0],cmap="gray_r")
axs[1].imshow(fruits[100],cmap="gray_r")
axs[2].imshow(fruits[200],cmap="gray_r")

plt.show()


# In[100]:


from sklearn.cluster import KMeans


# In[101]:


km = KMeans(n_clusters=3,random_state=42)  #클러스터를 몇개 할건지 정해야하는데 이것이 kmeans의 하이퍼파라메타이다
km.fit(fruits_2d)  #이건 타겟이 없다


# In[103]:


print(km.labels_)  #어떻게 나눠 졌느닞 보여준다, 잘 보면 중간에 잘못된 것들이 들어가있다 예를들어 2만 모여있어야 한느데 판단을 잘못해서 0도 껴있다


# In[105]:


km.cluster_centers_.shape


# In[106]:


km_center = km.cluster_centers_.reshape(-1, 100, 100)

print(km_center.shape)


# In[107]:


fig, axs = plt.subplots(1,3) #1열에 3장짜리 그림들을 그리는것
axs[0].imshow(km_center[0],cmap="gray_r")
axs[1].imshow(km_center[1],cmap="gray_r")
axs[2].imshow(km_center[2],cmap="gray_r")

plt.show()


# ### 위의 사진들은 센터의 모습을 보여준것으로 클러스터들을 모두 모은 것들이다 위에 km.labels_를 그래프로 보여준 것이다. 우리는 답을 알고 있어서 클러스터를 3개를 줬지만 실제의 경우에는 k를 여러개를 넣어봐야한다

# In[108]:


inertia = []

for k in range(2, 7):
    km = KMeans(n_clusters= k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
    
plt.plot(range(2, 7),inertia)   #이것이 elbow method
plt.show()


# ### 위의 것이 elbow method로 3에 꺽이는 것을 볼 수 있다. 하지만 저렇게 확연히 elbow가 나타날 정도로 꺽이는 경우는 거의 없다. 대부분의 대부분은 그냥 곡선을 그리면서 그냥 계속 내려간다

# In[110]:


km.predict(fruits_2d[100:101])   #2d 배열을 만들어서 넣어줘야한다 차원을 맞춰줘야 한다


# In[111]:


km.predict(fruits_2d[120:121])


# In[ ]:




