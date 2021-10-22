#!/usr/bin/env python
# coding: utf-8

# # k-최근접이웃 알고리즘을 활용한 분류
# 
# ## 2008-2018부터 나이, 학위, 과학/기술이 다음 세대를 진보시킬 것이라는 생각, 과학(새로운 발견)에 대한 관심을 조사한 자료를 바탕
# 
# ## 자료출처: https://gss.norc.org/

# In[1]:


#라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

#데이터 읽어오기 출처: https://gss.norc.org/
raw_data = pd.read_excel('./GSS 2.xls')
raw_data


# # 데이터 전처리 시작

# In[2]:


#컬럼명 간소화
raw_data.columns = ['year','age','degress','sci opportunity','interest in sci']
#컬럼명 확인
raw_data.columns


# In[3]:


#결측치들을 10과 100으로 치환
raw_data['sci opportunity'].replace('Not applicable', 10, inplace=True)
raw_data['interest in sci'].replace('Not applicable', 10, inplace=True)
raw_data['sci opportunity'].replace('Dont know', 10, inplace=True)
raw_data['interest in sci'].replace('Dont know', 10, inplace=True)
raw_data['sci opportunity'].replace('No answer', 10, inplace=True)
raw_data['interest in sci'].replace('No answer', 10, inplace=True)
raw_data['degress'].replace("Don't know", 10, inplace=True)
raw_data['degress'].replace('No answer', 10, inplace=True)
raw_data['age'].replace('No answer', 100, inplace=True)
raw_data['age'].replace('89 or older', 100, inplace=True)
raw_data.dropna(axis=0,inplace=True) #그외의 NaN값들 제거


# In[4]:


print('degress의 종류','\n',raw_data["degress"].value_counts(),"\n")
print('sci opportunity의 종류','\n',raw_data["sci opportunity"].value_counts(),"\n")
print('interest in sci의 종류','\n',raw_data["interest in sci"].value_counts(),"\n")
print('age의 종류','\n',raw_data["age"].value_counts(),"\n")


# In[5]:


#데이터 확인
raw_data


# In[6]:


#결측치들을 제외한 열들을 데이터로 활용하기 위해 마스크를 해준다
nan_to_10_data = (raw_data['sci opportunity'] != 10) & (raw_data['interest in sci'] != 10)& (raw_data['age']!=100) &(raw_data['degress']!=10)
nan_to_10_data


# In[7]:


#결측치들이 제거된 데이터
clean_data = raw_data.loc[nan_to_10_data, :]
print('깨끗한 데이터 원본','\n',clean_data,'\n')


# In[8]:


#컬럼명과 갯수확인
print('degress의 종류','\n',clean_data["degress"].value_counts(),"\n")
print('sci opportunity의 종류','\n',clean_data["sci opportunity"].value_counts(),"\n")
print('interest in sci의 종류','\n',clean_data["interest in sci"].value_counts(),"\n")
print('age의 종류','\n',clean_data["age"].value_counts(),"\n")


# ## 데이터전처리: 범주형-> 숫자형

# In[9]:


clean_data['interest in sci'].replace('Moderately interested', 1, inplace=True)
clean_data['interest in sci'].replace('Very interested', 2, inplace=True)
clean_data['interest in sci'].replace('Not at all interested', 3, inplace=True)


# In[10]:


clean_data


# In[11]:



    

# interest in sci 데이터를 숫자형으로 변환
#onehot_interest = pd.get_dummies(clean_data['interest in sci'])
onehot_degress = pd.get_dummies(clean_data['degress'])
onehot_opportunity = pd.get_dummies(clean_data['sci opportunity'])

#숫자형으로 바뀐 데이터들을 원본 데이터에 합쳐주기
#clean_data = pd.concat([clean_data, onehot_interest],axis=1)
clean_data = pd.concat([clean_data, onehot_degress],axis=1)
clean_data = pd.concat([clean_data, onehot_opportunity],axis=1)

#더미로 바꾼 interest in sci열 삭제
#clean_data.drop(['interest in sci'],axis=1,inplace=True)
clean_data.drop(['degress'],axis=1,inplace=True)
clean_data.drop(['sci opportunity'],axis=1,inplace=True)
clean_data


# # 훈련과 검증 데이터 분할

# In[12]:


#x축과 y축 지정

#예측할 항목의 바탕이 되는 자료들: 조사년도, 나이, 최고 학력, 과학/기술이 다음 세대를 진보 시킬 것이라는 생각
sci_train = clean_data[['year','age','Bachelor','Graduate','High school','Junior college','Lt high school',
                    'Agree','Disagree','Strongly agree','Strongly disagree']]



#이 알고리즘의 목표는 사람들의 과학에 대한 관심도의 예측이다
sci_target = clean_data[['interest in sci']]

# 데이터를 정규화
sci_train = preprocessing.StandardScaler().fit(sci_train).transform(sci_train)


# In[13]:


# 훈련용 데이터와 검증용 데이터의 비율조정
train_input, test_input, train_target, test_target = train_test_split(
sci_train, sci_target, stratify=sci_target)

print('train data의 갯수: ',train_input,'\n')
print('test data의 갯수: ', test_input,'\n')

print(type(sci_target))

sci_target.head()


# # KNN 분류

# In[16]:


# KNN 객체 생성 - k=5
kn = KNeighborsClassifier(n_neighbors=5)

#학습
kn.fit(train_input, train_target)

#예측값
y_hat = kn.predict(sci_train)

#1차 확인
print(y_hat[0:10])
print(sci_target[0:10])

print(sci_target[['interest in sci']].value_counts())


# ## 거참 결과한번 끔찍하네... 10개중 1개 맞음

# In[17]:


#정확도 확인!
# 확인 결과 약 36% 처참!!
kn.score(test_input,test_target)


# In[18]:


k_list = range(1,10)
accuracies = []
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_input, train_target)
    accuracies.append(classifier.score(test_input, test_target))


# In[19]:


plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Interest in Science")
plt.show()


# # KNN이 안되면 Decision Tree로 시도

# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

dt = DecisionTreeClassifier()
dt.fit(train_input, train_target)


# In[21]:


#plt.figure(figsize=(10,3))
#plot_tree(dt)
#plt.show()


# In[22]:


#plt.figure(figsize=(20,3))
#plot_tree(dt, max_depth=1, filled=True, feature_names=['year','age','Bachelor','Graduate','High school','Junior college','Lt high school',
#                    'Agree','Disagree','Strongly agree','Strongly disagree'])

#plt.show()


# In[23]:


dt = DecisionTreeClassifier(max_depth=5)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))


# In[24]:


dt.feature_importances_


# In[25]:


from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier()

papam = {'max_depth': np.arange(4, 20, 1),
        'min_impurity_decrease': np.arange(0.0001, 0.001, 0.001)}

gs = GridSearchCV(dt, param_grid=papam, cv=5,
                 n_jobs=-1)


# In[26]:


spliter = StratifiedKFold(n_splits=10,shuffle=True) 
score = cross_validate(dt, train_input, train_target, cv=10)


# In[27]:


score['test_score']


# In[28]:


gs.fit(train_input ,train_target)


# In[29]:


gs.cv_results_['mean_test_score']


# In[30]:


dt = gs.best_estimator_
print(dt.score(train_input, train_target))


# In[31]:


#['year','age','Bachelor','Graduate','High school','Junior college','Lt high school',
#                    'Agree','Disagree','Strongly agree','Strongly disagree']


# In[33]:


kn = KNeighborsClassifier(n_neighbors=5)
my_test_train=my_data
my_test_train = preprocessing.StandardScaler().fit(my_test_train).transform(my_test_train)
kn.fit(my_test_train, train_target)

y_hat = kn.predict(my_test_train)

print(y_hat)


# # 결론: 계속 Underfitting이 발생
# 
# # 데이터에 무슨 문제가 있다
# 
# # 범주형에서 숫자형으로 변환할때 문제가 있거나
# 
# # 애초에 데이터를 잘못 가져왔을 수도 있다. 예)데이터의 빈도수/  학위-다양한 학위들이 있을 텐데 이과와 관련되지 않은 학위들은 자연스럽게 과학에 관심이 없을테니 오류를 유발할 수 있다
