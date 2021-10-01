#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np

df=pd.read_csv('./auto-mpg.csv',header=None)
df.columns = ['mpg','cylinders','displacement',
              'horsepower','weight','acceleration','model_year',
              'origin','name']


# In[65]:


#horsepower 열의 누락 데이터 처리 ('?')삭제후 실수형으로 변경
df['horsepower'].replace('?',np.nan, inplace=True)
df.dropna(subset=['horsepower'],axis=0,inplace=True)
df['horsepower']=df['horsepower'].astype('float')


# In[66]:


#3개의 구간(bin)으로 나누어서 범위로 처리를 할려고 한다
count, bin_div=np.histogram(df['horsepower'],bins=3)  #경계리스트/구간 
# 위의 코드 중 numpy의 histogram이라는 함수는 리턴값을 2개를 주는 함수
print(bin_div)  #아래 결과를 보면 46/107.33/168.67/230이 있다
## 46-107 사이를 1개 구간 107-168 사이를 2번째 구간 168-230을 3번째 구간으로 나눈것임
bin_names = ['저출력','보통출력','고출력']


# In[67]:


df['hp_bin']=pd.cut(x=df['horsepower'],   #데이터 배열
                   bins = bin_div,   #경계 리스트
                   labels = bin_names,
                   include_lowest = True)
df[['horsepower','hp_bin']].head()


# In[55]:


# 'age' 컬럼으로 데이터 처리 bin을 4로 [유기,청소년,청년]
import seaborn as sns

df_t = sns.load_dataset('titanic')


# In[56]:


df_t.columns = ['survived','places','sex','age','sibsp',
              'parch','far','embarked','class','who',
             'adult_male','deck','embark_town','alive',
             'alone']
df_t['age']


# In[57]:


df_t['age'].replace('?',np.nan, inplace=True)
df_t.dropna(subset=['age'],axis=0,inplace=True)


# In[58]:


count, bin_titanic_age=np.histogram(df_t['age'],bins=4)
bin_titanic_names = ['영유아','청소년','청년','장년']
bin_titanic_names


# In[60]:


count, bin_titanic_age=np.histogram(df_t['age'],bins=4)
bin_titanic_names = ['영유아','청소년','청년','장년']
df_t['titanic_bin']=pd.cut(x=df['age'],   #데이터 배열
                   bins = bin_titanic_age,   #경계 리스트
                   labels = bin_titanic_names,
                   include_lowest = True)
df_t[['age','titanic_bin']].head(100)


# In[43]:


horse_dummies = pd.get_dummies(df['hp_bin'])
horse_dummies.head(10)
pd.get_dummies(df['hp_bin'],prefix='hp')


# In[54]:


pd.get_dummies[df['age','titanic_bin']].head(10)
pd.get_dummies(df['titanic_bin'])


# In[61]:


from sklearn import preprocessing

#전처리를 위한 encoder 객체 만들기
label_encoder = preprocessing.LabelEncoder()  #label encoder 생성
onehot_encoder = preprocessing.OneHotEncoder() #onehot encoder 생성


# In[71]:


#label encoder로 문자열 범주를 숫자형 범주로 변환
#머신러닝, 회귀분석 같은 것들을 위해 변환해주는 것
onehot_labeled = label_encoder.fit_transform(df['hp_bin'].head(15))
onehot_labeled = preprocessing.LabelEncoder().fit_transform(df['hp_bin'].head(15))
print(onehot_labeled)
print(type(onehot_labeled))
print(list(onehot_labeled))


# In[75]:


#위의 배열은 1차원
#그걸 2차원 행렬로 형태 변경 reshape(데이터행수,1ㅋ)
onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled),1)
print(onehot_reshaped)

#데이터 분석을 위해서슨ㄴ nparrAY 2차원 배열, 1차원 배열, n차워 배열


# In[77]:


from sklearn.datasets import load_iris

iris_data = load_iris()
iris_data  #data, target, target_names(인덱스명) feature_names(컬럼의 이름)


# In[96]:


arr = np.arange(24)
print(arr, type(arr), ":\n", arr.shape,":\n", arr.size,":", arr.dtype)
print()

arr=arr.reshape(3,4,2)
print(arr, type(arr), ":\n", arr.shape,":\n", arr.size,":", arr.dtype)
arr.sum(axis=2)  # 위의 3,4,2 ,에서 2를 기준으로 행렬을 더해줌으로 앞에 남은 3x4행렬로 나온다.
a = np.arange(-5, 5, 0.5)
a


# In[97]:


a_list = [1.0, 2.0, 3.0]
print(type(a_list))
arr_list = np.array(a_list)
type(arr_list)


# In[101]:


a1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([[1.0, 2.0],[3.0,4.0]])
print(a1, type(a1),"\n")
print(a2, type(a2),"\n")
print("a1차원: ",a1.ndim,"a1의 행과열의 수/모양",a1.shape,"a1의 크기: ",a1.size,"a1의 자료형",a1.dtype,"\n")
print("a2차원: ",a1.ndim,"a2의 행과열의 수/모양",a2.shape,"a2의 크기: ",a2.size,"a2의 자료형",a1.dtype)


# In[111]:


a = np.zeros(5)
b = np.zeros((2,3))
c = np.zeros((5,2),dtype="i")
d = np.zeros(5, dtype="U4")
e = np.zeros((2,3,4),dtype="i8")
f = np.zeros_like(b,dtype="f")
g = np.empty((4,3))  #아무값이나 들어가 있는 4x3배열 쓰래기 데이터임
f


# In[112]:


x1 = np.array([1.0, 2.0, 3.0])
y1 = np.array([5.0, 10.0, 15.0])
x2 = np.array([[1.0, 2.0],[ 3.0, 4.0]]) 
y2 = np.array([[5.0,10.0],[15.0,20.0]]) 
z1 = np.array([-1.0, -2.0])
z2 = np.array([[5.0],[10.0],[15.0]])

# ndarray basic operation  
print(x1 + y1) 
print(x1 - y1) 
print(x1 * y1) 
print(x1 / y1) 
print(x2 + y2) 
print(x2 * y2)


# In[113]:


# ndarray broadcast 
print(x2 + z1) 
print(x2 * z1) 
print(x1 + z2) 
print(x1**2) 
print(x1>=2)

# shape manipulation
print(x2.flatten()) 
print(x2.reshape(2,2))


# In[135]:


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#이 배열에서 3의 배수를 찾아라.
#이 배열에서 4로 나누면 1이 남는 수를 찾아라.
#이 배열에서 3으로 나누면 나누어지고 4로 나누면 1이 남는 수를 찾아라.
a_list = x
"""for i in x[:]:
    if i % 3 ==0:
        i=a
    if i % 4 ==1:
        i=b
    if (i % 3 == 0) and (i % 4 == 1):
        i=c
"""

#교수님 답안
print("3의 배수 : ", x[x%3==0])
print("4로 나누면 1이 남는 배수", x[x%4==1])
a_arr = list(np.array((x[x%3==0])&np.array(x[x%4==1]))
print("3의 배수이면서 4의 나머지 값이 1인 수 :",x[a_arr])


# In[137]:


#dataframe function
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.columns


# In[141]:


df = titanic.loc[: ,['age','fare']]  #모든 행에 대해서 'age'와 'fare'를 가져온다
df['ten'] = 10
df.head()

#매핑 함수를 이용하여 각 원소에 동일한 함수 실행
def add_10(n):
    return n+10
add_10(20)
#apply() 메서드를 활용하여 시리즈이 각 원소에 동일한 함수 실행
ar = df['age'].apply(add_10)
ar


# In[143]:


df['ten_10'] = df['age'].apply(add_10)
df.head()


# In[146]:


#apply() 메서드를 활용하여 시리즈이 각 원소에 동일한 함수 실행
ar = df['age'].apply(add_10)
df['age_lamb'] = df['age'].apply(lambda x:add_10(x))
df.head()


# In[144]:


df_map = df.applymap(add_10)
df.head()


# In[171]:


# 1. titanic 데이터를 load
# 2. age 와 fare만 추출
# 3. age에서 평균 나이를 차감한 값을 age_avg 컬럼으로 추가
# 4. 나이의 구간을 4단계로 나눠서
# 0-20 청소년 21-70 장년 71~ 노년 으로 컬럼 추가
# one_hot_encpding을 활용
# df['age_avg'] = df['titanic']
import seaborn as sns

df = sns.load_dataset('titanic')
df['age'].replace('?',np.nan, inplace=True)
df['fare'].replace('?',np.nan, inplace=True)
df.dropna(subset=['age'],axis=0,inplace=True)
df.dropna(subset=['fare'],axis=0,inplace=True)
titanic_age = df['age']
titanic_fare = df['fare']
avg = sum(titanic_age)/len(titanic_age)
def titanic_age_minus_avg(n):
    return n - avg
df['age_avg'] = df['age'].apply(titanic_age_minus_avg)
count, bin_titanic_age=np.histogram(df_t['age'],bins=4)
bin_titanic_names = ['청소년','청년','장년','노년']
bin_values = [0,21,71,100]
df['titanic_bin']=pd.cut(x=df['age'],   #데이터 배열
                   bins = bin_titanic_age,   #경계 리스트
                   labels = bin_titanic_names,
                   include_lowest = True)
df[bin_titanic_names]= pd.get_dummies(df['titanic_bin'])
df[['age','age_avg']] = df[['age','age_avg']].astype("int")
df[['age','titanic_bin','age_avg','fare']]

