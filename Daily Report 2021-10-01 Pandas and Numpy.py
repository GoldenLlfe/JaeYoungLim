#!/usr/bin/env python
# coding: utf-8

# # 데이터 사전 처리

# In[17]:



import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')


# In[3]:


#isnull() : NaN이면 True를 반환 notnull() : 값이 존재하면 데이터의 값을 반환, 아니면 NaN반환
# 누락된 데이터 처리
#제거, 치환
print(df.head().isnull())
print(df['deck'].isnull())
print()
print(df.notnull())  #값이 존재하면 값을 없으면 NaN을 출력
print()
print(df.isnull().sum(axis=0))  #누락된 데이터가 있는지 확인


# In[ ]:


#누락된 자료가 500개 이상인 자료를 찾기
df_1 = df.dropna(axis=1,thresh=500)
"deck" in (df_1.columns)


# In[9]:


df_2 = df.dropna(subset=['age'], how='any', axis=0)


# In[11]:


print(len(df_2))
len(df_2) - df['age'].isnull().sum(axis=0)


# In[16]:


#누락된 자료가 500개 이상인 자료를 찾기(찾아서 제거하는 것)
df_1 = df.dropna(axis=1,thresh=500)
#age열에 NaN 값이 있으면 행을 제거하는 방법
df_2 = df.dropna(subset=['age'], how='any', axis=0)
#age 열의 NaN값을 다른 나이 데이터의 평균으로 변경하기
mean_age = df['age'].mean(axis=0)
df_age = df['age'].fillna(mean_age)
#앞의 값으로 수정
df_age_1 = df['age'].fillna(method='ffill')
df_age_1
#가장 빈번하게 나오는 값으로 수정
most_cnt = df['age'].value_counts(dropna=True).idxmax()  #'age' 컬럼에서 널 값을 제외하고 가장 많이 나오는 값으로 준다
df_age_2 = df['age'].fillna(most_cnt)
print(df_age_2)


# In[13]:


print(df_age.isnull().sum(axis=0))
len(df_age)


# In[20]:


#중복 데이터 처리
df = pd.DataFrame({'c1' : ['a','a','b','a','b'],
                    'c2': [1, 1, 1, 2, 2],
                  'c3':[1,1,2,2,2]})
print(df,"\n")
df_dup = df.duplicated()  #행의 중복을 체크
print(df_dup)
#특정 열의 중복 체크
df['c1'].duplicated()
#데이터프레임에서 중복 행을 제거
df2 = df.drop_duplicates()
print(df2)
#특정 열을 기준으로 중복 행을 제거
df2 = df.drop_duplicates(subset=['c2','c3'])
df2


# In[22]:


df = pd.read_csv('./auto-mpg.csv', header=None)


# In[23]:


# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement',
              'horsepower','weight','acceleration','model_year',
              'origin','name']
df.head()


# In[24]:


# mpg(mile per gallon)을 kpl(illometer per liter)로 변환 (mpg to kpl = 0.425)
mpg_to_kpl = 1.60934/3.78541
#mpg 열에 0.425를 곱한 결과를 새로운 열(kpl)에 추가
df['kpl'] = (df['mpg'] * mpg_to_kpl).round(2)
df.kpl.head(3)


# In[27]:


df.horsepower.unique()


# In[42]:


# 1.데이터 타입 확인  , unique(), '?' -> NaN으로 처리
# 2.NaN 데이터 확인 후 -> 처리 -> 0.0 값으로 치환
# 3. 데이터 타입 변경 -> float으로 변경
# horsepower 컬럼에 대해서 순서대로 처리
import numpy as np
df.horsepower.unique()
df_hp = df.copy()
df.horsepower.replace('?',np.nan, inplace=True) #쓰레기 값을 Nan 값으로 처리


# In[45]:


df['horsepower'].fillna('0.0', inplace = True)  #치환된 NaN값을 0.0으로 대체
#df_hp.dropna(subset=['horsepower'],axis=0,inplace=True)이런 식으로 horsepower열에 누락데이터가 있는 행을 삭제 할 수도 있다
df_hp.dtypes   #아직 horsepower 열의 자료형은 object=str 이다


# In[48]:


df_hp['horsepower'] = df_hp['horsepower'].astype('float')
df_hp.dtypes    #윗 줄에서 horsepower의 자료형을 실수형으로 바꿔줌


# In[51]:


#category  origin은 제조국가를 코드(숫자)로 넣어놓은 상태
print(df['origin'].unique())  #정수형 데이터를 문자로 변경해야함
df_hp['origin'].replace({1: 'USA', 2: 'EU', 3:'JAPAN'},inplace=True)
print(df_hp['origin'].unique())
print(df['origin'].dtypes)
df['origin']=df['origin'].astype('category')
print(df['origin'].dtypes)

df['origin']= df['origin'].astype('object')
print(df['origin'].dtypes)


# In[ ]:




