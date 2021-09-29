#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'이름' : [ '서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)
print(df)


# In[46]:


import pandas as pd
exam_data = {'이름' : [ '서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)
# '이름' 열을 새로운 인덱스로 지정하고, df 객체에 변경사항 반영
#서준의 수학 점수를 100으로 변경
#우현과 인어의 영어와 수학점수를 출력 iloc와 범위 loc와 배열사용

df.iloc[1:3, 0:1]
df.loc[["우현","인아"],["수학","영어"]]
df.loc["우현":"인아","수학":"영어"]
#인아의 모든 점수를 100으로 수정
df.인아 = 100
#서준의 모든 점수 출력
print(df['서준']) #틀렸음 df.loc['서준']
#수학과 음악 점수만 출력
df.loc[['수학'],['음악']] #틀렸음 df[['수학'],[음악]]
df


# In[44]:


df.T


# In[47]:


df.T


# In[48]:


#reindex :index를 재배치
import pandas as pd

# 딕셔서리를 정의
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

# 딕셔서리를 데이터프레임으로 변환. 인덱스를 [r0, r1, r2]로 지정
df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
print(df)
print('\n')


# In[51]:


#인덱스를 [r0, r1,r2,r3,r4]로 재지정
new_index = ['r0','r1','r2','r3','r4'] #추가되는 데이터는 Nan로 입력됨
ndf = df.reindex(new_index, fill_value=0)
ndf


# In[62]:


#reindex :index를 재배치
import pandas as pd

# 딕셔서리를 정의
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

# 딕셔서리를 데이터프레임으로 변환. 인덱스를 [r0, r1, r2]로 지정
df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
print(df.reset_index())
print(df.sort_index(ascending=False)) #index를 asecding(오름차순)으로 정렬 해주는데 =False로 하면 역으로 정렬한다
print(df.sort_values(by='c0', ascending=False))
df.sort_values(by=['c0','c1'], ascending=False)


# In[67]:


import pandas as pd

#딕셔너리 데이터로 판다스 시리즈 만들기
student1 = pd.Series({'국어':100, '영어':80,'수학':90})
student2 = pd.Series({'국어':90, '영어':85,'수학':95})
print(student1)
print('\n')

print(student1/200, "\n")
print(student1 + student2, "\n") #series 연산자 (+, -, *, /) series 또는 숫자로 연산을 한다
print(student1 / student2, "\n")


# In[70]:


import pandas as pd

#딕셔너리 데이터로 판다스 시리즈 만들기
student1 = pd.Series({'국어':100, '영어':80,'수학':90})
student2 = pd.Series({'국어':90, '영어':85,'수학':95, '과학':100})
student1.add(student2, fill_value=0) #이때 student1에는 '과학' 열의 값이 없어서 NaN으로 값으로 대체되는데 fill_value=o로 0으로 값을 대체함, 원하는 값을 써 넣을수 있음
#student1 의 series에 '역사':80을 넣고 add를 진행시키면 student2 의 과학 100과 student1의 역사 80 둘다 그대로 나온다.


# In[85]:


#dataframe 연산, 기존의 모듈에서 dataset을 불러들임
import seaborn as sns

titanic = sns.load_dataset('titanic') #모듈에서 제공되는 dataset을 가져오는 메소드
type(titanic)
df = titanic.loc[ : , ['age','fare' ]] #titanic의 데이터에서 age와 fare 열/cloumn만 가져온다
df1 = titanic.loc[100:201, ['age','fare']] #age와 fare 컬럼에서 100-200까지의 인덱스만 가져온다
print("df의 상위 5개","\n",df.head(5),"\n")
print("df1의 상위 5개","\n",df1.head(5),"\n")
print("df1에 100을 채우고 NaN값은 666으로 대체","\n",df1.add(10, fill_value=666),"\n")
print("df의 상위 5개에 10을 더한다","\n",df.head(5) + 10)  #df의 상위 5개에 10을 더해준다
df2 = df.tail(10)  #df2에 df의 하위 10개의인덱스를 넣는다
print("df1에 df2를 더하고 없는값은 66으로 대체","\n",df1.add(df2, fill_value=666))
print(df2)


# In[111]:


print(df.age.sum()) #df의 나이 컬럼의 총함
print(df.age.count()) #df의 나이 컬럼의 인덱스 수
print(df.age.sum()/df.age.count(),"\n") #df의 나이 컬럼의 평균
#평균나이보다 적은 연령의 자료만 출력
avg_age = int(df.age.sum()/df.age.count())
print("titanic 의 손님들의 평균 연령:{}".format(avg_age))
for i in df1.age.head(10):
    if i<avg_age:
        age_to_int = int(i)
        print(age_to_int)
        print()
#이건 교수님 답안
df1 = titanic.loc[100:200, ['age','fare']]
for idx, age in enumerate(df1.age): #enumerate() index와 데이터를 return
    if age and age < avg_age:  #원래 답안 age < avg_age: 
        print("{} : {}".format(df1.iloc[idx,0], df.iloc[idx,1])) #원소 검색 df.iloc / df.loc


# In[ ]:




