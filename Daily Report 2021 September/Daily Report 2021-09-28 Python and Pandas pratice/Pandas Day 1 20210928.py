#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

dict_data = {'a': 1, 'b': 2, 'c': 3}

sr = pd.Series(dict_data)

print(type(sr))
print()
print(sr)
print()
print(dict_data)
print()
print(sr[['a','c']])
sr[1:2]  #인덱스 첨자로 접근하면 마지막 첨자 위치 데이터는 포함하지 않는다
print()
sr['b':'c'] #인덱스명으로 저버근하면 마지막 데이터를 포함 시킨다

#딕셔너리를 시리즈로 변경 : 딕셔너리의 키가 시리즈의 인덱스명이 됨
# 접근은 인덱스명 또는 인덱스 첨자로 접근 가능


# In[33]:


dict_data={'a':[1,2,3], 'b':[4,5,6],'c':[7,8,9]}  #'a'는 key값

sd = pd.DataFrame(dict_data)
print("sd데이터프레임")
print(sd)
print()
#index명을 부여
sd1 = pd.DataFrame(dict_data, index=["idx1", "idx2","idx3"], columns=['a', 'b', 'c'])
print("sd1 데이터프레임")
print(sd1)
print()
a_var = [[1,2,3],[4,5,6],[7,8,9]]
sd2 = pd.DataFrame(a_var, index=["idx1", "idx2","idx3"], columns=['a1', 'b1', 'c1'])
print("sd2 데이터프레임")
print(sd2)
print()
print("sd2 인덱스명 바꾸기")
print(sd2.rename(index={"idx1":'ix1', "idx2":'ix2',"idx3":'ix3'}))  #중괄호 밖에 ,inplace=True 를 넣으면 원래 데이터가 바뀐다.

print()
print("sd2 인덱스명 바꾼걸 원래 데이터에도 적용")
sd2.rename(index={"idx1":'ix1', "idx2":'ix2',"idx3":'ix3'}, inplace=True)
sd2


# In[49]:


#행 인덱스명 변경
a_data={'a':[1,2,3], 'b':[4,5,6],'c':[7,8,9]}
df = pd.DataFrame(a_data)

df.index=['a','b','d']
df.columns = ['c1', 'c2', 'c3']
df1=df  #df1은 df의 주소를 가리키고 있는 것이라 df1에서 drop을 하면 df에서 drop을 한 것
# 주소를 가져오는게 아니라 아예 복사를 하고 싶으면 df1 = df[:] 같으 df1은 df의 모든것 넣겠다 이렇게 하거나 다른 함수를 써야한다.
df1.rename(index={'a':'ida','b':'idb'}) #inplace=True)
print(df)

#행 삭제 axis=0 ,열 삭제는 axis=1
df1.drop("a", axis=0,inplace=True)
df2 = df1
df2.drop('c1',axis=1,inplace=True)
df2


# In[62]:


import pandas as pd

exam_data = {
    '수학' : [90,80,70],
    '영어' : [98,89,95],
    '음악' : [85,95,100],
    '체육' : [100,90,90]
}

df = pd.DataFrame(exam_data, index=['서준', '우현', '인아'])
print(df)
print('\n')

#데이터 프레임 df를 복제하여 변수 df2에 저장, df2의 1개 행 삭제
df2 =df.copy()   #df2에 df를 복사한것 위의 것처럼 df2=df[:]를 해도 실행은 되지만 버전이 바뀌면서 .copy()를 추천한다.
df2.drop('우현', inplace=True)

print(df2)
print('\n')

#데이터 프레임 df를 복제하여 변수 df3에 저장, df3의 1개 행 삭제
df3 = df.copy()
df3.drop(['우현','인아'],inplace=True)
print(df3)
print()
#행 선택
print("행선택")
print('loc를 이용한 행선택')
print(df.loc['서준':'인아'])
print()
print("iloc를 이용한 행선택")
print(df.iloc[0:2])
print()
print('인덱스명을 이용한 열선택')
print(df[['수학','체육']])
print()

df.loc["과학"] = [100,90,80,70] #row로 데이터를 추가하는 경우 df.loc[인덱스명] = [값, ...]
df


# In[96]:


def input_score():
    scores = input("국어, 영어, 수학 점수를 입력하세요 : ").split()
    for i, score in enumerate(scores):
        scores[i] = int(score)
    return scores


# In[98]:


#문제) 키보드에서 이름과 국어 영어 수학 점수를 입력받아 이름을 인덱스로 저장하는 score 데이터프레임을 만든다
import pandas as pd

df = pd.DataFrame(columns=["국어","영어","수학"])
#score.loc[input_name] = input_name #row로 데이터를 추가하는 경우 df.loc[인덱스명] = [값, ...]
while True:
    name = input("이름을 입력하세요 : ")
    if name == "quit":
        break
        
    score = input_score()
    df.loc[name] = score  #입력된 name 인덱스로 점수 추가
df #데이터 출력
# 컬럼을 접근하는 방법 df.컴럼명 또는 df[컬럼명]
#row를 접근하는 방법 df.iloc[인덱스], df.iloc[[1,3,5]], df.iloc[0:3]
#        df.loc['aa':'cc']     df.loc[['aa','cc']]
#df.iloc[시작:끝:간격]
#df.loc[인덱스명, 컬럼명], df.iloc[index 번호, column 열번호]


# In[101]:


class Student:   #class 클래스이름():
    def __init__(self,name,korean,math,english):
        self.name = name
        self.korean = df.iloc[0]
        self.math = df.iloc[2]
        self.english = df.iloc[1]
        
    #Student 클래스 a_class.인스턴스, 데이터


    #학생 점수의 합을 구하는 함수
    def get_sum(self):
        return self.korean + self.math + self.englis
    #평균을 구하는 함수
    def get_avg(self):
        return self.get_sum() / 3

    #출력하는 함수
    def to_string(self):
        return "{}: \t{} \t{}".format(self.name, self.get_sum(), self.get_avg())

    def __str__(self):
        return "{}: {}\t {}".format(self.name,                                   self.get_sum(),                                   self.get_avg())


# In[120]:


#문제) 키보드에서 이름과 국어 영어 수학 점수를 입력받아 이름을 인덱스로 저장하는 score 데이터프레임을 만든다
import pandas as pd

df = pd.DataFrame(columns=["국어","영어","수학"])
#score.loc[input_name] = input_name #row로 데이터를 추가하는 경우 df.loc[인덱스명] = [값, ...]
while True:
    name = input("이름을 입력하세요 : ")
    if name == "quit":
        break
        
    score = input_score()
    df.loc[name] = score  #입력된 name 인덱스로 점수 추가

search = input("검색하고자 하는 이름을 입력하세요 : ")
for name in list(df.index):
    a = df.iloc[name]
    if search == name:
        korean = a.loc[0]
        math = a.loc[2]
        english = a.loc[1]
        print("{}의 {} {} {}의 합계는 {}".format(search,korean,english,math,korea+math+english))
df #데이터 출력
#검색하고자 하는 이름을 입력받아 성적과 성적과 합계 출력
# 컬럼을 접근하는 방법 df.컴럼명 또는 df[컬럼명]
#row를 접근하는 방법 df.iloc[인덱스], df.iloc[[1,3,5]], df.iloc[0:3]
#        df.loc['aa':'cc']     df.loc[['aa','cc']]
#df.iloc[시작:끝:간격]
#df.loc[인덱스명, 컬럼명], df.iloc[index 번호, column 열번호]


# In[109]:


import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'이름' : [ '서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)

# '이름' 열을 새로운 인덱스로 지정하고, df 객체에 변경사항 반영
df.set_index('이름', inplace=True)
print(df)
print('\n')
print()
# 데이터프레임 df의 특정 원소 1개 선택 ('서준'의 '음악' 점수)
a = df.loc['서준', '음악']
print(a)
print()
b = df.iloc[0, 2]
print(b)
print('\n')

# 데이터프레임 df의 특정 원소 2개 이상 선택 ('서준'의 '음악', '체육' 점수) 
c = df.loc['서준', ['음악', '체육']]
print(c)
print()
d = df.iloc[0, [2, 3]]
print(d)
print()
e = df.loc['서준', '음악':'체육']
print(e)
print()
f = df.iloc[0, 2:]
print(f)
print('\n')

# df의 2개 이상의 행과 열로부터 원소 선택 ('서준', '우현'의 '음악', '체육' 점수) 
g = df.loc[['서준', '우현'], ['음악', '체육']]
print(g)
h = df.iloc[[0, 1], [2, 3]]
print(h)
i = df.loc['서준':'우현', '음악':'체육']
print(i)
j = df.iloc[0:2, 2:]
print(j)


# In[110]:


df


# In[ ]:


a


# In[119]:



# 데이터프레임 df의 특정 원소 1개 선택 ('서준'의 '음악' 점수)
a = df.loc['서준', '음악']
print(a)
b = df.iloc[0, 2] 
print(b)
print('\n')

# 데이터프레임 df의 특정 원소 2개 이상 선택 ('서준'의 '음악', '체육' 점수) 
c = df.loc["서준",["수학","영어"]]
print("서준의 수학 영어")
print(c)
print()
d = df.iloc[2, [0, 1]]
print("인아의 수학 영어")
print(d)
print()


print()
f = df.iloc[0, 2:]
print("서준의 음악부터 나머지")
print(f)
print('\n')

# df의 2개 이상의 행과 열로부터 원소 선택 ('서준', '우현'의 '음악', '체육' 점수) 
g = df.loc[['서준', '우현'], ['음악', '체육']]

print("서준과 우현의 음악 체육 점수")
h = df.iloc[[0, 1], [2, 3]]
print(h)
print()
i = df.loc['서준':'우현', '음악':'체육']
print("서준과 우현의 음악 체육")
print(i)
j = df.iloc[0:2, 2:]
print()
df["국어"]=80
print(df)
df.loc["행추가"]= 0
df


# #문제 
# 1.이름을 입력받아 데이터프레임의 인덱스로 저장하고
#  키와 몸무게를 입력 받아 저장합니다
#  키와 몸무게가 float이 입력도지 않으면 다시 입력 받고
#  이름에 'q'가 입력되면 입력 조오료
# 2. 찾고자 하는 이름을 입력받아 키와 몸무게를 출력하고
# 키와 몸무게를 다시 입력받아 기존의 값을 변경하세요
# 3. 평균보다 적은 키를 가진 사람의 이름과 키를 출력하세요

# In[207]:



def input_info():
    while True:
        infos = input("키와 몸무게를 입력하세요 : ").split()
        for i, info in enumerate(infos):
            if type(infos[1]) != float:
                break
            else:
                info[i] = float(infos)
        return info
    
def search_name(name, df):
    for i, idx_name in enumerate(list(df.index)):
        if name == idx_name:
            return i
def print_data(what_name, df):
    print("{} : {}, {}".format(df.iloc[what_name], df.iloc[what_name,0], df.iloc[what_name,1]))


# In[208]:


import pandas as pd
df = pd.DataFrame(columns=["키","몸무게"])

input_info()
while True:
    name = input("이름을 입력하세요 : ")
    if name == "q":
         break
        
    data_name = input_info()
    df.loc[name] = data_name  #입력된 name 인덱스로 키와 몸무게 추가

print(df)


# In[209]:


s_name = input("찾고자 하는 이름을 입력하세요 : ")
what_name = search_name(s_name, df)
print(what_name)
print(df.iloc[what_name,:])
print("검색한 내용",df.iloc[what_name])
if what_name == search_name(s_name, df):
    print_data(what_name, df)
    value = input_info()
    df.iloc[what_name] = value
print("{}과{}".format(df.iloc[0],df.iloc[1]))
df

