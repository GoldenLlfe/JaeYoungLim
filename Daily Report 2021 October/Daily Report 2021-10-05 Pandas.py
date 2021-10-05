#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

df = pd.read_csv('./stock-data.csv')
df


# In[6]:


df['New_Date'] = pd.to_datetime(df['Date']) #데이터 형변환을 위해 칼럼 추가
df.info()
#새로운 컬럼 New_Date를 인덱스로 설정
df.set_index('New_Date', inplace=True)
#기존의 컬럼 Date를 삭제
df.drop('Date', axis=1, inplace=True)

# 1.데이터 확인 df.info() , df.haed()
# 2.날짜형으로 형변환
# 3.시계열 데이터를 인덱스로 지정
# 4.기존의 데이터 삭제
# 5.자료형 및 데이터 확인 df.info(), df.head()


# In[25]:


df.reset_index(inplace=True) #index reset : 인덱스 제거


# In[26]:


df.drop('level_0', axis=1, inplace=True)


# In[29]:


#df 속성을 이용하여 new_Date 열의 년월일 정보를 년 월일로 구분
df['Year']= df['New_Date'].dt.year
df['Month']= df['New_Date'].dt.month
df['Day']= df['New_Date'].dt.month
df.head()


# In[33]:


import matplotlib as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[ : , ['age','fare']] #모든 row를 가져오고, 컬럼은 나이와 등급만
df.head(5)


# In[36]:


#사용자 정의 함수 :10을 더하는 함수
def add_10(a):
    return a+10

#두 객체의 합을 구하는 함수
def add_two_obj(a,b):
    return a+b


# In[37]:


#시리즈 객체에 10을 더하는 함수를 적용 ->함수명(값...)
df['new_age'] =df['age'].apply(add_10) #add_10(값 : df['age'의 원소의 값])


# In[39]:


df['add_obj'] = df['age'].apply(add_two_obj, b=10)
df.head(2)


# In[38]:


df.head(2)


# In[40]:


df['age'].apply(lambda x:x+10) # lambda함수를 적용


# In[46]:


#시리즈.apply(함수명):시리즈 각 원소에 함수를적용
#데이터프레임.applymap(함수명) : 데이터프레임에 함수 적용, 각 원소에 적용
df_map = df.applymap(add_10)
#데이터프레임의 각 열에 대해서 함수 매핑
result=df.apply(add_10, axis=0) #데이터프레임에 apply(함수명,axis=0) 각 열에 매핑함수

#최대값 - 최소값
df.apply(lambda x:x.max()-x.min())
result_row = df.apply(add_10, axis=1) #각 행에 함수 매핑
print(result_row.head(2))
print( df.head(2))


# In[44]:


result.head(2)


# In[47]:


df = titanic.loc[ : , ['age','fare']]
df.head(3)


# In[48]:


df.applymap(add_10).head(3)


# In[49]:


df.apply(add_10, axis=0).head(3)


# In[50]:


def min_max(x):
    return x.max() - x.min()


# In[51]:


df.apply(min_max, axis=1)


# In[52]:


df.apply(min_max, axis=0)


# In[55]:


df['add'] = df.apply(lambda x: x['age'] + x['fare'],axis=1)


# In[56]:


df.head()


# In[64]:


#pipe 함수 알아보기
df=titanic.loc[:,['age','fare']]

#각 열의 NaN 찾기 - 데이터프레임을 전달하면 데이터프레임을 반환
def missing_value(x):
    return x.isnull()

#각 열의 NaN 개수 반환 - 데이터프레임을 전달하면 시리즈 반환
def missing_count(x):
    return missing_value(x).sum()

#데이터프레이므이 총 NaN개수 - 데이터프레임을 전달하면 값을 반환
def total_number_missing(x):
    return missing_count(x).sum()


# In[68]:


result_series=df.pipe(missing_count)
result_df = df.pipe(missing_value)
print(result_df)
print(result_series)


# In[65]:


result_value = df.pipe(total_number_missing)
result_value


# In[71]:


titanic.columns
titanic.head(2)


# In[75]:


titanic.columns.values


# In[79]:


df = titanic.loc[0:4, 'survived':'age']
#열 이름의 리스트 만들기
columns = list(df.columns.values)
columns


# In[85]:


#열 이름을 알파벳 순으로 정렬
new_columns=sorted(columns)
new_columns

#열 이름을 정렬한 데이터프레임 생성
df_sorted = df[new_columns]
df_sorted

#열 이름을 역으로 정렬하기
df_reversed_sort=df[list(reversed(columns))]
df_reversed_sort


# In[86]:


list(reversed(columns))


# In[91]:


#열 분리
df = pd.read_excel('./주가데이터.xlsx')
df


# In[92]:


df.info()


# In[93]:


df.head()


# In[96]:


# 1.연월일을 문자열로 변경
df['연월일'] = df['연월일'].astype('str')
# 2.연월일을 '-'기준으로 split
dates = df['연월일'].str.split('-')
dates
# 3.데이터프레임에 '연', '월', '일' 컬럼을 추가
#리스트에서 각 리스트의 동일 위치의 원소를 추출하고자 할 경우
#              리스트.str.get(위치인덱스)
df['연'] = dates.str.get(0)  #dates의 원소 리스트의 0번째 인덱스 값을 가져온다
#원래는 df['연'] = dates[0][0] ...dates[0][1]이런식으로 해야함
#하지만 시리즈의 경우 str.get() 함수가 있어서 편히 할 수 있음
df['월'] = dates.str.get(1)
df['일'] = dates.str.get(2)
df


# In[100]:


print(type(df['연']))
df.dtypes


# In[102]:


s_list = pd.Series([[1,2,3],['abc','a','c'],['10','20']])
s_list.str.get(1)  #잘 보면 각 리스트이 1번째 원소를 추출함


# In[110]:


mask_age = (titanic.age >= 10) & (titanic.age< 20)



df_teenage = titanic.loc[mask_age, :]
df_teenage.describe()
df_teenage.info()


# In[111]:


#나이가 10세 미만 60세 이사인 승객의 age와 fare와 sex와 class 컬럼정보만 출력

mask_age = (titanic.age < 10) | (titanic.age>= 60)
df_alive = titanic.loc[mask_age,['age','fare','sex','class','alive']]
df_alive


# In[119]:


#나이가 10세 미만이고 여성인 승객만 따로 서너택
df_underteen_female = titanic.loc[(titanic.age < 10) & (titanic.sex=='female'), : ]


# In[120]:


df_underteen_female


# In[124]:


#함께 탑승한 형제 또는 배우자 수가 3,4,5인 승객만 따로 추출 -불린 인덱싱
#sibsp

mask2 = (titanic.sibsp == 3) | (titanic.sibsp == 4) | (titanic.sibsp == 5)
df_sibsp = titanic.loc[mask2, :]
df_sibsp


# In[126]:


#isin() 메서드를 활용하여 간편하게 추출
mask3 = titanic['sibsp'].isin([3,4,5])
df_isin = titanic.loc[mask3,['age','fare','sex','sibsp']]
df_isin.head()


# In[ ]:


#원하는 데이터만 추출 : boolean mask 활용, isin() 메서드 활용


# In[127]:


#데이터프레임 합치기: pandas.concat(데이터프레임 리스트)
df1 = pd.DataFrame({'a': ['a0', 'a1', 'a2', 'a3'],
                    'b': ['b0', 'b1', 'b2', 'b3'],
                    'c': ['c0', 'c1', 'c2', 'c3']},
                    index=[0, 1, 2, 3])
 
df2 = pd.DataFrame({'a': ['a2', 'a3', 'a4', 'a5'],
                    'b': ['b2', 'b3', 'b4', 'b5'],
                    'c': ['c2', 'c3', 'c4', 'c5'],
                    'd': ['d2', 'd3', 'd4', 'd5']},
                    index=[2, 3, 4, 5])


# In[132]:


df3 = pd.concat([df1,df2])
df3
#기존의 인덱스를 무시하고 새로운 인덱스 부여
df4 = pd.concat([df1,df2], ignore_index=True)
df4

#열로 붙이기
df5 = pd.concat([df1,df2],axis=1)
df5

#join키워드
#기본은 outer고 모든걸 다보여준다
#inner는 교집합인것만 보여준다
df6 = pd.concat([df1,df2],axis=1, join='inner')
df6


# In[138]:


#데이터프레임과 시리즈 연결/붙이기 : pd.concat(데이터프레임/시리즈,데이터프레임2/시리즈2)
#시리즈 생성
sr1 = pd.Series(['e0', 'e1', 'e2', 'e3'], name='e')
sr2 = pd.Series(['f0', 'f1', 'f2'], name='f', index=[3, 4, 5])
sr3 = pd.Series(['g0', 'g1', 'g2', 'g3'], name='g')

#df1과 sr1을 좌우 열(row) 방향으로 연결하기 (index 없음)
df_s1 = pd.concat([df1,sr1],axis=1)
df_s1

#df2와 sr2를 좌우 열방향으로 연결 (시리즈에 인덱스 존재 함)
df_s2 = pd.concat([df2,sr2],axis=1)
df_s2

#sr1과 sr3을 좌우 열 방향으로 연결하기 결과:데이터프레임
s1_s3 = pd.concat([sr1,sr3],axis=1,sort=True) #데이터가 간단해서 sort를 해도 똑같음
s1_s3
#sr1과 sr3을 좌우 행 방향으로 연결하기 결과:시리즈
s1_s3_col = pd.concat([sr1,sr3],axis=0)
s1_s3_col
s1_s3


# In[144]:


#주식 데이터를 가져와서 데이터프레임 만들기
df1 = pd.read_excel('./stock_price.xlsx')
df2 = pd.read_excel('./stock_valuation.xlsx')
df2


# In[141]:


len()


# In[149]:


#데이터프레임 합치기 -교집합
merge_inner = pd.merge(df1,df2)
merge_inner

#데이터프레임 합치기 -합집합
#df1과 df2를 합치는데 합집합을 id컬럼을 기준으로 합친다
merge_outer = pd.merge(df1,df2, how='outer', on='id')
merge_outer

#데이터프레임 합치기 -왼쪽프레임 기준, 키 값 분리, how='left', left_on=컬럼명
merge_left = pd.merge(df1,df2, how='left', left_on='id',right_on='id')
merge_left


# In[151]:


#데이터프레임 합치기 -왼쪽프레임 기준, 키 값 분리, how='left', left_on=컬럼명
merge_right = pd.merge(df1,df2, how='right', left_on='id',right_on='id')
merge_right


# In[152]:


#불린 인덱싱과 결합하여 원하는 데이터 찾기
price = df1[df1['price']<50000]
value = pd.merge(price,df2)
value


# In[ ]:


df1.set_index('id',inplace=True)
df2.set_index('id',inplace=True)


# In[158]:


#데이터프레임 결합(join)
df3= df1.join(df2,how='left') #how = left right outer inner
df3


# In[159]:


titanic['class'].unique()


# In[162]:


df = titanic.loc[:,['age','sex','class','fare','survived']]
len(df)

#class열을 기준으로 분할
grouped=df.groupby(['class'])
grouped.head()


# In[164]:


grouped.get_group('Third')


# In[168]:


df.loc[df['class']=='Third', :] #이렇게 해도 되는데 group을 쓰는 이유는 나중에 연산을 할 때 의미가 있음

#각 그룹에 대한 모든 열의 표준편차를 집계하여 데이터프레임으로 반환
std_all=grouped.std()
std_all

#각 그룹에 대한 fare열의 표준편차를 집계하여 시리즈로 반환
grouped.fare.std()


# In[174]:


grouped_two = df.groupby(['class','sex'])
#grouped_two 그룹 객체에 연산 메소드 적용
average_two = grouped_two.mean()
average_two

#grouped_two 그룹 객체에서 개별 그룹 선택하기
grouped_two.get_group(('Third','female'))

#여러 함수를 각 열에 동일하게 적용하게 집계
agg_all=grouped.agg(['min','max'])
#print(agg_all)

#각 열마다 다른 함수를 적용하여 집계
agg_sep = grouped.agg({'fare':['min','max'],'age':'mean'})
agg_sep


# In[176]:


#데이터 개수가 200개 이상인 그룹만을 필터링
#하여 데이터프레임으로 반환
grouped.filter(lambda x:len(x) >= 200)
#age열의 평균이 30보다 작은 그룹만을
#필터링하여 데이터프레임으로 반환
grouped.filter(lambda x:x.age.mean() <30)

grouped.head()


# In[178]:


grouped = df.groupby(['class','sex'])

gdf = grouped.mean()
gdf

#gdf도 결국 데이터 프레임이다

#class 값이 First인 행을 선택
gdf.loc['First']

#class 값이 First이고 sex값이 male인 행을 선택
gdf.loc[('First','male')]

#sex 값이 male인 행을 선택: 
#그룹함수결과.xs(그룹컬럼의 값, level=그룹컬럼명)
gdf.xs('male',level='sex')


# In[182]:


#행,열,값, 집계에 사용할 열을 1개씩 지정-평균 집계

pdf1 = pd.pivot_table(df,  #피벗할 데이터 프레임
                      index='class',  #행위치에 들어갈 열
                      columns='sex', #열 위치에 들어갈 열
                      values='age',   #데이터로 사용할 열
                      aggfunc='mean')  #데이터 집계 함수
pdf1
pdf2 = pd.pivot_table(df,  #피벗할 데이터 프레임
                      index='class',  #행위치에 들어갈 열
                      columns='sex', #열 위치에 들어갈 열
                      values='age',   #데이터로 사용할 열
                      aggfunc=['mean','sum','count'])  #데이터 집계 함수
pdf2
pdf3 = pd.pivot_table(df,  #피벗할 데이터 프레임
                      index=['class','sex'],  #행위치에 들어갈 열
                      columns='survived', #열 위치에 들어갈 열
                      values=['age','fare'],   #데이터로 사용할 열
                      aggfunc=['mean','max']) 
pdf3.xs('First')
                  #('First','female'), ('male',level='sex')
                  #(('Second', 'male'), level=[0,'sex'])
                  #('mean',axis=1), ('mean','age'), axis=1)
                  #(('max','fare', 0), level=[0,1,2],axis=1)


# In[183]:


#머신 러닝

#데이터 준비
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[184]:


# CSV 파일을 데이터프레임으로 변환
df = pd.read_csv('./auto-mpg.csv', header=None)
df


# In[186]:


# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 
df.horsepower.unique()  #'?' 라는 값이 존재하네..?라는걸 분석을 통해 알아야함


# In[187]:


# horsepower 열의 자료형 변경 (문자열 ->숫자)
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환


# In[188]:


df.info()


# In[198]:



# 분석에 활용할 열(속성)을 선택 (연비, 실린더, 출력, 중량)
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]

# ndf 데이터를 train data 와 test data로 구분(7:3 비율)
X=ndf[['cylinders', 'horsepower']]  #독립 변수 X    (2차원 배열인 경우가 많다)
y=ndf['mpg']     #종속 변수 Y


# In[199]:


# train data 와 test data로 구분(7:3 비율)
# 1.train과 typese
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=10) 

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_test.shape)   
print('\n')
print("종속 데이터 : ",y_train.shape)


# In[200]:


'''
Step 5: 비선형회귀분석 모형 - sklearn 사용, 원하는 모델 선택
'''

# sklearn 라이브러리에서 필요한 모듈 가져오기 
from sklearn.linear_model import LinearRegression      #선형회귀분석
from sklearn.preprocessing import PolynomialFeatures   #다항식 변환 

# 다항식 변환 
poly = PolynomialFeatures(degree=2)               #2차항 적용
X_train_poly=poly.fit_transform(X_train)     #X_train 데이터를 2차항으로 변형

print('원 데이터: ', X_train.shape)
print('2차항 변환 데이터: ', X_train_poly.shape)  
print('\n')


# In[201]:


# train data를 가지고 모형 학습
pr = LinearRegression()     #모형/모델을 생성
pr.fit(X_train_poly, y_train)   #모형/모델을 학습 결과

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
X_test_poly = poly.fit_transform(X_test)       #X_test 데이터를 2차항으로 변형
r_square = pr.score(X_test_poly,y_test)   #예측한 값과 원 데이터의 정확도 계산
print(r_square)
print('\n')


# In[203]:


# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력 
y_hat_test = pr.predict(X_test_poly)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_train, y_train, 'o', label='Train Data')  # 데이터 분포
ax.plot(X_test, y_hat_test, 'r+', label='Predicted Value') # 모형이 학습한 회귀선
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close()


# In[204]:


# 모형에 전체 X 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교 
X_ploy = poly.fit_transform(X)
y_hat = pr.predict(X_ploy)

plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(y, label="y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()


# In[206]:


from sklearn.datasets import load_iris

dataset = load_iris()


# In[210]:


X=dataset['data']
y=dataset['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, #test size가 30이면 30개를 말하는 거고 0.3 소수점이면 30%를 뜻한다
                                                    test_size=0.7) 


# In[ ]:


i

