#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 폰트를 인식 못해서 한글 폰트 지정해줌
plt.rcParams['axes.unicode_minus'] = False


# In[10]:


df = pd.read_csv("./auto-mpg.csv", header=None)
df.columns = ['mpg','cylinders','displacement',
              'horsepower','weight','acceleration','model_year',
              'origin','name']
print(df.head(3))
print("\n")
print(df.tail())


# In[11]:


print(df.shape) #데이터프레임의 행과 열의 갯수를 보여줌
print("rows : {}, columns : {}".format(df.shape[0],df.shape[1]),"\n")
df.info()


# In[12]:


print(df.describe())
df_desc = df.describe().loc[['count','std','mean'],['mpg','weight']]
df_desc


# In[13]:


print(df.count())  #전체 컬럼들의 갯수
print(df['mpg'].value_counts(),"\n")  #같은 값들이 갯수를 출력
print("같은 나라들 갯수 출력","\n", df['origin'].value_counts(),"\n","위의 'origin의 고유값-같은값은 3개가 있고 같은 제조국가를 가지고 있는 차들의 갯수가 오른쪽에 표시가 됐다'")


# In[14]:


# 객체.mean/median/min/max/std/corr() 또는 객체[”열이름”]/[(상관계수는 2개 이상이 되어야 성립해서)열이름 리스트].mean/median/min/max/std/corr():
print(df.mean(),"\n")
print("mean of mpg","\n",df["mpg"].mean(),"\n")
print("correaltion between mpg and weight","\n",df[['mpg','weight']].corr(),"\n")


# In[15]:


df = pd.read_excel('./남북한발전전력량.xlsx')
df


# In[16]:


#판다스 내장 그래프 활용
df = pd.read_excel('./남북한발전전력량.xlsx')
df_ns = df.iloc[[0,5],3:]
df_ns.index = ['남한','북한']
# df_ns.columns.map(int)  #컬럼(행)명을 int로 변경 그걸 df_ns의 columns/행들에 넣어준다
df_ns.columns = df_ns.columns.map(int)  #int로 변경된 컬럼들을 df_ns의 columns/행들에 넣어준다
df_ns


# 데이터 전처리 : 컬럼의 데이터 타입 변경, 원하는 정보만 추출, 데이터 가공, NaN 데이터 처리 ...

# In[17]:


# df_ns.plot() 이걸 실행하면 x축과 y축이 뭔지 몰라서 이상하게 나온다
df_ns_t = df_ns.T  #사람이 이해하기 쉽게 x축에 년도를을 놓고 y축에 남한과 북한의 발전전력량을 놓았다 즉, 왼쪽에 columns을 아래에 row를 배치한것.
df_ns_t.plot()


# In[18]:


df_ns_t = df_ns.T  # 데이터프레임의 행과 열을 바꿈
print("df_ns의 선 그래프")
print(df_ns_t.plot(kind='line'))
print("수직 막대 그래프")
print(df_ns_t.plot(kind='bar'))
print("히스토그램")
print(df_ns_t.plot(kind='hist'))

df_mpg = pd.read_csv("./auto-mpg.csv", header=None)
df_mpg.columns = ['mpg','cylinders','displacement',
             'horsepower','weight', 'acceleration',
             'model_year','origin','name']
print("박스 플롯")
print(df_mpg[['mpg','cylinders']].plot(kind='box'))
print("산점도 그래프")
print(df_mpg.plot(x="weight", y="mpg", kind="scatter"))


# In[19]:


print(df_mpg[['mpg','cylinders']].plot(kind='box'))


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


df = pd.read_excel("./시도별 전출입 인구수.xlsx", engine='openpyxl', header=0)

######시도별 전출입 인구 엑셀 데이터 전처리 과정



#누락값을 앞으 데이터로 채운다
df = df.fillna(method='ffill')#그냥 df.fillna() 를 쓰고 쉬프트+탭을 해서 value가 있는지 확인하고 숫자로 채워도 된다


#서울에서 다른 지역으로 이동한 데이터만 추출하여 정리
mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')  #판다스에서는 and가 아니라 &를 써야한다
df_seoul = df[mask]  #mask객체를 이용하여 값이 True인 데이터만 추출
df_seoul = df_seoul.drop(['전출지별'], axis=1)  #여기서는 어차피 전출지별이 전부 서울이니 필요가 없어서 전출지별 열을 뜻하는 axis=1로 drop 시켜줌
df_seoul.rename({'전입지별' : '전입지'}, axis=1, inplace=True)  #전입지별 열 이름을 전입지로 바꿔주고 원래 데이터를 inplace=True로 바꿔준다
df_seoul.set_index('전입지', inplace=True) #전입지 컬럼을 인덱스로 설정한다.
df_seoul


# In[22]:


#문제 부산에서 다른지역으로 전출한 데이터만 추출해서 df_busan으로 저장
#인덱스는 불필요한 컬럼을 제거하고 전입지별을 전입지로 바꿔서 설정
mask2 = (df['전출지별'] == '부산광역시') & (df['전입지별'] != '부산광역시')
df_busan = df[mask2]
df_busan = df_busan.drop(['전출지별'], axis=1)
df_busan.rename({'전입지별':'전입지'},axis=1, inplace=True)
df_busan.set_index('전입지', inplace=True)
df_busan = df_busan.drop(['세종특별자치시'], axis=0)  #그냥 세종특별자치시 행 지워봄
df_busan


# In[23]:


df_busan=df_busan.T
df_busan.plot()


# In[24]:


df_busan.plot(kind='hist')


# In[25]:


df_busan.plot(kind='barh')


# In[26]:


plt.style.use('ggplot')  #스타일 서식을 'ggplot'으로 지정함 , plot.style.available로 이용 가능한 스타일을 찾아 볼 수 있다
df_1 = df_seoul.loc['경기도'] #전입지(전입지별) 중에서 경기도만 보고 싶어서 경기도 행만 찾음, x,y 축으로 구성됨
plt.plot(df_1.index, df_1.values)  #plt.plot(x축,y축)

plt.title("서울에서 경기도로 전출한 인구수")  #제목
plt.xlabel("연도")   #x축 이름
plt.ylabel("인구수",size=20) #y축 이름을 정해주고 폰트 사이즈를 20으로 지정
plt.xticks(rotation="vertical") #x축 변수를 수직으로 돌려준다 아니면 rotation=숫자 로 몇도를 기울일지 써줘도 된다 45를 쓰면 45도 기울기
plt.legend(labels=["서울 -> 경기"], loc=4) #범례를 정중앙에 지정 쉬프트+탭을 참고해서 정수로 된 코드를 넣어서 지정가능
plt.show()


# In[27]:


#서울에서 충청남도, 경상북도, 강원도로 이동한 인구 데이터만 선택해서 사용
#NaN 데이터 처리
# 년도가 string이므로 int로 변경
#인덱스는 년도로 바꾸기
df = pd.read_excel("./시도별 전출입 인구수.xlsx", header=0)
df=df.fillna(method='ffill')

mask3 = ((df['전출지별'] == '서울특별시') & ((df['전입지별'] == '충청남도') | (df['전입지별'] == '경상북도')|(df['전입지별'] == '강원도')))
df_seoul_to = df[mask3]

df_seoul_to = df_seoul_to.drop(['전출지별'], axis=1)
df_seoul_to.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul_to.set_index('전입지', inplace=True)
df_seoul_to.sort_index(inplace=True)
df_seoul_to.columns = df_seoul_to.columns.map(int)
col_years = list(map(int, range(1970,2018)))
df_2 = df_seoul_to.loc[['충청남도','경상북도','강원도'], col_years]
df_seoul_to


# In[28]:


df_seoul_to=df_seoul_to.T
plt.plot(df_seoul_to)


# In[41]:


fig = plt.figure(figsize=(20,10))
 #add_subplot(행의 갯수, 열의 갯수, 위치)
ax2= fig.add_subplot(2,2,1)
ax3= fig.add_subplot(2,2,2)
ax4= fig.add_subplot(2,2,3)

plt.style.use('ggplot')

ax2.plot(col_years, df_2.loc['충청남도'])
ax3.plot(col_years, df_2.loc['경상북도'])
ax4.plot(col_years, df_2.loc['강원도'])

ax2.set_title("서울->충남 경북 강원 이동 인구", size=30)
ax3.set_xlabel("년도",size=15)
ax4.set_ylabel("머릿수",size=15)
plt.show()


# In[43]:


import seaborn as sns
fig = plt.figure(figsize=(20,10))
 #add_subplot(행의 갯수, 열의 갯수, 위치)
ax2= fig.add_subplot(2,2,1)
ax3= fig.add_subplot(2,2,2)
ax4= fig.add_subplot(2,2,3)
sns.regplot(x=col_years,        #x축 변수
            y=['충청남도'],       #y축 변수
            data=df_2,   #데이터
            ax=ax2)         #axe 객체 - 1번째 그래프 

# 그래프 그리기 - 선형회귀선 미표시(fit_reg=False)
sns.regplot(x=col_years,        #x축 변수
            y=['경상북도'],       #y축 변수
            data=df_2,   #데이터
            ax=ax3,         #axe 객체 - 2번째 그래프        
            fit_reg=False)  #회귀선 미표시


# In[ ]:


fig = plt.figure(figsize=(20,10))
 #add_subplot(행의 갯수, 열의 갯수, 위치)
ax2= fig.add_subplot(2,2,1)
ax3= fig.add_subplot(2,2,2)
ax4= fig.add_subplot(2,2,3)

plt.style.use('ggplot')

ax2.plot(col_years, df_2.loc['충청남도'])
ax3.plot(col_years, df_2.loc['경상북도'])
ax4.plot(col_years, df_2.loc['강원도'])

ax2.set_title("서울->충남 경북 강원 이동 인구", size=30)
ax3.set_xlabel("년도",size=15)
ax4.set_ylabel("머릿수",size=15)

fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# 그래프 그리기 - 선형회귀선 표시(fit_reg=True)
sns.regplot(x='age',        #x축 변수
            y='fare',       #y축 변수
            data=titanic,   #데이터
            ax=ax1)         #axe 객체 - 1번째 그래프 

# 그래프 그리기 - 선형회귀선 미표시(fit_reg=False)
sns.regplot(x='age',        #x축 변수
            y='fare',       #y축 변수
            data=titanic,   #데이터
            ax=ax2,         #axe 객체 - 2번째 그래프        
            fit_reg=False)  #회귀선 미표시

plt.show()


# In[33]:


df_2


# In[ ]:


mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시') 
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

# 서울에서 '충청남도','경상북도', '강원도', '전라남도'로 이동한 인구 데이터 값만 선택
col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도','경상북도', '강원도', '전라남도'], col_years]


# In[ ]:


df_4.plot(kind='area', stacked=False, alpha=0.2, figsize=20,10)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,10))
ax1= fig.add_subplot(2,2,1)
ax2= fig.add_subplot(2,2,2)
ax3= fig.add_subplot(2,2,3)
ax4= fig.add_subplot(2,2,4)

ax1.plot(col_years, df_4.loc['충청남도'])
ax2.plot(col_years, df_4.loc['경상북도'])
ax3.plot(col_years, df_4.loc['강원도'])
ax4.plot(col_years, df_4.loc['전라남도'])

df_4.plot(kind='area', stacked=False, alpha=0.2, figsize=20,10)
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 폰트를 인식 못해서 한글 폰트 지정해줌
plt.rcParams['axes.unicode_minus'] = False


# In[8]:


df = pd.read_excel("./시도별 전출입 인구수.xlsx", header=0)
df=df.fillna(method='ffill')
mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시') 
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul = df_seoul.set_index('전입지', inplace=True)

# 서울에서 '충청남도','경상북도', '강원도', '전라남도'로 이동한 인구 데이터 값만 선택
col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도','경상북도', '강원도', '전라남도'], col_years]
print(df_4)


# In[ ]:




