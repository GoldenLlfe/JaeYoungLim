#!/usr/bin/env python
# coding: utf-8

# In[42]:


# %load graph.py.py
#!/usr/bin/env python

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 폰트를 인식 못해서 한글 폰트 지정해줌
plt.rcParams['axes.unicode_minus'] = False


# In[5]:


df = pd.read_excel('./남북한발전전력량.xlsx', engine= 'openpyxl', convert_float=True)
df


# In[9]:


plt.style.use('ggplot')   # 스타일 서식 지정
plt.rcParams['axes.unicode_minus']=False   # 마이너스 부호 출력 설정

# Excel 데이터를 데이터프레임 변환 
df = pd.read_excel('./남북한발전전력량.xlsx', engine= 'openpyxl', convert_float=True)
df = df.loc[5:9]
df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
df.set_index('발전 전력별', inplace=True)
df=df.T 
df.head()


# In[8]:


# 증감율(변동률) 계산
df = df.rename(columns={'합계':'총발전량'})  #컬럼명 변경
df['총발전량 - 1년'] = df['총발전량'].shift(1)
df['증감율'] = ((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100   
df


# In[3]:


# 2축 그래프 그리기
ax1 = df[['수력','화력']].plot(kind='bar', figsize=(20, 10), width=0.7, stacked=True)  
ax2 = ax1.twinx()  #ax1 위에 덧부여서 그려줌
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=20, 
         color='green', label='전년대비 증감율(%)')  

ax1.set_ylim(0, 500)
ax2.set_ylim(-50, 50)

ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억 KWh)')
ax2.set_ylabel('전년 대비 증감율(%)')

plt.title('북한 전력 발전량 (1990 ~ 2016)', size=30)
ax1.legend(loc='upper left')

plt.show()


# In[12]:


# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')
 
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

# 그래프 객체 생성 (figure에 2개의 서브 플롯을 생성)
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


# In[22]:


titanic


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
 
# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')
 
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

# 그래프 객체 생성 (figure에 3개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
 
# x축, y축에 변수 할당
sns.barplot(x='sex', y='survived', data=titanic, ax=ax1) 

# x축, y축에 변수 할당하고 hue 옵션 추가 
sns.barplot(x='sex', y='survived', hue='class', data=titanic, ax=ax2) 

# x축, y축에 변수 할당하고 hue 옵션을 추가하여 누적 출력
sns.barplot(x='sex', y='survived', hue='class', dodge=False, data=titanic, ax=ax3)       

# 차트 제목 표시
ax1.set_title('titanic survived - sex')
ax2.set_title('titanic survived - sex/class')
ax3.set_title('titanic survived - sex/class(stacked)')

plt.show()


# In[13]:


import folium


# In[16]:


#서울 지도 만들기
seoul_map = folium.Map(location=[37.5,125.8], zoom_start=10)
seoul_map2 = folium.Map(location=[37.5,125.8],tiles='Stamen  Terrain', zoom_start=10)
#지도를 html 파일로 저장하기
seoul_map2.save("./seoul_map2.html")


# In[ ]:


###문제
#iris 데이터를 sns 모듈에서 load하여
#종별로 잎의 넓이와 길이를 산점도와 선그래프로 출력
#legend(종)와 제목("붓꽃의 길이와 넓이와 종의 관계")출력


# In[18]:


import matplotlib.pyplot as plt
import pandas as pd


# In[58]:


import seaborn as sns

iris = sns.load_dataset('iris')
iris.set_index('species', inplace=True)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
x = iris.index
s_length = iris['sepal_length']
s_width = iris['sepal_length']
x


# In[57]:


plt.plot(kind='scatter', x, s_length)
plt.show()


# In[43]:


def scatter_graph(X, Y):
    sns.scatterplot(x=X, y=Y)
def line_graph(X, Y):
    line(x=X, y=Y)

iris = sns.load_dataset('iris')
iris.set_index('species', inplace=True)

x_data = iris.index
y_data = iris["petal_width"]
scatter_graph(x_data, y_data)
plt.plot(x_data, y_data)

y_data = iris["petal_length"]
scatter_graph(x_data, y_data)
plt.plot(x_data, y_data)

plt.title("붓꽃의 잎의 길이와 넓이, 종의 관계")
labels = iris[["petal_width","petal_length"]]
plt.legend(labels)
plt.show()

