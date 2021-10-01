#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

file_path = "./read_csv_sample.csv"

df=pd.read_csv(file_path) #read_csv 컬럼 데이터가 '.'로 구분된 자료를 읽어온다
print(df)
print()
df1 = pd.read_csv(file_path,header=None)
print("df1 ===== ")
print(df1)

df2 = pd.read_csv(file_path, index_col='c0') #index 컬럼을 지정
print("df2 ===")
print(df2)


# read_csv(옵션,...)
#  - path = 파일의 위치 포함한 파일명
#  - sep : 필드를 구분하는 구분자 ','
#  - header : 헤더가 정의되어 있는지 보여줌, 없으면 None
#  - index_col : 인덱스로 사용될 컬럼명, None은 인덱스가 없음을 뜻한다
#  - names : 컬럼 이름으로 사용 될 문자열 리스트
#  - skiprows : 처음 행 부터 skip 하고자 하는 행 수
#  - skipfooter : 마지막 행 부터 skip하고자 하는 행 수
#  - encoding : 텍스트 인코딩 종류를 지정 많이 쓰는 종류가 'utf-8'

# In[111]:


save_file_path = "./sample_to_csv.csv"
df.to_csv(save_file_path)

save_file_path = "./sample_to_csv1.csv"
df.to_csv(save_file_path,index=None) #df의 인덱스는 저장하지 않음
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
#titanic 데이터를 load해서 
#숫자 컬럼만 데이터를 가져와서 데이터를 save_titanic.csv 파일에 저장, 인덱스는 빼고
titanic.dtypes
column_names = titanic.columns
columns = []
for idx, dtype in enumerate(titanic.dtypes):
    if dtype in ['float', 'int64']:
        columns.append(titanic.columns[idx])
titanic_num_col = titanic.loc[ : ,columns]
titanic_num_col.to_csv("./save_titanic_num_col.csv",index=None)


# In[88]:


data_types = titanic.dtypes
column_names = titanic.columns

columns = []  #컬럼명을 리스트로
for idx, dtype in enumerate(data_types):
    if dtype in ['float', 'int64']:
        columns.append(column_names[idx])
        
titanic_select = titanic.loc[ : , columns]  #df.loc[행 또는 배열, 컬럼 또는 배열]
titanic_select.to_csv("./titanic_save_file_tutor.csv",index=None)


# In[121]:


df = pd.read_json("./read_json_sample.json")
import os
print(os.getcwd(),"\n")
print(df.index,"\n")
print("type of df is : ",type(df),"\n")
df


# In[128]:


url ="./sample.html"
tables = pd.read_html(url)
print(tables,"\n")

for i in range(len(tables)):
    print("tables {}".format(i))
    print(tables[i])
    print("\n")

df = tables[1]
df.set_index(['name'], inplace=True)  #name 컬럼을 인덱스로 setting
df

df1 = tables[0]
df1.set_index(['c0'], inplace=True)
print(df1)


# In[132]:


from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

# 위키피디아 미국 ETF 웹 페이지에서 필요한 정보를 스크래핑하여 딕셔너리 형태로 변수 etfs에 저장
url = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'lxml')   
rows = soup.select('div > ul > li'
    
etfs = {}
for row in rows:
    
    try:
        etf_name = re.findall('^(.*) \(NYSE', row.text)
        etf_market = re.findall('\((.*)\|', row.text)
        etf_ticker = re.findall('NYSE Arca\|(.*)\)', row.text)
        
        if (len(etf_ticker) > 0) & (len(etf_market) > 0) & (len(etf_name) > 0):
            etfs[etf_ticker[0]] = [etf_market[0], etf_name[0]]

    except AttributeError as err:
        pass    

# etfs 딕셔너리 출력
print(etfs)
print('\n')


# In[133]:


get_ipython().system('pip install googlemaps')


# In[134]:


## google 지오코딩 API 통해 위도, 경도 데이터 가져오기 

# 라이브러리 가져오기
import googlemaps
import pandas as pd

# my_key = "----발급받은 API 키를 입력-----"

# 구글맵스 객체 생성하기
maps = googlemaps.Client(key='AIzaSyBhMr1XeliRMaUGATRvlDYDl2A0jcH4b44')  # my key값 입력

lat = []  #위도
lng = []  #경도

# 장소(또는 주소) 리스트
places = ["서울시청", "국립국악원", "해운대해수욕장"]

i=0
for place in places:
    i = i + 1
    try:
        print(i, place)
        # 지오코딩 API 결과값 호출하여 geo_location 변수에 저장
        geo_location = maps.geocode(place)[0].get('geometry')
        lat.append(geo_location['location']['lat'])
        lng.append(geo_location['location']['lng'])

    except:
        lat.append('')
        lng.append('')
        print(i)

# 데이터프레임으로 변환하기
df = pd.DataFrame({'위도':lat, '경도':lng}, index=places)
print('\n')
print(df)


# #문 seaborn 에서 dataset "iris"를 불러와서
# # 1. 'species'컬럼을 인덱스로 설정
# # 2. 나머지 데이터의 합과 평균을 데이터 프레임에 추가
# # 3. 변경된 데이터를 파일에 csv 형식으로 저장
# # 4. 저장된 파일을 프로그램으로 불러 옴
# # 5. 불러온 데이터를 출력해서 확인

# In[150]:


import seaborn as sns
import pandas as pd

life = sns.load_dataset('iris')
print(life)
print("문 seaborn 에서 dataset iris 를 불러와서","\n","1. 'species'컬럼을 인덱스로 설정","\n","2. 나머지 데이터의 합과 평균을 데이터 프레임에 추가","\n","3. 변경된 데이터를 파일에 csv 형식으로 저장","\n","4. 저장된 파일을 프로그램으로 불러 옴","\n","5. 불러온 데이터를 출력해서 확인","\n")


# In[157]:


import seaborn as sns
import pandas as pd

life = sns.load_dataset('iris')
# 1. 'species'컬럼을 인덱스로 설정 
life.set_index(['species'], inplace=True)
life


# In[175]:


import seaborn as sns
import pandas as pd

life = sns.load_dataset('iris')
# 1. 'species'컬럼을 인덱스로 설정 
life.set_index(['species'], inplace=True)
life
# 2. 나머지 데이터의 합과 평균을 데이터 프레임에 추가 
total = []
avg = []
len(life)
for value in range(len(life)):
    total.append(life.iloc[value].sum())
    avg.append(life.iloc[value].sum()/len(life.iloc[idx]))
life['Total']=total
life['Average']=avg
# 3. 변경된 데이터를 파일에 csv 형식으로 저장 
life.to_csv("./Iris_species_sample.csv")
# 4. 저장된 파일을 프로그램으로 불러 옴 
life2 = pd.read_csv("./Iris_species_sample.csv")
# 5. 불러온 데이터를 출력해서 확인 
life2

