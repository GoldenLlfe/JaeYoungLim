#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


# In[3]:


math.sin(10)


# In[12]:


math.ceil(10.56)


# In[13]:


math.trunc(10.5)
math.floor(10.56)   #숫자보다 작은 정수 중의 최대값


# In[14]:


#from 모듈명 import 가져오고자 하는 변수 또는 함수
from math import sin, trunc, floor
#from math import +


sin(10)


# In[15]:


import math as m
m.sin(1)
m.cos(1)
m.tan(1)
m.floor(10)


# In[ ]:


#import 모듈명
#import 모듈명 as 약어
from 모듈명 import 변수또는함수


# In[22]:


#random 모듈의 함수
import random
print("# random 모듈")

print("- random(): ",random.random())

print("- uniform(10,20):", random.uniform(10,20))

print("-randrange(10): ", random.randrange(10))

print("- choice([1,2,3,4,5]):",random.choice([1,2,3,4,5]))

print("-shuffle([1,2,3,4,5]):",random.shuffle([1,2,3,4,5]))
      
print("-sample([1,2,3,4,5], k=2)", random.sample(random.sample([1,2,3,4,5], k=2)))


# In[23]:


import sys
# 명령 매개변수를 출력합니다.
print(sys.argv)
print("---")

# 컴퓨터 환경과 관련된 정보를 출력합니다.
print("getwindowsversion:()", sys.getwindowsversion())
print("---")
print("copyright:", sys.copyright)
print("---")
print("version:", sys.version)

# 프로그램을 강제로 종료합니다.
sys.exit()


# In[25]:


import os

print("현재 운영체제:", os.name)
print("현재폴더: ",os.getcwd())
print("현재 폴더 내부의 요소:", os.listdir())

os.mkdir("Hello")
os.rmdir("Hello")

with open("original.txt", "w") as file:
    file.write("hello")
os.rename("original.txt", "new.txt")

os.remove("new.txt")

os.system("dir")


# In[ ]:


import os

print("현재 운영체제 :",os.name)
print("working directory : ", os.getcwd())

print("working directory elements :")
dir_list = os.lisdtdir()
for file_name in dir_list:
    print(file_name)
    
os
#중간에 끊으심


# In[26]:


# datetime 모듈 : date , time 관련된 코드들
import datetime

print("현재 시간 : ", datetime.datetime.now())

now = datetime.datetime.now()  # now는 변수
print("{}년 {}월 {}일 {}시 {}분 {}초".format(now.year,now.month,
                                       now.day, now.hour, now.minute,
                                       now.second))

out_now = now.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}".format(*"년월일시분초"))
print("{}년 {}월 {}일 {}시 {}분 {}초".format(now.year,now.month,
                                       now.day, now.hour, now.minute,
                                       now.second))
print("out_now : ",out_now)

import time
print("시간 정지 ")
time.sleep(5)
print("시간 정지 end")


# In[28]:


#urllib 모듈: 인터넷 주소에서 자료를 가져오는 라이브러리
from urllib import request

target = request.urlopen("https://google.com")

output = target.read()

output  #html 구문을 해석해야함


# In[29]:


#교재 330페이지 비교해보기
import os

def read_folder(path):
    #폴더의 요소 읽어 들이기
    output = os.listdir(path)
    for item in output:
        if os.path.isdir(item):
            read_folder(item)
        else:
            print("파일 : ",item)

read_folder(input("검색하고자 하는 디렉토리 입력 > "))
    
    
    


# In[33]:


from bs4 import BeautifulSoup
from urllib import request
target = request.urlopen("https://google.com")

output = target.read()


# In[37]:


html = '''
<html>
  <head>
    <title>BeautifulSoup test</title>
  </head>
  <body>
    <div id='upper' class='test' custom='good'>
      <h3 title='Good Content Title'>Contents Title</h3>
      <p>Test contents</p>
    </div>
    <div id='lower' class='test' custom='nice'>
      <p>Test Test Test 1</p>
      <p>Test Test Test 2</p>
      <p>Test Test Test 3</p>
    </div>
  </body>
</html>'''

soup = BeautifulSoup(html)
soup.find_all('p')
soup.find('h3')
soup.find('div',class_ ='test')


# In[36]:


for item in soup.select("div"):
    for value in item.select('p'):
        print(value)


# In[46]:


# https://news.v.daum.net/v/20210927112602253
#f12개발자 도구 적극 이용 노가다로 원하는 내용에 뭐가 있는지 봐야함
url = 'https://news.v.daum.net/v/20210927112602253'
#제목을 출력
target = request.urlopen(url)

for item in soup.select("div"):
    for value in item.select('h3'):
        print(value)
        
soup = BeautifulSoup(html)
soup.find_all("이재명")
soup.find('h3')
print(soup.find(value))


# In[48]:


#교수님 버전 내껀 잘 안됨
url = 'https://news.v.daum.net/v/20210927112602253'
soup = BeautifulSoup(request.urlopen(url))

print(soup.find("h3").text)
for item in soup.select("p"):
    print(item.text)


# In[49]:


#함수 데코레이터를 생성
def test(function):
    def wrapper():
        print("ㅎㅇ")
        function()
        print("ㅂㅇ")
    return wrapper

@test  #데코레이터를 붙여 함수를 만든다
def hi_bye():
    print("안부인사")
    
#함수 호출
hi_bye()

