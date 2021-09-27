#!/usr/bin/env python
# coding: utf-8

# In[9]:


curl "https://openapi.naver.com/v1/papago/n2mt" -H "Content-Type: application/x-www-form-urlencoded; charset=UTF-8" -H "X-Naver-Client-Id: 8dFU4S5LPqI1XjgA5L87" -H "X-Naver-Client-Secret: DIBOEtuucc" -d "source=ko&target=en&text=만나서 반갑습니다." -v
import requests


# In[10]:


import requests
text='''Yesterday
All my troubles seemed so far away
Now it looks as though they're here to stay
Oh, I believe in yesterday
Suddenly
I'm not half the man I used to be
There's a shadow hanging over me
Oh, yesterday came suddenly
Why she had to go, I don't know
She wouldn't say
I said something wrong
Now I long for yesterday
Yesterday
Love was such an easy game to play
Now I need a place to hide away
Oh, I believe in yesterday
Why she'''
request_url = "https://openapi.naver.com/v1/papago/n2mt"

headers= {"X-Naver-Client-Id": "8dFU4S5LPqI1XjgA5L87", 
          "X-Naver-Client-Secret":"DIBOEtuucc"}
params = {"source": "en", "target": "ko", "text": text}

response = requests.post(request_url, headers=headers, data=params)
print(type(response.text))

result = response.json()
result
print(result['message']['result']['translatedText'])


# In[13]:


while True:
    text = input("번역할 한글 입력 : ")
    if text == '종료':
        break
    params = {"source": "ko", "target": "en", "text": text}

    response = requests.post(request_url, headers=headers, data=params)

    result = response.json()
    
    print(result['message']['result']['translatedText'])

