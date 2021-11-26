#!/usr/bin/env python
# coding: utf-8

# ## 공백 기반 토큰화

# 한 문장에서 단어의 수는 어떻게 정의할 수 있을까요? "그녀는 나와 밥을 먹는다" 라는 문장이 주어지면 공백 기준으로 나누어 1: 그녀는 2: 나와 3: 밥을 4: 먹는다 4개 단어로 이루어졌다고 단정 지을 수 있을까요? 어쩌면 1: 그녀 2: 는 3: 나 4: 와 5: 밥 6: 을 7: 먹는다 처럼 잘게 잘게 쪼개어 7개 단어로 이루어졌다고 할 수도 있겠죠? 그것은 우리가 정의할 토큰화 기법이 결정할 부분이랍니다!
# 
# 문장을 어떤 기준으로 쪼개었을 때, 쪼개진 각 단어들을 토큰(Token) 이라고 부릅니다. 그리고 그 쪼개진 기준이 토큰화(Tokenization) 기법에 의해 정해지죠. 이번 스텝에서는 토큰화의 여러 가지 기법에 대해 배워보도록 하겠습니다.
# 
# 자연어의 노이즈를 제거하는 방법 중 하나로 우리는 Hi, 를 Hi와 ,로 나누기 위해 문장부호 양옆에 공백을 추가해 주었습니다. 그것은 이 공백 기반 토큰화를 사용하기 위해서였죠! 당시의 예제 코드를 다시 가져와 공백을 기반으로 토큰화를 진행해 보겠습니다.

# In[15]:


corpus = """
in the days that followed i learned to spell in this uncomprehending way a great many words ,  among them pin ,  hat ,  cup and a few verbs like sit ,  stand and walk .  
but my teacher had been with me several weeks before i understood that everything has a name . 
one day ,  we walked down the path to the well house ,  attracted by the fragrance of the honeysuckle with which it was covered .  
some one was drawing water and my teacher placed my hand under the spout .  
as the cool stream gushed over one hand she spelled into the other the word water ,  first slowly ,  then rapidly .  
i stood still ,  my whole attention fixed upon the motions of her fingers .  
suddenly i felt a misty consciousness as of something forgotten a thrill of returning thought  and somehow the mystery of language was revealed to me .  
i knew then that  w a t e r  meant the wonderful cool something that was flowing over my hand .  
that living word awakened my soul ,  gave it light ,  hope ,  joy ,  set it free !  
there were barriers still ,  it is true ,  but barriers that could in time be swept away . 
"""
# HINT : split()을 사용하여 공백토큰화를 수행하세요.
tokens = corpus.split()

print("문장이 포함하는 Tokens:", tokens)


# # 형태소 기반 토큰화

# 하지만 우리에겐 영어 문장이 아닌 한국어 문장을 처리할 일이 더 많을 것이고, 한국어 문장은 공백 기준으로 토큰화를 했다간 엉망진창의 단어들이 등장하는 것을 알 수 있습니다. 문장부호처럼 "은 / 는 / 이 / 가" 양옆에 공백을 붙이자구요? 글쎄요... 가로 시작하는 단어만 해도 가면, 가위, 가족, 가수... 의도치 않은 변형이 너무나도 많이 일어날 것 같네요!
# 
# 이를 어떻게 해결할 수 있을까요? 정답은 형태소에 있습니다. 어릴 적 국어 시간에 배운 기억이 새록새록 나시나요? 상기시켜드리면 형태소의 정의는 아래와 같습니다.
# 
# (명사) 뜻을 가진 가장 작은 말의 단위.
# 
# 예를 들어, 오늘도 공부만 한다 라는 문장이 있다면, 오늘, 도, 공부, 만, 한다 로 쪼개지는 것이 바로 형태소죠. 한국어는 이를 활용해 토큰화를 할 수 있습니다!
# 
# 한국어 형태소 분석기는 대표적으로 아래 두 가지가 사용됩니다.
# 
# KoNLPy 파이썬 한국어 패키지
# kakao/khaiii
# KoNLPy는 내부적으로 5가지의 형태소 분석 Class를 포함하고 있습니다. Khaiii까지 총 6개나 되는 형태소 분석기들은 특수한 문장(띄어쓰기 X / 오탈자) 처리 성능, 속도 측면에서 차이를 보입니다. 천하무적인 것은 (아직은) 없으니, 각 분석기를 직접 테스트해보고 적합한 것을 선택해 사용하면 됩니다.

# In[3]:


get_ipython().system('git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git ')


# In[12]:


get_ipython().system('pip install konlpy')


# In[5]:


cd Mecab-ko-for-Google-Colab/


# In[8]:


get_ipython().system('bash install_mecab-ko_on_colab190912.sh')


# In[47]:


from konlpy.tag import Mecab


# In[52]:


pip install konlpy


# In[50]:


from eunjeon import Mecab


# In[53]:


from konlpy.tag import Mecab


# In[49]:


mecab = Mecab()


# In[4]:


# None자리에 문장을 넣어보고 토큰화 결과를 출력해보세요.

# 예시문장 : 자연어처리가너무재밌어서밥먹는것도가끔까먹어요
print(mecab.morphs('자연어처리가너무재밌어서밥먹는것도가끔까먹어요'))


# In[8]:


# None자리에 문장을 넣어보고 토큰화 결과를 출력해보세요.

# 예시문장 : 자연어처리가너무재밌어서밥먹는것도가끔까먹어요
print(mecab.morphs('자연어처리가너무재밌어서밥먹는것도가끔까먹어요'))


# In[54]:


from konlpy.tag import Hannanum,Kkma,Komoran,Mecab,Okt
tokenizer_list = [Hannanum(),Kkma(),Komoran(),Mecab(),Okt()]

kor_text = '코로나바이러스는 2019년 12월 중국 우한에서 처음 발생한 뒤 전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다.'

for tokenizer in tokenizer_list:
    print('[{}] \n{}'.format(tokenizer.__class__.__name__, tokenizer.pos(kor_text)))


# ## 인코딩

# ![%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png](attachment:%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png)

# In[37]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


print(tf.__version__)


# In[32]:


pip install tensorflow


# In[38]:


from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
  'I love my dog',
  'I love my cat'
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences) # 문자 데이터를 입력받아서 리스트의 형태로 변환
word_index = tokenizer.word_index # 토큰별 단어에 index를 매핑시켜준다.
print(word_index)


# In[16]:


import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# In[17]:


path_to_file = './korean-english-park.train.ko'


# In[23]:


cd "myreposit/Natural Language Processing NLP"


# In[24]:


with open(path_to_file, "r", encoding='utf-8') as f:
  raw = f.read().splitlines()

print("Data Size", len(raw))
print("Example:")
for sen in raw[0:100][::20]: print(">>", sen)


# In[25]:


min_len = 999
max_len = 0
sum_len = 0

for sen in raw:
    length = len(sen)
    if min_len > length : min_len = length
    if max_len < length : max_len = length
    sum_len += length
print('문장의 최단길이: ',min_len)
print('문장의 최장길이: ',max_len)
print('문장의 평균길이:', sum_len//len(raw))


# In[26]:


sentence_length = np.zeros((max_len),dtype=np.int)
for sen in raw:
    sentence_length[len(sen)-1] += 1
plt.bar(range(max_len), sentence_length, width=1.0)
plt.title('sentence length distribution')
plt.show()


# 길이가 1인 문장은 어떤지 보기
# 특정 길이에 해당하는 문자보기

# In[27]:


def check_sentence_with_length(raw,length):
    count = 0
    
    for sen in raw:
        if len(sen) == length:
            print(sen)
            count +=1
            if count > 100: return
            
check_sentence_with_length(raw, 1)


# In[28]:


for idx, _sum in enumerate(sentence_length):
    if _sum > 1500:
        print("outlier index : ",idx+1)


# In[29]:


min_len = 999
max_len = 0
sum_len = 0

cleaned_corpus = list(set(raw)) #set 사용해서 중복 제거
print('data size:',len(cleaned_corpus))

for sen in cleaned_corpus:
    length = len(sen)
    if min_len > length : min_len = length
    if max_len < length : max_len = length
    sum_len += length
print('문장의 최단길이: ',min_len)
print('문장의 최장길이: ',max_len)
print('문장의 평균길이:', sum_len//len(cleaned_corpus))


# In[30]:


sentence_length = np.zeros((max_len),dtype=np.int)
for sen in cleaned_corpus:   #중복이 제거된 코퍼스
    sentence_length[len(sen)-1] += 1
plt.bar(range(max_len), sentence_length, width=1.0)
plt.title('sentence length distribution')
plt.show()


# In[31]:


max_len = 150
min_len = 10

filtered_corpus = [s for s in cleaned_corpus if (len(s) < max_len) & (len(s) >= min_len)]

sentence_length = np.zeros((max_len), dtype=np.int)
for sen in filtered_corpus: # 필터가 적용된 코퍼스
    sentence_length[len(sen)-1] += 1

plt.bar(range(max_len), sentence_length, width =1.0)
plt.title("Sentence Length Distribution")
plt.show()


# ### 공백 기반 토큰화

# In[34]:


# Quiz : 정제된 데이터를 공백 기반으로 토큰화하여 list에 저장한 후, tokenize()함수를 사용해 단어 사전과 Tensor데이터를 얻으세요!
# 그리고 단어 사전의 크기를 확인하세요.

def tokenize(corpus):

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus) # 문자 -> 리스트

    tensor = tokenizer.texts_to_sequences(corpus)  # 텍스트 -> 시퀀스
    tensor = tf.preprocessing.sequence.pad_sequences(tensor, padding='post') # 패딩처리 , padding='post'

    return tensor, tokenizer


# In[35]:


# 정제된 데이터를 공백 기반으로 토큰화하여 저장하는 코드를 직접 작성해보세요.
split_corpus = []

for kor in filtered_corpus:
    split_corpus.append(kor.split())
    # 코드작성해주세요.


# In[72]:


from eunjeon import Mecab
mecab = Mecab()


# In[73]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing


# In[74]:


split_tensor, split_tokenizer = tokenize(split_corpus)  #split_tensor 에는 시퀀스가 숫자로 저장 되어있다
print("Split Vocab Size :", len(split_tokenizer.index_word))


# In[42]:


for idx, word in enumerate(split_tokenizer.word_index):
    print(idx, ":", word)

    if idx > 10: break


# ### 형태소 기반 토큰화

# In[56]:


pip install mecab


# In[59]:


pip install eunjeon


# In[67]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing


# In[68]:


from eunjeon import Mecab
mecab = Mecab()


# In[69]:


# 위에서 사용한 코드를 활용해서 Mecab단어 사전을 만들어주세요
#힌트! mecab.morphs() ->형태소 분석 수행

def mecab_split(sentence): #형태소 분석
    return mecab.morphs(sentence)
    
mecab_corpus = []

#mecab 단어장 생성
for kor in filtered_corpus:
    mecab_corpus.append(mecab_split(kor))


# In[70]:


mecab_tensor, mecab_tokenizer = tokenize(mecab_corpus)
print("Mecab Vocam size: ", len(mecab_tokenizer.index_word))


# ### 위 코드의 실행 결과는
# Mecab Vocab size : 52279  이다

# 1) tokenizer.sequences_to_texts() 함수를 사용하여 Decoding<br> 2) tokenizer.index_word 를 사용하여 Decoding
# 
# 두 가지 방법으로 mecab_tensor[100] 을 원문으로 되돌려 보세요! (여기서 띄어쓰기는 고려하지 않습니다!)

# In[75]:


# Case 1 : mecab_tokenizer.sequences_to_texts()

# Case 1
texts = mecab_tokenizer.sequences_to_texts(mecab_tensor[100]) # 코드 작성

print(texts[0])


# 위 코드의 정답은 
# 중국 관영 언론 은 2000 가지 음식 명 을 제시 한 170 페이지 분량 의 책 을 베이징 호텔 에 제공 했 다 .

# In[76]:


# Case 2 : mecab_tokenizer.index_word[]
sentence = ""

for w in mecab_tensor[100]:
    if w -- 0: continue
    sentence += mecab_tokenizer.index_word[w] + " "

print(sentence)


# 위 코드의 정답은
# 중국 관영 언론 은 2000 가지 음식 명 을 제시 한 170 페이지 분량 의 책 을 베이징 호텔 에 제공 했 다 .

# ### 사전에 없는 단어의 문제

# 코로나바이러스는 2019년 12월 중국 우한에서 처음 발생한 뒤
# 전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다.
# 
# →
# 
# <unk>는 2019년 12월 중국 <unk>에서 처음 발생한 뒤
# 전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다.

#  만약 위 문장을 영문으로 번역해야 한다면 어떨까요? 핵심인 단어 `코로나바이러스`와 `우한`을 모른다면 제대로 해낼 수 있을 리가 없습니다. 이를 **OOV(Out-Of-Vocabulary)** 문제라고 합니다. 이처럼 **새로 등장한(본 적 없는) 단어에 대해 약한 모습**을 보일 수밖에 없는 기법들이기에, 이를 해결하고자 하는 시도들이 있었습니다. 그리고 그것이 우리가 다음 스텝에서 배울, ***Wordpiece Model***이죠!

# In[ ]:




