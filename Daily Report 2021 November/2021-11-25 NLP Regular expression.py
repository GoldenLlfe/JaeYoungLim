#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# ### 정규화

# In[37]:


# 숫자 찾기
example = '2021 11 25 thursday'
#print(re.findall("['2','0','2','1','1','1','2','5']", example))
print(re.findall("[0-9]",example))
#숫자 아닌 것 찾기
#print(re.findall("['','','','','','','','','t','h','u','r','s','d','a','y']", example))
print(re.findall("[^0-9]",example))


# In[35]:


#이메일 주소 찾기
example = '제 이메일은 hreeee@yonsei.ac.kr 입니다. hreeee@gmail.com으로 변경되었습니다.'
print(re.findall('[a-z]+@[a-z]+.[a-z]+.[a-z]+', example))


# In[39]:


#url 찾기
example = 'https://www.google.com/에 네이버를 검색하시면 https://www.naver.com/이 상단에 노출 됩니다.'
print(re.findall('https://[a-z]{3}.[a-z]+.[a-z]+/', example))


# In[42]:


# 파일 찾기
example = '제 바탕화면 보시면 sonata01.png 파일과 avante02.jpg 파일 그리고 socar2021.gif 파일이 존재합니다.'
print(re.findall('[a-z0-9]+\.[(png)|(jpg)|(gif)]+', example))


# In[43]:


def pad_punctuation(sentence, punc):
    for p in punc:
        sentence = sentence.replace(p, " "+p+" ")
    return sentence

sentence = "Hi, my name is john."

print(pad_punctuation(sentence, [".","?","!",","]))


# In[44]:


sentence = "Fisrt, open the first chapter."

print(sentence.lower())


# In[46]:


sentence = "First, open the first chapter."

print(sentence.upper())


# In[47]:


sentence = "He is a ten-year-old boy."
sentence = re.sub("([^a-zA-Z.,?!])"," ",sentence)

print(sentence)


# In[55]:


def cleaning_text(text, punc, regex):
    #노이즈 유형 1
    for p in text:
        text=text.replace(p, " "+p+" ")
    #노이즈 유형 2
    
    # 노이즈 유형 3
    text = re.sub("regex"." ",text)
    return text


# In[56]:


corpus = """
It was in the spring of 1890 that I learned to speak. 
The impulse to utter audible sounds had always been strong within me. 
I used to make noises, keeping one hand on my throat while the other hand felt the movements of my lips. 
I was pleased with anything that made a noise and liked to feel the cat purr and the dog bark. 
I also liked to keep my hand on a singer's throat, or on a piano when it was being played. 
Before I lost my sight and hearing, I was fast learning to talk, but after my illness it was found that I had ceased to speak because I could not hear. 
I used to sit in my mother's lap all day long and keep my hands on her face because it amused me to feel the motions of her lips; and I moved my lips, too, although I had forgotten what talking was. My friends say that I laughed and cried naturally, and for awhile I made many sounds and word-elements, not because they were a means of communication, but because the need of exercising my vocal organs was imperative. There was, however, one word the meaning of which I still remembered, WATER. I pronounced it "wa-wa." Even this became less and less intelligible until the time when Miss Sullivan began to teach me. I stopped using it only after I had learned to spell the word on my fingers.
"""


print(cleaning_text(corpus, [".", "?", "!", ","], "([^a-zA-Z.,?!\n])"))


# In[ ]:




