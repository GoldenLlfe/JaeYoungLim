#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import sys
import cv2 as cv


# In[8]:


print(cv.__version__)


# In[13]:


#강아지와 고양이 사진 할당
img = cv.imread('./fig/puppy.bmp')
img2 =  cv.imread('./fig/cat.bmp')


# In[23]:


#창-window 열기
cv.namedWindow('puppy',cv.WINDOW_NORMAL)
#cv2,WINDOW_NORMAL: 영상크기를 창 크기에 맞게 지정
#cv2.WINDOW_AUTOSIZE:창크기를 영상크기에 맞춰 변경

#cv.namedWindow('cat')

#'uint8' 부호없는8비트정수
cv.imshow('puppy',img)
#cv.imshow('cat',img2)
while True:
    if cv.waitKey() == ord('a') or cv.waitKey() ==27:
        break  
    #esc(ascii코드 27)이나 a를 입력했을 때 창을 닫는다
#cv.waitKey(1000)  #1000밀리세컨드정도

# 창을 닫는 명령
cv.destroyAllWindows()


# In[14]:


print(type(img))
print(np.shape(img)) 
#이미지를 일반 좌표로 읽으면 (640, 480)이다
#하지만 비트맵이니 행렬로 취급되어 (480,640,3(색깔b g r))이 된것이다


# In[20]:


#영상 저장
img=cv.resize(img, (1280,960),cv.INTER_AREA)
cv.imwrite('./fig/puppy_1280_960.png',img)


# In[3]:


import matplotlib.pyplot as plt


# In[30]:


imgBGR = cv.imread('./fig/puppy.bmp',cv.IMREAD_COLOR)

if img is None:
    print('image read failed')
    sys.exit()
imgBGR = cv.cvtColor(imgBGR,cv.COLOR_BGR2RGB)
    
plt.imshow(imgBGR)
plt.axis('off')
plt.show()  
#색이 이상하게 나오는데 그 이유는 비트맵은 rgb인데
#맷플롯은 bgr이기 대문이다
#cv.COLOR_BGR2RGB를 하면 제대로 나온다

#cv.namedWindow('puppy', cv.WINDOW_AUTOSIZE)
#cv.imshow('puppy',img)

#while True:
#    if cv.waitKey()  == 27:
#        break
#cv.destroyAllWindows()


# In[31]:


imgGray = cv.imread('./fig/puppy.bmp',cv.IMREAD_GRAYSCALE)

if imgGray is None:
    print('imgGray read failed')
    sys.exit()

plt.subplot(121), plt.imshow(imgBGR), plt.axis('off')
plt.subplot(122), plt.imshow(imgGray), plt.axis('off')
plt.show()


# In[4]:


# 이미지 슬라이드 쇼 만들기
import glob


# In[7]:


img_list = glob.glob('./fig/images/*.jpg')
#print(img_list)

cv.namedWindow('scene',cv.WINDOW_NORMAL)
cv.setWindowProperty('scene',cv.WND_PROP_FULLSCREEN,
                    cv.WINDOW_FULLSCREEN)
indx=0
while True:
    img = cv.imread(img_list[indx])
    cv.imshow('scene',img)
    
    if cv.waitKey(1000) == 27:
        break
    indx+=1
    if indx>=5:
        indx=0
# for i in img_list:
#     img = cv.imread(i, cv.IMREAD_COLOR)
#     cv.imshow('scene', img)
    
#     if cv.waitKey(1000) == 27:
#         break
cv.destroyAllWindows()


# In[5]:


#내 배경화면 이미지 슬라이드 쇼
mywallpaper_list = glob.glob('./mywallpaper/*.jpg')
#print(img_list)

cv.namedWindow('mywallpaper',cv.WINDOW_NORMAL)
cv.setWindowProperty('mywallpaper',cv.WND_PROP_FULLSCREEN,
                    cv.WINDOW_FULLSCREEN)
indx=0
while True:
    img = cv.imread(mywallpaper_list[indx])
    cv.imshow('mywallpaper',img)
    
    if cv.waitKey(1000) == 27:
        break
    indx+=1
    if indx>=13:
        indx=0
cv.destroyAllWindows()


# In[12]:


print(np.array(mywallpaper_list[indx]))


# In[ ]:




