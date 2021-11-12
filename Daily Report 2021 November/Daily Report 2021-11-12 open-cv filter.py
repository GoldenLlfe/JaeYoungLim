#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv


# In[3]:


src = cv.imread('fig/field.bmp', cv.IMREAD_COLOR)

# dst_equl = cv.equalizeHist(src)
src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)

y, cr, cb = cv.split(src_ycrcb)
y = cv.equalizeHist(y)

src_ycrcb = cv.merge([y, cr, cb])
src_bgr = cv.cvtColor(src_ycrcb, cv.COLOR_YCrCb2BGR)

cv.imshow('src',src)
cv.imshow('src_bgr',src_bgr)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[11]:


src = cv.imread('fig/candies.png')


#bgr
dst_bgr = cv.inRange(src, (120,0,0),(255,150,150))

# hsv
dst_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
dst_hsv = cv.inRange(dst_hsv, (90,200,0),(135,255,255))

cv.imshow('src',src)
cv.imshow('dst_bgr',dst_bgr)
cv.imshow('dst_hsv',dst_hsv)



while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[12]:


#색 역투영 backprojection: 색을 특정하기 힘들때 그림의 영역을 지정해서 원하는 것만 따오기


img = cv.imread('fig/cropland.png')

x,y,w,h = cv.selectROI(img)

img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

#이미지 영역 잘자내기 crop
crop = img_ycrcb[y:y+h, x:x+w]

#히스토그램 그리기
hist = cv.calcHist([crop], [1,2], None, [64,64],[0,256,0,256])

#역투영
backproj = cv.calcHist([img_ycrcb], [1,2], hist, [0,256,0,256],1)
dst = cv.copyTo(img, backproj)


cv.imshow('img',img)
cv.imshow('backproj',backproj)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### 가우시안 필터

# In[9]:


src = cv.imread('fig/blue_eyes.png', cv.IMREAD_GRAYSCALE)

kernel_3 = np.ones((3,3), dtype=np.float32)/9.
kernel_5 = np.ones((5,5), dtype=np.float32)/25.
kernel_11 = np.ones((11,11),dtype= np.float32)/121.

# dst_3 = cv.filter2D(src,-1, kernel_3 )
# dst_5 = cv.filter2D(src,-1, kernel_5 )
# dst_11 = cv.filter2D(src,-1, kernel_11 )
# dst_blur = cv.blur(src, (5,5)) #mean filter


#GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
dst_gaussian_1 = cv.GaussianBlur(src, (0,0), 1 ) #시그마값=1
dst_mean = cv.blur(src, (7,7)) #49로 나눠야 하는데 위의 가우시안이랑 비슷하다
#가우시안 필터의 커널은 시그마값(편차)으로 조절이 가능하다
dst_gaussian_2 = cv.GaussianBlur(src, (0,0), 2 )

cv.imshow('src',src)
# cv.imshow('dst_3',dst_3)
# cv.imshow('dst_5',dst_5)
# cv.imshow('dst_11',dst_11)
# cv.imshow('dst_mean_blur', dst_blur)
cv.imshow('dst_gaussian_1',dst_gaussian_1)
cv.imshow('dst_gausssian_2',dst_gaussian_2)


while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ## 샤프니스 필터 open-cv에 없어서 만든다

# ### 이미지에 2를 곱해준 뒤 가우시안 필터를 빼주면 된다

# In[11]:


src = cv.imread('fig/blue_eyes.png', cv.IMREAD_GRAYSCALE)

src_gblur = cv.GaussianBlur(src, (0,0), 1)

dst_sharp = cv.addWeighted(src, 2, src_gblur, -1 ,0)
#바로 윗 줄 코드가 원본 이미지에 2를 곱해준뒤
#가우시안에 -1을 붙여서 더해준것

cv.imshow('src',src)
cv.imshow('src_gblur',src_gblur)
cv.imshow('dst_sharp',dst_sharp)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### 미디안 median 필터

# In[20]:


src = cv.imread('fig/blue_eyes.png', cv.IMREAD_GRAYSCALE)
# print(src.shape)  #(790, 1200)


salt_pepper_1 = np.random.choice((0,255), src.shape, p=(0.95, 0.05)).astype(np.uint32) 
#src 픽셀 안에 (0,255): 흰색과 검은색(소금과 후추)를 랜덤하게 뿌리는 것
salt_pepper_2 = np.random.choice((0,255), src.shape, p=(0.95, 0.05)).astype(np.uint32)



src_noise = src - salt_pepper_1 + salt_pepper_2
src_clip = np.clip(src_noise, 0, 255).astype(np.uint8)


cv.imshow('salt_pepper',salt_pepper_1)
cv.imshow('src_noise',src_noise)
    
while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ## 양방향 필터

# In[22]:


src = cv.imread('fig/blue_eyes.png', cv.IMREAD_GRAYSCALE)

src_gaussian = cv.GaussianBlur(src, (0,0),3.0)
src_biateral = cv.bilateralFilter(src, -1,10,3)

cv.imshow('src',src)
cv.imshow('src_g',src_gaussian)
cv.imshow('src_bilateral',src_biateral)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[ ]:




