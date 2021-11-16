#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import cv2 as cv


# In[3]:


src = cv.imread('fig/blue_eyes.png',cv.IMREAD_GRAYSCALE)


# kernel = np.ones((3,3),np.float32)/9.0

dst = cv.blur(src, (3,3))

cv.imshow('src',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[4]:


src = cv.imread('fig/blue_eyes.png',cv.IMREAD_GRAYSCALE)



dst = cv.blur(src, (7,7))
dst_gau = cv.GaussianBlur(src, (0,0), 1)

cv.imshow('src',src)
cv.imshow('dst_gau',dst_gau)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[5]:


src = cv.imread('fig/blue_eyes.png',cv.IMREAD_GRAYSCALE)

src_blur = cv.GaussianBlur(drc,(0,0),1)


dst = cv.blur(src, (7,7))
dst_gau = cv.GaussianBlur(src, (0,0), 1)
dst_22


cv.imshow('src',src)
cv.imshow('dst_gau',dst_gau)


while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[11]:


src = cv.imread('fig/puppy.bmp')

m = np.array([[1,0,200],[0,1,-100]],np.float32)



dst= cv.warpAffine(src, m, (0,0))


cv.imshow('puppy',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[16]:


src = cv.imread('fig/rose.jpg')
print(src.shape)

dst1 = cv.resize(src, (2*src.shape[0],2*src.shape[1]), interpolation=cv.INTER_NEAREST)
dst2 = cv.resize(src, (2*src.shape[0],2*src.shape[1]), interpolation=cv.INTER_LINEAR)
dst3 = cv.resize(src, (2*src.shape[0],2*src.shape[1]), interpolation=cv.INTER_CUBIC)


cv.imshow('src',src)
cv.imshow('dst1',dst1)
cv.imshow('dst2',dst2)
cv.imshow('dst3',dst3)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[22]:


src = cv.imread('fig/rose.jpg')

rad = 20*np.pi/180
rad2 = 70*np.pi/180

m = np.array([[np.cos(rad),np.sin(rad),200],
              [-np.sin(rad),np.cos(rad),-100]],np.float32)
m2 = np.array([[np.cos(rad),np.sin(rad2),200],
              [-np.sin(rad),np.cos(rad2),-100]],np.float32)

dst= cv.warpAffine(src, m, (0,0))
dst2= cv.warpAffine(src, m2, (0,0))


cv.imshow('src',src)
cv.imshow('dst',dst)
cv.imshow('dst2',dst2)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[30]:


src = cv.imread('fig/lenna.bmp', cv.IMREAD_GRAYSCALE)

kernel_dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)

dst_dx = cv.filter2D(src, -1,kernel_dx)

#Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
#ddepth: 출력영상의 데이터 타입 (-1)
dx = cv.Sobel(src, cv.CV_32F,1,0)
dy = cv.Sobel(src, cv.CV_32F,0,1)  #, delta = 128 삭제함


mag=np.clip(cv.magnitude(dx,dy),0,255).astype(np.uint8)


dx = np.clip(dx,0,255).astype(np.uint8)
dy = np.clip(dy,0,255).astype(np.uint8)

dst = np.zeros(mag.shape[:2], np.uint8)
dst[mag>100]=255


cv.imshow('src',src)
# cv.imshow('dst',dst_dx)
# cv.imshow('dx',dx)
# cv.imshow('dy',dy)
cv.imshow('mag',mag)
cv.imshow('dst',dst)


while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[40]:


src = cv.imread('fig/building.jfif',cv.IMREAD_GRAYSCALE)


dx = cv.Sobel(src, cv.CV_32F, 1, 0)
dy = cv.Sobel(src, cv.CV_32F, 0, 1)

mag=np.clip(cv.magnitude(dx,dy),0,255).astype(np.uint8)
dst_sobel = np.zeros(mag.shape, np.uint8)
dst_sobel[mag>100]=255

dst_canny = cv.Canny(src,200,255)


cv.imshow('src',src)
cv.imshow('mag',mag)
cv.imshow("dst_canny",dst_canny)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### Binary Thresholing

# In[48]:


src = cv.imread('fig/cells.png', cv.IMREAD_GRAYSCALE)


if src is None:
    print('img load fail')
    sys.exit()
    
retval, dst1 = cv.threshold(src, 100,255,cv.THRESH_BINARY)
retval2, dst2 = cv.threshold(src, 200,255,cv.THRESH_BINARY)

def call_track(pos):
    retval, dst = cv.threshold(src,100,255,cv.THRESH_BINARY)
    cv.imshow(dst)

cv.imshow('src',src)
cv.imshow('dst1',dst1)
cv.createTrackbar('level','dst',0, 255, call_track)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[5]:


src = cv.imread('fig/rice.png',cv.IMREAD_GRAYSCALE)

thres, dst = cv.threshold(src,0,255,cv.THRESH_OTSU)


dst2= np.zeros(src.shape, np.uint8)

bw = src.shape[1]//4
bh = src.shape[0]//4



for y in range(4):
    for x in range(4):
        src_ = src[y*bh+(y+1)*bh, x*bw:(x+1)*bw]
        dst_= dst2[y*bh+(y+1)*bh, x*bw:(x+1)*bw]#한칸씩 옆으로 가면서 otsu를 진행
        cv.threshold(src_, 0, 255, cv.THRESH_OTSU, dst_) #진행한 otsu를 dst_에 저장/업데이트
        
        
cv.imshow('src',src)
cv.imshow('dst',dst)
cv.imshow('dst2',dst2)



while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[9]:


src = cv.imread('fig/sudoku.jpg',cv.IMREAD_GRAYSCALE)


# dst = cv.adaptiveThreshold(src, 255,
#                            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                           cv.THRESH_BINARY, 9,5)

for y in range(4):
    for x in range(4):
        src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_= dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]#한칸씩 옆으로 가면서 otsu를 진행
        cv.threshold(src_, 0, 255, cv.THRESH_OTSU, dst_)

cv.imshow('src',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[14]:


src = cv.imread('fig/rice.png',cv.IMREAD_GRAYSCALE)

dst1= np.zeros(src.shape, np.uint8)

bw = src.shape[1]//4
bh = src.shape[0]//4

for y in range(4):
    for x in range(4):
        src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_= dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]#한칸씩 옆으로 가면서 otsu를 진행
        cv.threshold(src_, 255, cv.THRESH_OTSU, dst_)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
dst2 = cv.morphologyEx(dst1, cv.MORPH_OPEN, kernel, None)
dst_erode = cv.morphologyEx(dst1, cv.MORPH_OPEN, kernel, None)


cv.imshow('src',src)
cv.imshow('dst1',dst1)
cv.imshow('dst2',dst2)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### labelling

# In[26]:


src = cv.imread('fig/keyboard.jpg',cv.IMREAD_GRAYSCALE)

ret, dst = cv.threshold(src, 0, 255,cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
dst_morph=cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)


cnt, labels, stats, centroids = cv.connectedComponentsWithStats(dst_morph)

for i in range(1, cnt): #0은 배경
    x, y, w, h, area = stats[i] #여기서
    #알파벳만 따오고 싶으면 x y w h의 범위를 지정해주거나
    #centroid의 위치를 정해주는 등 여러가지를 해봐야한다
    
    if area > 1000 or area < 140:
        continue
    
    cv.rectangle(dst, (x,y,w,h), (0,0,255),2)



cv.imshow('src',src)
cv.imshow('dst_morph',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[ ]:




