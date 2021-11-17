#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import cv2 as cv
import numpy as np


# In[4]:


src = cv.imread('fig/puppy.bmp')



# M = np.array([[1,0,100],[0,1,-100]],dtype=np.float32)  

rot = np.array([[1,0,100],[0,1,-100]],dtype=np.float32)  
#확대를 하려면 x와 y를 원하는 n배로 하면된다 예)2,0,100 - x축으로 2배확대

dst = cv.warpAffine(src, M, (0,0))


cv.imshow('src',src)
cv.imshow('dst',dst)


while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[10]:


img = cv.imread('fig/lenna.bmp',cv.IMREAD_GRAYSCALE)

# dx = cv.Sobel(img,cv.CV_32F,1,0,delta=128)
# dy=cv.Sobel(img,cv.CV_32F,0,1,delta=128)

# mag=cv.magnitude(dx,dy)

# mag=np.clip(mag,0,255).astype(np.uint8)

# dst = np.zeros(mag.shape[:2], np.uint8)
# dst[mag>120] = 255

dst = cv.Canny(img, 100,200)

cv.imshow('img',img)
# cv.imshow('dx',dx)
# cv.imshow('dy',dy)
# cv.imshow('mag',mag)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[4]:


src = cv2.imread('./fig/cells.png', cv2.IMREAD_GRAYSCALE)

ret, dst = cv2.threshold(src, 100, 255, cv2.THRESH_OTSU)
# ret, dst_100 = cv.threshold(src, 100,255, cv.THRESH_BINARY)
# ret, dst_200 = cv.threshold(src, 200,255, cv.THRESH_BINARY)

cv2.imshow('src',src)
cv2.imshow('dst',dst)
# cv.imshow('dst200',dst_200)

while True:
    if cv2.waitKey() == 27:
        break
cv2.destroyAllWindows()


# In[11]:


src = cv.imread('fig/rice.png', cv.IMREAD_GRAYSCALE)

dst = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv.THRESH_BINARY,255,0 )

cv.imshow('src',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[19]:


src = cv.imread('fig/circuit.bmp', cv.IMREAD_GRAYSCALE)

struct = cv.getStructuringElement(cv.MORPH_RECT, (3,4))

dst1=cv.erode(src,struct)

dst2=cv.dilate(src, struct,iterations=1)
dst2=cv.erode(dst2,struct,iterations=1)

cv.imshow('src',src)
cv.imshow('dst1',dst1)
cv.imshow('dst2',dst2)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[21]:


src = cv.imread('fig/rice.png', cv.IMREAD_GRAYSCALE)

thres, dst = cv.threshold(src,0,255,cv.THRESH_OTSU)

dst1 = np.zeros(src.shape, np.uint8)

bw = src.shape[1]//4
bh = src.shape[0]//4

for y in range(4):
    for x in range(4):
        src_ = src[y*bh+(y+1)*bh, x*bw:(x+1)*bw]
        dst_= dst1[y*bh+(y+1)*bh, x*bw:(x+1)*bw]#한칸씩 옆으로 가면서 otsu를 진행
        cv.threshold(src_, 0, 255, cv.THRESH_OTSU, dst_) #진행한 otsu를 dst_에 저장/업데이트

struct = cv.getStructuringElement(cv.MORPH_RECT, (3,4))

cv.imshow('src',src)
cv.imshow('dst1',dst1)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[27]:


src = cv.imread('fig/keyboard.jpg',cv.IMREAD_GRAYSCALE)

struct = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

ret, src_binary = cv.threshold(src, 0,255, cv.THRESH_OTSU)
src_binary = cv.morphologyEx(src_binary,cv.MORPH_OPEN,struct)

cnt, labels, stats, centroids = cv.connectedComponentsWithStats(src_binary)

dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)

for i in range(1,cnt):
    (x,y,w,h,area) = stats[i]
    
    if area> 1000 or area<200:
        continue
    
    cv.rectangle(dst, (x,y,w,h), (0,0,255),2)


cv.imshow('src',src)
cv.imshow('src_binary',src_binary)
cv.imshow('dst',dst)


while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ## Detection 감지

# In[1]:


import sys
import cv2 as cv
import numpy as np


# In[7]:


cap = cv.VideoCapture('fig/PETS2000.avi')


_, bg = cap.read()   #여기선 return value를 안 받는다

bg = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)
bg = cv.GaussianBlur(bg, (0,0),1.0)  #영상의 노이즈 제거
fbg = bg.astype(np.float32)  #처음 프레임과 새로들어온 프레임과 비교하기/업데이트 위해 float형으로 변환 

if not cap.isOpened():   #예외처리
    print('video open failed')
    cap.release()
    sts.exit()

while True:
    
    ret, frame = cap.read()
    if not ret:   #예외처리
        print('no return value')
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   #처리를 위해 흑백으로 변환
    frame_gray = cv.GaussianBlur(frame_gray, (0,0), 1.0)   #흑백 영상의 노이즈 제거
    fframe_gray = frame_gray.astype(np.float32)
    
#     fbg = fbg.astype(np.uint8)   #업데이트를 위한 프레임을 처리하기 위해 데이터 타입 변환

    cv.accumulateWeighted(fframe_gray, fbg, 0.005,None)  
    #처음 프레임(bg)와 새로 들어온 frame_gray를 정한 rate마다 업데이트/비교하면서 
    
    bg= fbg.astype(np.uint8)
    diff = cv.absdiff(bg, frame_gray)
    thre, diff_binary = cv.threshold(diff, 30, 500, cv.THRESH_BINARY)   #처음 영상과 업데이트 영상의 차이점을 출력
    
    cnt, labels, stats, area = cv.connectedComponentsWithStats(diff_binary)  #프레임 비교해서 차이 있는 곳에 네모박스 표시
    
    for i in range(1, cnt):
        x, y, w,h, area = stats[i]   #네모박스를 표시하기 위한 파라미터
        
        if area<100:   #어느정도 사이즈 이상/이하로 네모박스 표시 유무 처리
            continue
        
        cv.rectangle(frame, (x, y, w, h),(0,0,255),2)   #빨간 네모박스 표시
        
    
    cv.imshow('frame',frame)
    cv.imshow('background',bg)
    cv.imshow('diff',diff)
    cv.imshow('diff_binary',diff_binary)
    
    if cv.waitKey(30) == 27:
        break
        
cap.release()
cv.destroyAllWindows()


# In[ ]:




