#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# In[2]:


img1 = cv.imread('fig/lenna256.bmp', cv.IMREAD_GRAYSCALE)
img2 = np.zeros((256,256),np.uint8)*255

cv.circle(img2,(128,128),100,120,-1)
cv.circle(img2,(128,128),50,30,-1)


dst1 = cv.add(img1, img2)
dst2 = cv.addWeighted(img1,0.75, img2,0.25, 0)
dst3 = cv.subtract(img1, img2)
dst4 = cv.absdiff(img2, img1)


# cv.imshow('img1',img1)
# cv.imshow('img2',img2)

plt.subplot(231), plt.axis('off'), plt.imshow(img1,'gray'), plt.title('img1')
plt.subplot(232), plt.axis('off'), plt.imshow(img2,'gray'), plt.title('img2')
plt.subplot(233), plt.axis('off'), plt.imshow(dst1,'gray'), plt.title('dst1')
plt.subplot(234), plt.axis('off'), plt.imshow(dst2,'gray'), plt.title('dst2')
plt.subplot(235), plt.axis('off'), plt.imshow(dst3,'gray'), plt.title('dst3')
plt.subplot(236), plt.axis('off'), plt.imshow(dst4,'gray'), plt.title('dst4')
plt.show()



# while True:
#     if cv.waitKey() == 27:
#         break
# cv.destroyAllWindows()


# In[3]:


src = cv.imread('fig/flowers_rgb.jpg', cv.IMREAD_COLOR)

src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

h, s, v = cv.split(src_hsv)

b, g, r = cv.split(src)

# cv.imshow('src',src)
# cv.imshow('blue',b)
# cv.imshow('green',g)
# cv.imshow('red',r)

plt.subplot(331),plt.axis('off'),plt.imshow(src), plt.title('src')
plt.subplot(332),plt.axis('off'),plt.imshow(b), plt.title('blue')
plt.subplot(333),plt.axis('off'),plt.imshow(g), plt.title('green')
plt.subplot(334),plt.axis('off'),plt.imshow(r), plt.title('red')
plt.subplot(335),plt.axis('off'),plt.imshow(h), plt.title('hue')
plt.subplot(336),plt.axis('off'),plt.imshow(s), plt.title('saturation')
plt.subplot(337),plt.axis('off'),plt.imshow(v), plt.title('value')
plt.show()


# while True:
#     if cv.waitKey() == 27:
#         break
# cv.destroyAllWindows()


# In[13]:


src = cv.imread('fig/lenna.bmp')

b, g, r = cv.split(src)

hist_b = cv.calcHist([b],[0], None, [256],[0,256])
hist_g = cv.calcHist([g],[0], None, [256],[0,256])
hist_r = cv.calcHist([r],[0], None, [256],[0,256])


hist = cv.calcHist([src],[0], None, [256],[0,256])


plt.plot(hist_b,color='b')
plt.plot(hist_g,color='g')
plt.plot(hist_r,color='r')
plt.plot(hist)
plt.show()

cv.imshow('src',src)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[23]:


src = cv.imread('fig/field.bmp',cv.IMREAD_COLOR)

# MIN, MAX, _, _ =cv.minMaxLoc(src)

src_ycrcb= cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
y, cr, cb = cv.split(src_ycrcb)

y_norm = cv.normalize(y, None, 0, 255, cv.NORM_MINMAX,-1)
y_equal = cv.equalizeHist(y)
# dst_norm = cv.normalize(src,None, 0,255,cv.NORM_MINMAX,-1)
# dst_equal = cv.equalizeHist(src)
# dst1 = cv.add(dst_norm, dst_equal)
dst_y_norm = cv.merge([y_norm, cr, cb])
dst_y_equal = cv.merge([y_equal, cr, cb])

dst_y_norm = cv.cvtColor(dst_y_norm, cv.COLOR_YCrCb2BGR)
dst_y_equal = cv.cvtColor(dst_y_equal, cv.COLOR_YCrCb2BGR)

cv.imshow('src',src)
cv.imshow('dst_y_n',dst_y_norm)
cv.imshow('dst_y_e',dst_y_equal)
# cv.imshow('dst_n',dst_norm)
# cv.imshow('dst_e',dst_equal)
# cv.imshow('dst1',dst1)
              
while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[26]:


src=cv.imread('fig/candies.png', cv.IMREAD_COLOR)

src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

#inRange(src, lowerb, upperb[, dst])
dst_rgb_g = cv.inRange(src, (0,128,0),(100,255,100)) #초록색만 뽑는다 하더라도 다른 색도 조금씩은 뽑을 수 밖에 없다, 애초에 그래프가 그렇게 생겼다
dst_hsv = cv.inRange(src_hsv, (50,170,0),(80,255,255))

cv.imshow('src',src)
cv.imshow('dst_rgb_g',dst_rgb_g)
cv.imshow('dst_hsv',dst_hsv)
while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[28]:


flower=cv.imread('fig/flowers.jpg', cv.IMREAD_COLOR)

flower_hsv = cv.cvtColor(flower, cv.COLOR_BGR2HSV)

flower_hsv_green = cv.inRange(flower_hsv,(50,170,0),(80,255,255))
flower_hsv_blue = cv.inRange(flower_hsv,(100,170,0),(120,255,255))
flower_hsv_red = cv.inRange(flower_hsv,(150,170,0),(179,255,255))

cv.imshow('src',flower)
cv.imshow('red',flower_hsv_red)
cv.imshow('green',flower_hsv_green)
cv.imshow('blue',flower_hsv_blue)


while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### 역 투영 back projection

# In[31]:


src = cv.imread('fig/cropland.png')

x, y, w, h = cv.selectROI(src) #x,y좌표 w너비와 h 높이

src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)

crop = src_ycrcb[y:y+h, x:x+w]
#확률분포(histogram)를 그릴 곳을 지정
#[y:y+h, x:x+w]에 해당하는 공간에 있는 색들을
#지정하고 다음 코드에서 그걸로 히스토그램을 그린다

hist = cv.calcHist([crop], [1,2], None, [256,256],[0,256,0,256])
# 중간에 [256,256]은 y, cr, cb의 3차원 histogram(색의 분포)을 256개로 나눈다는 것이다
#즉 256을 쓴다는 것은 1개 1개 전부다 맞춰져야
#(히스토그램은 확률분포이기 때문에 100% 맞춘다는 뜻) 마스크가 나온다는것
#대충 비슷한걸 맞추고 싶다면 수를 적게쓰면 된다
#예를 들어 32나 64를 쓰면 된다
#calcBackProject(images, channels, hist, ranges, scale[, dst])
backproj = cv.calcBackProject([src_ycrcb],[1,2],hist,[0,256,0,256],1)

dst= cv.copyTo(src, backproj)

cv.imshow('src',src)
cv.imshow('backproj',backproj)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### Mean filter

# In[37]:


src = cv.imread('fig/blue_eyes.png', cv.IMREAD_GRAYSCALE)


kernel_3 = np.ones((3,3),np.float32)/9
kernel_5 = np.ones((5,5),np.float32)/25
# kernel의 필터 크기를 늘릴 수록
#이미지의 섬세함이 줄어간다


src_mean_filter3 = cv.filter2D(src, -1, kernel_3)
src_mean_filter5 = cv.filter2D(src, -1, kernel_5)

cv. imshow('src', src)
cv.imshow('filter3',src_mean_filter3)
cv.imshow('filter5',src_mean_filter5)



while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[ ]:




