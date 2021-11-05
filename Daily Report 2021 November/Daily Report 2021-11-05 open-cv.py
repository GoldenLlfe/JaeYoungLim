#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


print(cv.__version__)


# In[10]:


img = cv.imread('./fig/puppy.bmp', cv.IMREAD_COLOR)

if img is None:
    print('image read failed')
    sys.exit()
    
cv.namedWindow('puppy', cv.WINDOW_AUTOSIZE)
cv.imshow('puppy',img)

img_resize = cv.resize(img, (1200,600), cv.INTER_AREA)
cv.imwrite('./fig/puppy_resize_1200_600.png', img_resize)


print('img, img_resize = ',img_resize.shape)

while True:
    key=cv.waitKey()
#     if cv.waitKey() == 27:
#          break    #esc를 입력받으면 종료
        if cv.waitKey() == ord('q'):
            break   #q를 입력받으면 종료
cv.destroyAllWindows()


# In[11]:


imgBGR = cv.imread('./fig/puppy.bmp',cv.IMREAD_COLOR)

if imgBGR is None:
    print('read failed')
    sys.exit()
#imgRGB=cv.cvtColor(imgBGR,cv.COLOR_BGR2RGB)

plt.imshow(imgBGR)
plt.axis('off')
plt.show()


# In[2]:


import glob


# In[3]:


img_list = glob.glob('./mywallpaper/*.jpg')

cv.namedWindow('wallpaper', cv.WINDOW_NORMAL)
cv.setWindowProperty('wallpaper', cv.WND_PROP_FULLSCREEN,
                    cv.WINDOW_FULLSCREEN)

indx=1
while True:
    img=cv.imread(img_list[indx])
    
    cv.imshow('wallpaper', img)
    
    if cv.waitKey(1000) == 27:
        break
    indx += 1
    
    if indx >= 13:
        index = 0
        
cv.destroyAllWindows()


# In[17]:


#

img1 = cv.imread('./fig/puppy.bmp', cv.IMREAD_COLOR)
img2 = cv.imread('./fig/puppy_1280_853.jpg', cv.IMREAD_COLOR)

print('img1= ',img1.shape)
print('img2= ',img2.shape)

print('img1 type= ',img1.dtype)
print('img2 type= ',img2.dtype)

# height, width = img1.shape
# height2, width2 = img2.shape[:2]

# print('height & width of img1 =',height,width)
# print('height & width of img2 =',height2,width2)

cv.namedWindow('puppy',cv.WINDOW_NORMAL)
cv.namedWindow('big_puppy',cv.WINDOW_NORMAL)

img1[10:100, 100:300] = 0 
#row방향으로 10부터 200까지, column방향으로 10부터 200까지 0으로 지정
#0 = 검정색
img2[10:100, 100:300] = (0,0,255 )
#rgb 값때문에 3차원이므로 값을 지정하려면 3차원으로 해야한다
#b g r중 r(빨강)이 255니 빨강색으로 칠해진다
cv.imshow('puppy',img1)
cv.imshow('big_puppy',img2)

while True:
    if cv.waitKey() == 27:
        break
    
cv.destroyAllWindows()


# In[25]:


#영상 생성

img1 = np.zeros((320, 640, 3), dtype=np.uint8) #img1에 강아지 이미지를 640에 320사이즈의 3컬러로 지정
#검정색으로 나오지만 컬러이미지이고 마지막에 ,3을 지우면 흑백사진이다(여전히 검은색 이미지)

img2 = np.ones((320,480),dtype=np.uint8)*255
img3 = np.full((320,480,3),(255,0,255),dtype=np.uint8)
img4 = np.random.randint(0,255,(320,480),dtype=np.uint8)

cv.imshow('img1',img1)
cv.imshow('img2',img2)
cv.imshow('img3',img3)
cv.imshow('img4',img4)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[27]:


# 영상 복사

img1 = cv.imread('./fig/puppy.bmp', cv.IMREAD_COLOR)

if img1 is None:
    print('image read failed')
    sys.exit()

img2 = img1
#이렇게 하면 img2가 img1을 복사한것 같지만
#사실 img1이 가리키는 주소를 똑같이 가리키는 것 뿐이다.
#따라서 복사는 아니다

img3 = img1.copy() #이것이 복사

img1[200:300, 240:400] = (0,255,255)

cv.imshow('img1', img1)
cv.imshow('img2', img2)
cv.imshow('img3', img3)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()

1


# In[12]:


#circle(img, center, radius, 
#color[, thickness[, lineType[, shift]]])

#img = np.ones((480,640,3),dtype = np.uint8)*255 #이미지를 하얀색으로

img1 = cv.imread('./fig/puppy.bmp')

img2 = img1[200:400, 300:500]
img3 = img1[200:400, 300:500].copy()

#cv.circle(img, (300,300),100,(0,0,255),10, cv.LINE_AA)
cv.circle(img2, (100,100),50,(0,0,255),3, cv.LINE_AA)
#img2와 1에는 강아지 코부근에 빨간색 원이 그려지지만
# 3번에는 그려지지 않았다

src = cv.imread('./fig/airplane.bmp', cv.IMREAD_COLOR)
mask = cv.imread('./fig/mask_plane.bmp', cv.IMREAD_GRAY)
dst = cv.imread('./fig/field.bmp', cv.IMREAD_COLOR)

if src is None or mask is None or dst is None:
    print('image read fail')
    sys.exit()
    
dst[mask>0] = src[mask>0]
# cv.imshow('img1',img1)
# cv.imshow('img2',img2)
# cv.imshow('img3',img3)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[3]:


src = cv.imread('./fig/puppy.bmp')
sunglass = cv.imread('./fig/imgbin_sunglasses_1.png',cv.IMREAD_UNCHANGED)

sunglass = cv.resize(sunglass, (300,150))

mask = sunglass[:, :,3]
glass = sunglass[:, :,0:3]

h,w = mask.shape[:2]
crop = src[120:120+h, 220:220+w]

cv.copyTo(glass, mask, crop)

cv.imshow('srs',src)
#cv.imshow('sunglass',sunglass)
cv.imshow('mask',mask)
cv.imshow('crop',crop)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()



# In[ ]:




