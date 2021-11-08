#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import cv2

print(cv2.__version__)


# In[3]:


img = cv2.imread('./fig/puppy.bmp', cv2.IMREAD_COLOR)

if img is None:
    print('failed')
    sys.exit()
    
cv2.namedWindow('puppy', cv2.WINDOW_AUTOSIZE)
cv2.imshow('puppy', img)

while True:
    if cv2.waitKey() == 27:
        break
    
cv2.destroyAllWindows()


# In[2]:


import matplotlib.pyplot as plt
imgBGR = cv2.imread('./fig/puppy.bmp')

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.axis('off')
plt.show()


# In[2]:


import glob


# In[4]:


img_list = glob.glob('./mywallpaper/*jpg')

print(img_list)

if img_list is None:
    print('failed')
    sys.exit()
    
cv2.namedWindow('wallpaper', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('wallpaer', cv2.WND_PROP_FULLSCREEN,
                     cv2.WINDOW_FULLSCREEN)

indx=0
while True:
    img = cv2.imread(img_list[indx])
    cv2.imshow('wallpaper', img)
    
    if cv2.waitKey(1000) == 27:
        break
        
    indx += 1
    
    if indx >= 13:
        indx = 0
        
cv2.destroyAllWindows()        


# In[8]:


img1 = cv2.imread('fig/puppy.bmp', cv2.IMREAD_COLOR)
img2 = img1
img3 = img1.copy()


cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
# print(type(img))
# print(img.shape)
# print(img.dtype)


while True:
    if cv2.waitKey() == 27:
        break
    
cv2.destroyAllWindows()


# In[7]:


img1 = np.zeros((640,640,3), dtype=np.uint8)
img2 = np.ones((640,640), dtype=np.uint8)*255 #흰색
img3 = np.full((640,640,3),(0,0,255), dtype=np.uint8)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)

while True:
    if cv2.waitKey() == 27:
        break
    
cv2.destroyAllWindows()


# In[10]:


src = cv2.imread('fig/airplane.bmp')
mask = cv2.imread('fig/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('fig/field.bmp')


cv2.copyTo(src,mask,dst)
cv2.imshow('src',src)
cv2.imshow('mask',mask)
cv2.imshow('dst',dst)

while True:
    if cv2.waitKey() == 27:
        break
    
cv2.destroyAllWindows()


# In[13]:


sunglass = cv2.imread('./fig/imgbin_sunglasses_1.png', cv2.IMREAD_UNCHANGED)
src = cv2.imread('fig/puppy.bmp', cv2.IMREAD_COLOR)

sunglass = cv2.resize(sunglass, (300,150))

mask = sunglass[:,:,-1]
glass=sunglass[:,:,0:3]

h,w = mask.shape
crop = src[120:120+h, 220:220+w] #눈 사이즈만큼 잘라낸다
#cv2.copyTo(glass, mask, crop)
crop[mask>0] = (0,0,255) #안경에 색넣기
cv2.imshow('src',src)
cv2.imshow('mask',mask)
cv2.imshow('crop',crop)

while True:
    if cv2.waitKey() == 27:
        break
    
cv2.destroyAllWindows()


# In[15]:


img = np.ones((600,1200,3),dtype=np.uint8)*255

#cv2.line(img, (50,50),(300,50),(0,0,255),2,cv2.LINE_8)
cv2.circle(img,(100,100), 50,[0,0,0 ,5],cv2.LINE_AA)
cv2.line(img,(175,50),(175,150),(0,0,0),5,cv2.LINE_AA)
cv2.rectangle(img, (75,175),(200,250),(0,0,0),-1)

cv2.line(img,(275, 50),(400,50),(0,0,0),3,cv2.LINE_AA)
cv2.line(img,(400, 50),(275,250),(0,0,0),3,cv2.LINE_AA)
cv2.line(img,(330, 145),(400,250),(0,0,0),3,cv2.LINE_AA)
cv2.line(img,(415, 50),(415,250),(0,0,0),3,cv2.LINE_AA)
cv2.line(img,(415, 170),(500,170),(0,0,0),3,cv2.LINE_AA)
cv2.line(img,(500, 50),(500,250),(0,0,0),3,cv2.LINE_AA)

cv2.circle(img,(600,100), 50,[0,0,0 ,5],cv2.LINE_AA)
cv2.line(img,(650,700),(800,850),(0,0,0),5,cv2.LINE_AA)
cv2.line(img,(650,75),(700,75),(0,0,0),5,cv2.LINE_AA)
cv2.line(img,(650,110),(700,110),(0,0,0),5,cv2.LINE_AA)
cv2.line(img,(700,50),(700,140),(0,0,0),5,cv2.LINE_AA)
cv2.circle(img,(650,210), 50,[0,0,0 ,5],cv2.LINE_AA)
cv2.imshow('myname',img)

# text = 'opencv version' + cv.__version__
# cv2.putText(img, text, (500,100), cv2.FONT_HERSHEY_SIMPLEX,
#            0.8, (255,0,0), 3, cv2.LINE_AA)


while True:
    if cv2.waitKey() == 27:
        break
    
cv2.destroyAllWindows()


# In[23]:


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('no bueno')
    cap.release()
    sys.exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
#또는 *'DIVX'를 써도 된돠, *는 리스트를 풀어준다는 명령어임
out = cv2.VideoWriter('my_webcam_record_edge.avi',fourcc, fps,
                      (width, height))
# out = cv2.VideoWriter('my_webcam_record.avi',fourcc, fps,
#                       (width, height))
# print(type(width))
# print(width, height)
# print(fps)
while True:
    ret, frame = cap.read()
    
    if not ret:
        print('no bueno')
        break
    #########################
    frame= cv2.Canny(frame,50,150)
    #########################
    cv2.imshow('cap', frame)
    out.write(frame)
    
    if cv2.waitKey(2000) == 27:
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




