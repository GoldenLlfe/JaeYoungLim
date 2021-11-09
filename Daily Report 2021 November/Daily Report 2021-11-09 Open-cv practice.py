#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import cv2 as cv


# In[4]:


img = np.full((600,1200,3), 255, np.uint8)

cv.line(img,(50,50),(300,50),(0,0,255),5)
cv.rectangle(img,(50,70),(300,100),(255,0,0),-1)
cv.circle(img,(500,200),150,(0,255,0),5, cv.LINE_AA)

text='Carpe Diem'

cv.putText(img,text,(600,400),cv.FONT_ITALIC,
          2,(0,0,255), 2, cv.LINE_AA)
cv.namedWindow('img')
cv.imshow('img',img)

while True:
    if cv.waitKey() == 27:
        break
    
cv.destroyAllWindows()


# In[3]:


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print('fail')
    cap.release()
    sys.exit()

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = cv.VideoWriter_fourcc(*'DIVX')

out = cv.VideoWriter('edge3.avi',fourcc, fps,
                    (width,height))

cv.namedWindow('edge', cv.WINDOW_AUTOSIZE)
cv.namedWindow('webcam', cv.WINDOW_AUTOSIZE)
while True:
    ret, frame = cap.read()
    
    if not ret:
        print('video read fail')
        break
    
    ###########   #영상 처리할때 이부분을 건드리면 됨
#     edge = cv.Canny(frame, 50, 150)
    edge = cv.cvtColor(edge. cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)#영상 이미지를 흑백으로
    ########
    
    out.write(edge)
    cv.imshow('webcam', frame)
    cv.imshow('edge',edge)
    
    if cv.waitKey(10) == 27:
        break
        

cap.release()
out.release()
cv.destroyAllWindows()


# In[5]:


img = cv.imread('fig/puppy.bmp', cv.IMREAD_GRAYSCALE)

if img is None:
    print('load failed')
    sys.exit()
    
cv.imshow('img',img)

while True:
    key = cv.waitKey()
    
    if cv.waitKey() == 27 or key == ord('q'):
        break
    elif key == ord('i'):
#         img = cv.bitwise_not(img)
        img = ~img  #흑백 이미지를 역상으로 만들때
        cv.imshow('img',img)
        
cv.destroyAllWindows()


# ### 마우스 이벤트

# In[14]:


oldx=-1  #변수를 사용해야해서 그냥 의미없는 걸로 선언
pldy=-1
def call_mouse(event, x, y, flags, param):
    global oldx, oldy
    
    if event == cv.EVENT_LBUTTONDOWN:
#         print('EVENT_LBUTTONDOWN: ',x,y)
        oldx, oldy = x, y
#     elif event == cv.EVENT_LBUTTONUP:
#         print('EVENT_LBUTTONUP: ',x,y)
    elif event == cv.EVENT_MOUSEMOVE:
        if flags == cv.EVENT_FLAG_LBUTTON: #좌클릭 한 상태에서 움직일때
            cv.line(img, (oldx,oldy),(x,y),
                   6, cv.LINE_AA)
            cv.imshow('img',img)
            oldx, oldy = x,y



img = np.ones((400,600,3),dtype=np.uint8)*255

cv.namedWindow('img')
#마우스 함수(콜백함수)를 슬 때 이미지 창안에서 할 건데 
#그 작업을 할 윈도우 창이 없으면 안되니까
#꼭 윈도우를 만드는 명령어 아래에 해야한다

#setMouseCallback(windowName, onMouse [, param])
cv.setMouseCallback('img', call_mouse,img)


cv.imshow('img',img)

while True:
    if cv.waitKey() == 27:
        break
    
cv.destroyAllWindows()


# ### 트랙바 실습

# In[20]:


def call_track(pos):
    img[:] = pos
    cv.imshow('img',img)
#     print(pos)



# img = np.ones((400,600,3),dtype=np.uint8)
img = cv.imread('./fig/imgbin_sunglasses_1.png',cv.IMREAD_UNCHANGED)


#createTrackbar(trackbarName, 
#              windowName, value, count, onChange)
cv.namedWindow('img')
cv.createTrackbar('level','img',0, 255, call_track)

cv.imshow('img',img)

while True:
    if cv.waitKey() == 27:
        break
    
cv.destroyAllWindows()


# In[22]:


def call_track(pos):
    img_glass = img*pos  #아래에서 0이상인 것은 1로 
    #초기화를 해줘서 0과1밖에 없고 pos는 255까지 있으니
    #1*255 까지 된다

    cv.imshow('img', img_glass)


img_alpha = cv.imread('fig/imgbin_sunglasses_1.png', cv.IMREAD_UNCHANGED)

img = img_alpha[:,:,-1]
img[img>0]=1

cv.namedWindow('img')
cv.createTrackbar('level','img',0, 255, call_track)
cv.imshow('img',img)

while True:
    if cv.waitKey() == 27:
        break
    
cv.destroyAllWindows()


# In[29]:


src = cv.imread('./fig/lenna.bmp', cv.IMREAD_COLOR)

if src is None:
    print('fail')
    sys.exit()

# dst = src+100
dst = cv.add(src,(100,100,100,0))
# dst = np.clip(src+100.,0,255).astype(np.uint8)


cv.imshow('img',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
    
cv.destroyAllWindows()


# In[ ]:




