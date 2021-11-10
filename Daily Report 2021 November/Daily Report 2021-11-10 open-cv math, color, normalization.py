#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import sys
import cv2 as cv

cv.__version__


# In[2]:


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print('open failed')
    cap.release()
    sys.exit()

cv.namedWindow('webcam')

wideth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = cv.VideoWriter_fourcc(*'DIVX')

out = cv.VideoWriter('2021-11-10_practice_edge_vid.avi',fourcc, fps, 
                     (wideth,height))

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print('read fialed')
        break
    
    
    ###################################
    edge = cv.Canny(frame, 50, 100)
    edge = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    
    
    ##################################
    cv.write(edge)
    cv.imshow('webcam',frame)
    cv.imshow('edge',edge)
    
    while True:
        if cv.waitKey(10) == 27:
            break
    

cap.release()
out.release()
cv.destroyAllWindows()


# In[7]:



oldx = -1
oldy = -1

def call_mouse(event, x, y, flags, param):
    global oldx, oldy
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)
        oldx, oldy
    elif event == cv.EVENT_MOUSEMOVE:
        if flags == cv.EVENT_FLAG_LBUTTON:
            cv.line(img, (oldx,oldy), (x,y),(255,0,0),
                   6, cv.LINE_AA)
            cv.imshow('img',img)
            oldx, oldy = x,y

img = np.ones((400,600,3), dtype=np.uint8)*255




cv.namedWindow('img')
cv.setMouseCallback('img', call_mouse, img)

cv.imshow('img',img)

while True:
    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()


# In[9]:


def call_trackbar(pos):
    img[:] = pos
    cv.imshow('img',img)
#     print(pos)

img = np.ones((400,600,3), dtype=np.uint8)

cv.namedWindow('img')
cv.createTrackbar('level','img',5,255,call_trackbar)

cv.imshow('img',img)

while True:
    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()


# In[7]:


src = cv.imread('fig/lenna.bmp',cv.IMREAD_COLOR)

dst = cv.add(src, (100,100,100,0))
# dst = np.clip(src + 100., 0, 255).astype(np.uint8)

dst = cv/cvtColor(src, cv.COLOR_BGR2HSV)


cv.imshow('src',src)
cv.imshow('dst',dst)


while True:
    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()


# In[16]:


img1 = cv.imread('fig/lenna256.bmp',cv.IMREAD_GRAYSCALE)
img2 = np.zeros((256,256),np.uint8)


cv.circle(img2, (128,128),100,100,-1)
cv.circle(img2, (128,128),50,50,-1)

dst1 = cv.add(img1, img2)
dst2 = cv.addWeighted(img1, 0.5, img2, 0.5, 0.0)
dst3 = cv.subtract(img1, img2)
dst4 = cv.absdiff(img1, img2)


cv.imshow('img1',img1)
cv.imshow('img2',img2)
cv.imshow('dst1',dst1)
cv.imshow('dst2',dst2)
cv.imshow('dst3',dst3)
cv.imshow('dst4',dst4)

while True:
    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()


# In[21]:


src = cv.imread('fig/flowers_rgb.jpg',cv.IMREAD_COLOR)
print(src.shape)

# b, g, r = cv.split(src)
# b= src[:,:,0]
# g= src[:,:,1]
# r= src[:,:,2]

src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
h , s, v = cv.split(src_hsv)

# h = np.clip(h*2., 0, 255).astype(np.uint8)

cv.imshow('src',src)
# cv.imshow('b',b)
# cv.imshow('g',g)
# cv.imshow('r',r)

cv.imshow('hue',h)
cv.imshow('saturation',s)
cv.imshow('value',v)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[22]:


import matplotlib.pyplot as plt


# In[29]:


src = cv.imread('fig/lenna.bmp',cv.IMREAD_COLOR)
# calcHist(images, channels, mask, 
#          histSize, ranges[, hist[, accumulate]])

src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

# b, g, r = cv.split(src)
h, s, v =cv.split(src_hsv)


# hist_b = cv.calcHist([b],[0],None, [256],[0,256])
# hist_g = cv.calcHist([g],[0],None, [256],[0,256])
# hist_r = cv.calcHist([r],[0],None, [256],[0,256])

hist_h = cv.calcHist([h],[0],None,[256],[0,256])
hist_s = cv.calcHist([s],[0],None,[256],[0,256])
hist_v = cv.calcHist([v],[0],None,[256],[0,256])


cv.imshow('src',src)



while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()

plt.plot(hist_h, color='b')
plt.plot(hist_s,color='g')
plt.plot(hist_v,color='r')
plt.show()


# ### histogram normalization

# In[34]:


src = cv.imread('fig/Hawkes.jpg',cv.IMREAD_GRAYSCALE)

cv.imshow('src',src)

# smin, smax, _, _ = cv.minMaxLoc(src)

# dst = np.clip(255*(src - smin)/(smax-smin), 
#               0, 255).astype(np.uint8)

# print('min : ', smin)
# print('max : ', smax)


#normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]])
dst = cv.normalize(src, None,0,255, cv.NORM_MINMAX,-1)


cv.imshow('src',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# ### histogram equalization

# In[37]:


src = cv.imread('fig/Hawkes.jpg',cv.IMREAD_GRAYSCALE)

#equalizeHist(src[, dst])
dst_equal = cv.equalizeHist(src)
dst_norm = cv.normalize(src, None,0,255, cv.NORM_MINMAX,-1)

cv.imshow('src',src)
cv.imshow('dst_n',dst_norm)
cv.imshow('dst_e',dst_equal)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[39]:


src = cv.imread('fig/field.bmp', cv.IMREAD_COLOR)

src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

h, s, v = cv.split(src_hsv)

v = cv.equalizeHist(v)

#merge(mv[, dst])
dst_hsv = cv.merge([h, s, v])

dst = cv.cvtColor(dst_hsv, cv.COLOR_HSV2BGR)

cv.imshow('src',src)
cv.imshow('dst',dst)
# cv.imshow('dst_n',dst_norm)
# cv.imshow('dst_e',dst_equal)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[42]:


src = cv.imread('fig/field.bmp')

src_YCrCb= cv.cvtColor(src, cv.COLOR_BGR2YCrCb)

y, cr, cb = cv.split(src_YCrCb)

y = cv.equalizeHist(y)

dst_ycrcb = cv.merge([y, cr, cb])

dst = cv.cvtColor(dst_ycrcb, cv.COLOR_YCrCb2BGR)

cv.imshow('src',src)
cv.imshow('dst',dst)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[ ]:




