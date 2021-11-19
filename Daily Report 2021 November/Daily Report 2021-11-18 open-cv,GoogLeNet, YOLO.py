#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import cv2 as cv


# In[4]:


cap = cv.VideoCapture('fig/PETS2000.avi')

if not cap.isOpened():
    print('video open failed')
    cap.realeas()
    sys.exit()

ret1, bg = cap.read()  #bg에 원본 영상 저장
bg = cv.cvtColor(bg, cv.COLOR_BGR2GRAY) #연산에 컬러는 문제가 되니 흑백으로
bg = cv.GaussianBlur(bg, (0,0),1.0) #노이제 제거를 위해 가우시안 필어적용
fbg = bg.astype(np.float32) #처리및 연산을 위해 플롯으로 바꿔준다

while True:
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (0,0),1.0)
    
    fgray = frame.astype(np.float32)
    
    cv.accumulateWeighted(fgray, fbg, 0.001) #0.001의 속도로 fbg를 fgray를 업데이트
    #accumulate 이 명령어는 플롯 타입만 받아서 안의 객체들을 플롯으로 줘야함
    bg = fbg.astype(np.uint8) #다시 다른 연산들을 위해
    #정수없는8비트 형으로 바꿔준다
    diff = cv.absdiff(frame, bg)
    _, diff_thres = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
    #100 만큼 차이 나는 것만 표시
    cnt, labels, stats, centroid = cv.connectedComponentsWithStats(diff_thres)
    # centroid 를 중심으로 물체를 잡는다
    
    for i in range(1,cnt):
        x, y, w, h, area = stats[i]
        # 위에 connectedComponetsWithStats에서 받은 값들로 박스 생성
        cv.rectangle(frame, (x, y, w, h),(0,0,255), 2)
    
    cv.imshow('frame',frame)
    cv.imshow('diff',diff)
    cv.imshow('bg',bg)
    cv.imshow('diff_thres',diff_thres)

cap.realeas()
while True:
    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()


# In[6]:


filename = 'googlenet/apple1.png'

img = cv.imread(filename)

if img is None:
    print('imgae read failed')
    sys.exit()
    
model = 'googlenet/bvlc_googlenet.caffemodel'
#사용할 모델
config = 'googlenet/deploy.prototxt'
#모델에 필요한 config 파일

net = cv.dnn.readNet(model, config)

if net.empty():
    print('network read failed')
    sys.exit()


classNames = []
#클래스들을 리스트형태로 만들어준다
with open('googlenet/classification_classes_ILSVRC2012.txt') as f:
    classNames= f.read().rstrip('\n').split('\n')
    
# print(type(classNames))
# print(classNames[10])

blob = cv.dnn.blobFromImage(img,1, (224,224), (104,117, 123),swapRB=False) #일부러 blob으로 지정해준는 이유는 모델을 처리할 때 사이즈가 정해져 있기때문
#위 코드가 네트워크 마다 달라서 다 찾아서 써줘야 한다

net.setInput(blob)
prob=net.forward()
#대부분 이 확률이 복잡한 array형태로 나오는데 이걸 뽑을 수 있어야함

out=prob.flatten()
classID = np.argmax(out)
confidence = out[classID]

text = f'{classNames[classID]}({confidence*100:4.2f}%)'
cv.putText(img, text, (20,20), cv.FONT_HERSHEY_SIMPLEX,
          1,(0,0,255),1,cv.LINE_AA)

cv.imshow('img',img)

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()


# In[ ]:




