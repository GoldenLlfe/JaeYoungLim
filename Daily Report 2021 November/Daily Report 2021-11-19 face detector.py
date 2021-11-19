#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import cv2 as cv


# In[6]:


model = 'opencv_face_detector/opencv_face_detector_uint8.pb'
config = 'opencv_face_detector/opencv_face_detector.pbtxt'

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print('Video open failed')
    cap.release()
    sys.exit()
    
net = cv.dnn.readNet(model, config)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    blob = cv.dnn.blobFromImage(frame, 1, (300,300),
                                (104,117,123),swapRB=False)
                            #swapRB는 rgb 랑 bgr 차이때문
    net.setInput(blob)   #
    out = net.forward()  #net을 학습해서 결과를 내라는 뜻
    # 위의 out의 shape을 보면 (1,1,200,7)을 주는데
    # 1,1은 행렬의 형태고 200, 7이 중요한데
    # 200 x 7 의 행렬을 준다 그 중에
    # 1과 2행을 필요없는 정보고
    #3 행의 confidence(신뢰도, 최소 50%이상)
    #그리고 4행부터 7행까지(얼굴의 정보)
    #쓸모 있는 정보이다.
    
    detect = out[0,0,:,:]
    (h,w) = frame.shape[:2]
    
    for i in range(detect.shape[0]): #print(detect.shape을 하면 200이 나온다)
        confidence = detect[i,2]
        if confidence>0.5:
            x1=int(detect[i,3]*w)
            y1=int(detect[i,4]*h)
            x2=int(detect[i,5]*w)
            y2=int(detect[i,6]*h)    #이런 것들은 network마다 스케일링이 된건지, 아니면 다른걸 하던지 다 다르다
            
            cv.rectangle(frame,(x1,y1),(x2,y2),
                        (0,0,255),2)
            label = f'Face : {confidence:4.2f}'
            cv.putText(frame, label, (x1,y1-1),cv.FONT_HERSHEY_COMPLEX,
                      0.8, (0,0,255),1,cv.LINE_AA)
            
    cv.imshow('frame',frame)
            

    if cv.waitKey() == 27:
        break
cap.release()
cv.destroyAllWindows()


# In[ ]:




