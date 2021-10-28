# -*- coding: utf-8 -*-
"""2021-10-28.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1npMHhjSVb-13LaVmxvCJv5TguRciTjWZ
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from google.colab import files
file_uploaded = files.upload()

# MNIST 데이터 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## TEST할 이미지 선택
test_image = x_test[0]

## NN 이미로 이차원으로 데이터를 넣어주어야 해서 1x784 형태로 reshape
test_image_reshape = test_image.reshape(1, 784).astype('float64')

## 모델 불러오기
model = tf.keras.models.load_model('/content/my_NN_Test.h5') # 모델을 새로 불러옴
# 불러온 모델로 값 예측하기.
y_prediction =model.predict(test_image_reshape)

## 10개의 class가 각 확률 값으로 나오기 때문에 가장 높은값을 출력하는 인덱스를 추출. 그럼 이것이 결국 class임.
### np.argmax는 들어온 행렬에서 가장 높은값이 있는 index를 반환해주는 함수.
index = np.argmax(y_prediction)
value = y_prediction[:, index]
plt.imshow(test_image, cmap='Greys')
plt.xlabel(str(index)+"   "+str(value))
plt.show()

"""# 이미지 로테이션"""

# 이미지 회전 변환 매트릭스 구하기
M = cv2.getRotationMatrix2D((28/2, 28/2), 0, 1) # Matrix생성

# 이미지 이동 변환 매트릭스 구하기
M[0, 2] = M[0, 2] +3
M[1, 2] = M[1, 2] +3

# 이미지 변환 매트릭스 적용
test_image = cv2.warpAffine(x_train[5], M, (28, 28)) #image에 matrix곱

plt.imshow(test_image, cmap="Greys")

test_image_reshape = test_image.reshape(1,784).astype('float64')

model = tf.keras.models.load_model('/content/my_NN_Test.h5')

y_pred = model.predict(test_image_reshape)

index = np.argmax(y_pred)
value = y_pred[:, index]
plt.imshow(test_image, cmap='Greys')
plt.xlabel(str(index)+"   "+str(value))
plt.show()

print(value)

plt.imshow(x_train[5], cmap='Greys')

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## 데이터를 (배치사이즈 x 28 x 28 x 1)로 이미지를 변환해줌. -> 그레이스케일이므로 채널은 1
x_train =x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test =x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

## 정답을 바이너리화 함.
y_train =tf.keras.utils.to_categorical(y_train)
y_test =tf.keras.utils.to_categorical(y_test)

# CNN 모델 설계.
## 모델
input_Layer =tf.keras.layers.Input(shape=(28,28,1))
x=tf.keras.layers.Conv2D(32,(3,3),strides=1, activation='relu',padding='same')(input_Layer)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x=tf.keras.layers.Conv2D(64,(3,3),strides=1, activation='relu')(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Dense(512, activation='relu')(x)
out_Layer=tf.keras.layers.Dense(10,activation='softmax')(x)

model =tf.keras.Model(inputs=[input_Layer],outputs=[out_Layer])
model.summary()

loss_function=tf.keras.losses.categorical_crossentropy
optimize=tf.keras.optimizers.RMSprop(lr=0.0001)
metric=tf.keras.metrics.categorical_accuracy
model.compile(loss=loss_function,
              optimizer=optimize,
              metrics=[metric])

# 모델 최적화 설정
MODEL_DIR = './sample_data/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./CNN_MNIST_model/{epoch:02d}-{val_loss:.4f}.hdf5"
callback_list=[tf.keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                  monitor='val_loss,',
                                                  verbose=1,
                                                  save_best_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]

# 모델의 실행
history = model.fit(x_train, y_train,validation_split=0.2,epochs=5,batch_size=200,
                    verbose=1, callbacks=callback_list)

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

(x_train, y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data(
)

x_train =x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test =x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)

# CNN 모델 설계.
## 모델
input_Layer =tf.keras.layers.Input(shape=(28,28,1))
x=tf.keras.layers.Conv2D(32,(3,3),strides=1, activation='relu',padding='same')(input_Layer)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x=tf.keras.layers.Conv2D(64,(3,3),strides=1, activation='relu')(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Dense(512, activation='relu')(x)
out_Layer=tf.keras.layers.Dense(10,activation='softmax')(x)

model =tf.keras.Model(inputs=[input_Layer],outputs=[out_Layer])
model.summary()

loss_function=tf.keras.losses.categorical_crossentropy
optimize=tf.keras.optimizers.RMSprop(lr=0.0001)
metric=tf.keras.metrics.categorical_accuracy
model.compile(loss=loss_function,
              optimizer=optimize,
              metrics=[metric])

# 모델 최적화 설정
MODEL_DIR = './sample_data/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./CNN_MNIST_model/{epoch:02d}-{val_loss:.4f}.hdf5"
callback_list=[tf.keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                  monitor='val_loss,',
                                                  verbose=1,
                                                  save_best_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]

# 모델의 실행
history = model.fit(x_train, y_train,validation_split=0.2,epochs=5,batch_size=200,
                    verbose=1, callbacks=callback_list)

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

"""# cifal10 으로 CNN돌리기"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 시각화
plt.figure(figsize=(10, 10))
for i, img in enumerate(x_train[:8]):
    plt.subplot(2,4,i+1)
    plt.imshow(x_train[i])
plt.show()

# y라벨 one hot encoding수행

# 정규화

# 모델1 설계 (kernel_initializer, callback, Dropout (0.5))
model = keras.Squential(
        conv,
        conv
)
# 32채널 커널사이즈 3 Conv2D, relu
# 32

# 64채널
# 64채널

# 128채널
# 128채널

# FCL
# Flatten()
# Dense() # 512채널
# 마지막 분류

#model1.summary()
#model1.compile()
#model1.fit() #EarlyStopping적용 patience=7

# 모델2 설계 (kernel_initializer, callback, BatchNormalization)



