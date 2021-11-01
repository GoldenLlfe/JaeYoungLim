#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os


# In[22]:


#DATA를 만듬
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # Augmentation 추가
   rotation_range =40, # 회전 범위
   width_shift_range=0.2, # 가로로 이동 비율
   height_shift_range=0.2, # 세로로 이동 비율
   shear_range=0.2, # 전단의 강도
    zoom_range=0.2, # 확대와 축소 범위 (1-0.2 ~ 1+0.2)
    horizontal_flip =True,
    validation_split=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()


# In[23]:


train_dir = r"C:\Users\USER\myreposit\Deep Learning\cifar_10_small\train"
test_dir = r"C:\Users\USER\myreposit\Deep Learning\cifar_10_small\test"


# In[24]:


#만든 데이터를 불러와서 parsing한다
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150,150),
    batch_size=20,
    interpolation='bilinear',
    color_mode = 'rgb',
    shuffle='True',
    class_mode='categorical',
    subset='training'
)


# ## 위에서 subset을 만들어준 이유는 4가지 항목에 대해 모델을 따로 실행을 해줘야 하기 때문이다

# In[25]:


#만든 데이터를 불러와서 parsing한다
valid_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150,150),
    batch_size=20,
    interpolation='bilinear',
    color_mode = 'rgb',
    shuffle='True',
    class_mode='categorical',
    subset='validation'
)


# In[26]:


test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(150,150),
    batch_size=20,
    interpolation='bilinear',  #resize시 interpolatrion 기법
    shuffle='True',
    color_mode='rgb',
    class_mode='categorical'
)


# In[27]:


## 모델
#Conv2D가 많을 수록 무조건 학습이 좋아히는 것이 아니다, 학습이 느려질
#수 도 있고, 과적합이 일어날 수 도 있다.
input_Layer = tf.keras.layers.Input(shape=(150,150,3))
x=tf.keras.layers.Conv2D(32,(3,3),strides=1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_Layer)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)
x=tf.keras.layers.Conv2D(64,(3,3),strides=1,activation='relu', padding='same')(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)
x=tf.keras.layers.Conv2D(128,(3,3),strides=1,activation='relu')(x)
x=tf.keras.layers.BatchNormalization()(x)

#평탄화후 덴스층에 추출한 데이터(피쳐)를 입력
x=tf.keras.layers.MaxPool2D((2,2))(x)
x=tf.keras.layers.Flatten()(x)
x= tf.keras.layers.Dense(512, activation='relu')(x)
x=tf.keras.layers.Dropout(0.5)(x)
Out_Layer= tf.keras.layers.Dense(4, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_Layer], outputs=[Out_Layer])
model.summary()


# In[28]:


loss_function=tf.keras.losses.categorical_crossentropy
optimize=tf.keras.optimizers.Adam()
metric=tf.keras.metrics.categorical_accuracy
model.compile(loss=loss_function,
              optimizer=optimize,
              metrics=[metric])


# In[29]:


epochs = 100
batch_size = 20
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

#학습시작
history = model.fit(
    train_generator,
    #에포크 한번에 몇번의 과정을 수행할지 결정
    steps_per_epoch=16000/batch_size,
    steps_per_epoch=100,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[es],
    validation_freq=1
)


# ### 강화 학습 전 돌아간 epoch 회수 :45

# # 시각화

# In[30]:


history.history.keys()


# In[31]:


acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r-', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.show()


# # 예측(test_dir)

# In[32]:


print(model.evaluate(test_generator))
#model.save('cats_and_dogs_binary_classification.hdf5')


# In[ ]:




