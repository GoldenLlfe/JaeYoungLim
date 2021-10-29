#!/usr/bin/env python
# coding: utf-8

# # 어제 실습한 것을 Dropout을 이용해서 복습

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow import keras


# In[ ]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[ ]:


# shape확인
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


plt.figure(figsize=(20, 20))
for i, img in enumerate(x_train[:4]):
    plt.subplot(1, 4, i+1)
    plt.imshow(x_train[i])
plt.show()


# In[ ]:


print(f"y_train original shape : {y_train.shape}")
y_train = keras.utils.to_categorical(y_train)
print(f"y_train one-hot shape :{y_train.shape}")


# In[ ]:


y_test = keras.utils.to_categorical(y_test)


# In[ ]:


conv2D = keras.layers.Conv2D(filters=32, kernel_size =3, strides= 1, kernel_initializer="he_normal", activation='relu', padding='same')


# In[ ]:


conv2D_2 = keras.layers.Conv2D(filters=32, kernel_size =3, strides= 1, kernel_initializer="he_normal", activation='relu', padding='same')


# In[ ]:


model_drop = keras.Sequential([
    keras.layers.Conv2D(32, 3, kernel_initializer="he_normal", activation='relu', padding="same", input_shape=(32, 32, 3)),
    keras.layers.Dropout(0.15),   #과적합 방지를 위해 노드의 15%는 버리고 나머지 85%만 활용
    keras.layers.MaxPooling2D(),  #이미지를 학습을 안하고도 특징을 추출하기 때문에 계산의 부담을 줄이는 효과도 있다
    keras.layers.Conv2D(32, 3, kernel_initializer="he_normal", activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, kernel_initializer="he_normal", activation='relu', padding='same'),
    keras.layers.Dropout(0.15),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, kernel_initializer="he_normal", activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, kernel_initializer="he_normal", activation='relu', padding='same'),
    keras.layers.Dropout(0.15),
    keras.layers.Conv2D(128, 3, kernel_initializer="he_normal", activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),   #덴스층에 넣기 위해서는 1차원=벡터로 만들어야 한다
    keras.layers.Dropout(0.15),
    keras.layers.Dense(512, kernel_initializer="he_normal", activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
])


# In[ ]:


model_drop.summary()


# In[ ]:


model_drop.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics=["accuracy"]
)


# In[ ]:


epoch = 10
batch_size = 128
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


# In[ ]:


result = model_drop.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.2, shuffle=True, callbacks=[es])


# In[ ]:


loss, acc = model_drop.evaluate(x_test, y_test)


# # Batch Nomalization

# In[ ]:


#batch normalization
model_batch = keras.Sequential([
        keras.layers.Conv2D(32, 3, kernel_initializer='he_normal', padding='same', input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, 3, kernel_initializer='he_normal', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(512, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dense(10, activation='softmax')])


# In[ ]:


model_batch.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics= ['accuracy']
)


# In[ ]:


epoch = 1000
batch_size = 128
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


# In[ ]:


model_batch.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.2, shuffle=True, callbacks=[es])


# In[ ]:


loss, acc = model_batch.evaluate(x_test, y_test)   #위의 에포크가 끝난 시점(과적합이 일어나기전에 콜백에 의해서 멈춰짐)의 정확도


# # Augmentation 추가

# In[7]:


augment = keras.Sequential(
    [
     keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(32,32,3)),
     keras.layers.experimental.preprocessing.RandomRotation(0.2),
     keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)


# In[ ]:


model2 = keras.Sequential([
    augment,
    keras.layers.Conv2D(32,3, kernel_initializer='he_normal', activation='relu', padding='same'),
    # Dropout
    keras.layers.Dropout(0.15),
    # conv2D 32
    keras.layers.Conv2D(32, 3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # MaxPooling
    keras.layers.MaxPooling2D(),
    # Conv2D 64
    keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Dropout
    keras.layers.Dropout(0.2),
    # Maxpooling
    keras.layers.MaxPooling2D(),
    # conv2D 64
    keras.layers.Conv2D(64,3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Maxpooling
    keras.layers.MaxPooling2D(),
    # Conv2D 128
    keras.layers.Conv2D(128,3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Dropout
    keras.layers.Dropout(0.3),
    # Conv2D 128
    keras.layers.Conv2D(128,3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Maxpooling
    keras.layers.MaxPooling2D(),
    # Flatten
    keras.layers.Flatten(),
    # Dropout
    keras.layers.Dropout(0.3),
    # Dense
    keras.layers.Dense(512, kernel_initializer='he_normal', activation='relu'),
    # Dropout
    keras.layers.Dropout(0.2),
    # Dense
    keras.layers.Dense(10, kernel_initializer='he_normal',activation='softmax')
])


# In[ ]:


model2.summary()


# #Dropout 을 이용해서 실습

# In[ ]:


model2.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics= ['accuracy']
)


# In[ ]:


epoch = 1000
batch_size = 128
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


# In[ ]:


model2.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.2, shuffle=True, callbacks=[es])


# In[ ]:


loss, acc = model2.evaluate(x_test, y_test)


# In[3]:


#DATA를 만듬
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
valid_datagent =tf.keras.preprocessing.image.ImageDataGenerator()
test_datagent = tf.keras.preprocessing.image.ImageDataGenerator()


# In[18]:


train_dir = r"C:\Users\USER\myreposit\Deep Learning\cat&dog_train"
#만든 데이터를 불러와서 parsing한다
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(128,128),
    batch_size=20,
    interpolation='bilinear',
    color_mode = 'rgb',
    shuffle=True,
    #바이너리 크로스 엔트로피 손실을 사용하기 때문에 이진 레이블이 필용하다
    class_mode='binary'
)


# In[13]:





# In[5]:


print(train_generator.class_indices)
print(train_generator.classes)


# In[19]:


validation_dir = r"C:\Users\USER\myreposit\Deep Learning\cat&dog_validation"
validation_generator = valid_datagent.flow_from_directory(
    directory=validation_dir,
    target_size=(128,128),
    batch_size=20,
    interpolation='bilinear',  #resize시 interpolatrion 기법
    shuffle='True',
    color_mode='rgb',
    #바이너리 크로스 엔트로피 손실을 사용하기 때문에 이진 레이블이 필용하다
    class_mode='binary'    
)


# In[20]:


test_directory = r"C:\Users\USER\myreposit\Deep Learning\cat&dog_test"
test_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(128,128),
    batch_size=20,
    interpolation='bilinear',  #resize시 interpolatrion 기법
    shuffle='True',
    color_mode='rgb',
    #바이너리 크로스 엔트로피 손실을 사용하기 때문에 이진 레이블이 필용하다
    class_mode='binary'    
)


# In[28]:


augment = keras.Sequential(
    [
     keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
     keras.layers.experimental.preprocessing.RandomRotation(0.2),
     keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)


# In[29]:


model3 = keras.Sequential([
    keras.Input(shape = (128,128,3)),
    augment,
    keras.layers.Conv2D(32,3, kernel_initializer='he_normal', activation='relu', padding='same'),
    # Dropout
    keras.layers.Dropout(0.3),
    # conv2D 32
    keras.layers.Conv2D(32, 3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # MaxPooling
    keras.layers.MaxPooling2D(),
    # Conv2D 64
    keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Dropout
    keras.layers.Dropout(0.2),
    # Maxpooling
    keras.layers.MaxPooling2D(),
    # conv2D 64
    keras.layers.Conv2D(64,3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Maxpooling
    keras.layers.MaxPooling2D(),
    # Conv2D 128
    keras.layers.Conv2D(128,3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Dropout
    keras.layers.Dropout(0.3),
    # Conv2D 128
    keras.layers.Conv2D(128,3, kernel_initializer='he_normal', activation='relu',padding='same'),
    # Maxpooling
    keras.layers.MaxPooling2D(),
    # Flatten
    keras.layers.Flatten(),
    # Dropout
    keras.layers.Dropout(0.3),
    # Dense
    keras.layers.Dense(512, kernel_initializer='he_normal', activation='relu'),
    # Dropout
    keras.layers.Dropout(0.2),
    # Dense
    keras.layers.Dense(1,activation='sigmoid')
])
model3.summary()


# In[29]:


model3.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics=['binary_accuracy']
)


# In[30]:


## 모델
input_Layer = tf.keras.layers.Input(shape=(128,128,3))
x=tf.keras.layers.Conv2D(32,(3,3),strides=1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_Layer)
x=tf.keras.layers.MaxPool2D((2,2))(x)
x=tf.keras.layers.Conv2D(64,(3,3),strides=1,activation='relu', padding='same')(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)
x=tf.keras.layers.Conv2D(128,(3,3),strides=1,activation='relu')(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.Activation('relu')(x)
x=tf.keras.layers.Conv2D(64,(3,3),strides=1,activation='relu')(x)
x=tf.keras.layers.Dropout(0.5)(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)
x=tf.keras.layers.Flatten()(x)
x= tf.keras.layers.Dense(512, activation='relu')(x)
Out_Layer= tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_Layer], outputs=[Out_Layer])
model.summary()


# In[44]:


loss_function=tf.keras.losses.binary_crossentropy
optimize=tf.keras.optimizers.RMSprop(learning_rate=0.004)
metric=tf.keras.metrics.binary_accuracy
model.compile(loss=loss_function,
              optimizer=optimize,
              metrics=[metric])


# In[45]:


import os as os


# In[47]:


tf.test.is_gpu_available()


# In[48]:


tf.config.list_physical_devices('GPU')


# In[49]:


train_cats_dir = os.path.join(train_dir,'cat')
train_dogs_dir = os.path.join(train_dir,'dog')
epochs = 100
batch_size = 20

history = model.fit(
    train_generator,
    steps_per_epoch=(len(os.listdir(train_cats_dir))+len(os.listdir(train_dogs_dir)))/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    #callbacks=[es],
    validation_freq=1
)


# In[140]:


loss, acc = model3.evaluate(x_test, y_test)


# In[21]:


#DATA를 만듬
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator( # Augmentation 추가
    rotation_range =40, # 회전 범위
    width_shift_range=0.2, # 가로로 이동 비율
    height_shift_range=0.2, # 세로로 이동 비율
    shear_range=0.2, # 전단의 강도
    zoom_range=0.2, # 확대와 축소 범위 (1-0.2 ~ 1+0.2)
    horizontal_flip =True 
)
valid_datagen =tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()


# In[22]:


valid_dir = r"C:\Users\USER\myreposit\Deep Learning\cat&dog_validation"
test_dir = r"C:\Users\USER\myreposit\Deep Learning\cat&dog_test"
train_dir = r"C:\Users\USER\myreposit\Deep Learning\cat&dog_train"


# In[23]:


train_cats_dir = os.path.join(train_dir,'cat')
train_dogs_dir = os.path.join(train_dir,'dog')
valid_cats_dir = os.path.join(valid_dir,'cats')
valid_dogs_dir = os.path.join(valid_dir,'dogs')
test_cats_dir = os.path.join(test_dir,'cats')
test_dogs_dir = os.path.join(test_dir,'dogs')


# In[25]:


epochs = 100
batch_size = 20
## 만든데이터를 불러와서 파씽함.
validation_generator = valid_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle='True',
        interpolation='bilinear',  ## resize시 interpolatrion 기법
        color_mode='rgb',
        class_mode='binary') #categorical

test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle='True',
        interpolation='bilinear',  ## resize시 interpolatrion 기법
        color_mode='rgb',
        class_mode='binary') #categorical

train_generator = train_datagen.flow_from_directory(
        directory=train_dir,         # 타깃 디렉터리
        target_size=(128, 128),      # 모든 이미지를 128 × 128 크기로 바꿉니다
        batch_size=batch_size,
        interpolation='bilinear',  ## resize시 interpolatrion 기법
        color_mode ='rgb',
        shuffle='True',
        # binary_crossentropy 손실을 사용하기 때문에 이진 레이블이 필요합니다
        class_mode='binary') # binary, categorical , sparse , input

