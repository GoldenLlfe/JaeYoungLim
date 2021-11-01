#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# ### x_train, y_train, x_test, y_test shape을 뽑아보세요

# In[2]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ### plt.imshow를 이용해 train데이터를 plot해보세요

# In[3]:


plt.figure(figsize=(20, 20))
for i, img in enumerate(x_train[:4]):
    plt.subplot(1, 4, i+1)
    plt.imshow(x_train[i])
plt.show()


# ### 위에서 뽑은 train데이터의 라벨값을 출력하세요.

# In[4]:


y_train = keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)
print(f"y_train shape :{y_train.shape}")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# ### 최대/최소로 정규화를 해주세요.

# In[5]:


print(np.min(x_train))
print(np.max(x_train))
print(np.min(y_train))
print(np.max(y_train))
x_train = x_train/np.max(x_train)
print(x_train.shape)
print(np.max(x_train))


# ### CNN모델을 설계해주세요.

# In[6]:


model = keras.Sequential([
    keras.layers.Input(shape=(32,32,3)),
    # Conv2D 16, 3
    keras.layers.Conv2D(16,(3,3), kernel_initializer='he_normal', activation='relu', padding='same'),
    # MaxPool2D
    keras.layers.MaxPool2D(),
    # Conv2D 32, 3
    keras.layers.Conv2D(32, (3,3),kernel_initializer='he_normal', activation='relu', padding='same'),
    # MaxPool2D
    keras.layers.MaxPool2D(),
    # Flatten
    keras.layers.Flatten(),
    # Dense 32
    keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
    # Dense
    keras.layers.Dense(10, activation='softmax')
])


# ### model.summary()를 이용해 모델요약정보를 뽑아주세요.

# In[7]:


model.summary()


# ### model.compile해주세요

# In[8]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ### model.fit으로 학습시켜주세요.

# In[9]:


es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=200, verbose=1, callbacks=es)


# ### test로 model.evaluate해주세요.

# In[10]:


model.evaluate(x_test, y_test)


# ### model.predict결과를 뽑아주세요.

# In[12]:


print(model.predict(x_test))
print(np.argmax(model.predict(x_test), axis = 1))


# In[14]:


model2=keras.Sequential()
# Conv2D 16, 3(커널사이즈)
model2.add(keras.layers.Conv2D(16,(3,3),padding='same', activation='relu',input_shape=(32,32,3)))
model2.add(keras.layers.Conv2D(16,(3,3),kernel_initializer='he_normal', activation='relu', padding='same'))
# MaxPool2D
model2.add(keras.layers.MaxPool2D(2,2))
# Conv2D 32, 3
model2.add(keras.layers.Conv2D(32,(3,3),kernel_initializer='he_normal', activation='relu', padding='same'))
model2.add(keras.layers.Conv2D(32,(3,3),kernel_initializer='he_normal', activation='relu', padding='same'))
# MaxPool2D
model2.add(keras.layers.MaxPool2D(2,2))

model2.add(keras.layers.Conv2D(64,(3,3),,kernel_initializer='he_normal', activation='relu', padding='same'))
model2.add(keras.layers.Conv2D(64,(3,3),,kernel_initializer='he_normal', activation='relu', padding='same'))
model2.add(keras.layers.MaxPool2D(2,2))

model2.add(keras.layers.Conv2D(128,(3,3),,kernel_initializer='he_normal', activation='relu', padding='same'))
model2.add(keras.layers.Conv2D(128,(3,3),,kernel_initializer='he_normal', activation='relu', padding='same'))
model2.add(keras.layers.MaxPool2D(2,2))

# Flatten
model2.add(keras.layers.Flatten())
# Dense 32
model2.add(keras.layers.Dense(32, activation='relu'))
# Dense
model2.add(keras.layers.Dense(10, activation='softmax'))


# In[15]:


model2.summary()


# In[16]:


model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[18]:


es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=200, verbose=1)


# In[24]:


def build_vgg_block(input_layer,
                    num_cnn=3,
                    channel=64,
                    block_num=1,
                    ):
    # 입력레이어
    x = input_layer

    # CNN 레이어
    for cnn_num in range(num_cnn):
        x = keras.layers.Conv2D(
            filters = channel,
            kernel_size = (3,3),
            activation='relu',
            kernel_initializer='he_normal',
            padding='same',
            name=f'block{block_num}_conv{cnn_num}'
        )(x)
    
    # Max Pooling 레이어
    x = keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = 2,
        name=f'block{block_num}_pooling'
    )(x)
    
    return x


# In[25]:


vgg_input_layer = keras.layers.Input(shape=(32, 32, 3)) #입력레이어 생성
vgg_block_output = build_vgg_block(vgg_input_layer) # VGG 블록 생성


# In[26]:


model3 = keras.Model(inputs=vgg_input_layer, outputs=vgg_block_output)
model3.summary()


# In[28]:


def build_vgg(input_shape=(32,32,3),
             num_cnn_list=[2,2,3,3,3,],
             channel_list=[64,128,256,512,512],
             num_classes=10):
    assert len(num_cnn_list) == len(channel_list)
    
    input_layer = keras.layers.Input(shape=input_shape)
    output = input_layer
    
    for i,(num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
        output = build_vgg_block(
            output,
            num_cnn=num_cnn,
            channel = channel,
            block_num= i
        )
        output=keras.layers.Flatten(name='flatten')(output)
        output=keras.layers.Dense(4096,activation='relu',name='fc1')(output)
        output=keras.layers.Dense(4096,activation='relu',name='fc2')(output)
        output = keras.layers.Dense(num_classes, activation='softmax', name= 'predictions')(output)
        
        model = keras.Model(
        inputs=input_layer,
        outputs= output
        )
        return model


# In[30]:


vgg_16 = build_vgg


# ![vgg_structure.max-800x600.png](attachment:vgg_structure.max-800x600.png)

# In[ ]:


# 원하는 블록의 설계에 따라 매개변수로 리스트를 전달해 줍니다.
vgg_11 = build_vgg(
    num_cnn_list=[ # To do ],
    channel_list=[ # To do ]
)

vgg_11.summary()


# In[ ]:


# 원하는 블록의 설계에 따라 매개변수로 리스트를 전달해 줍니다.
vgg_13 = build_vgg(
    num_cnn_list=[ # To do ],
    channel_list=[ # To do ]
)

vgg_13.summary()


# In[ ]:


# 원하는 블록의 설계에 따라 매개변수로 리스트를 전달해 줍니다.
vgg_19 = build_vgg(
    num_cnn_list=[ # To do ],
    channel_list=[ # To do ]
)

vgg_19.summary()


# ### VGG16 실습

# In[ ]:


vgg_input_layer = keras.layers.Input(shape=(32, 32, 3)) #입력레이어 생성
vgg_block_output = build_vgg_block(vgg_input_layer) # VGG 블록 생성


# In[ ]:


model4 = keras.Model(inputs=vgg_input_layer, outputs=vgg_block_output)
model4.summary()


# In[33]:


vgg_16 = build_vgg


# In[34]:


BATCH_SIZE = 256
EPOCH = 20


# In[35]:


vgg_16.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.),
    metrics=['accuracy'],
)


# In[39]:


vgg_16.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=200, verbose=1)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history_11.history['loss'], 'g')
plt.plot(history_13.history['loss'], 'k')
plt.plot(history_16.history['loss'], 'r')
plt.plot(history_19.history['loss'], 'b')
plt.title('Model training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['vgg_11','vgg_13','vgg_16', 'vgg_19'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




