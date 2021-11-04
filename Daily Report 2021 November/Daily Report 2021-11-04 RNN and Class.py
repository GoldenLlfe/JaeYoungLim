#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Input
from sklearn.model_selection import train_test_split


# In[2]:



seq_length= 2
x_data_dim=4
batch_size= 100
min_max_normalization_flag=True

data_dir = './'
fname = os.path.join(data_dir, 'a_company_stock.csv')
df = pd.read_csv(fname)

dataset=df.copy()
ori_Y= dataset.pop("Close")
ori_X=dataset.copy()



X_train, X_test, Y_train, Y_test = train_test_split(ori_X,ori_Y, test_size=0.2, shuffle=False)
X_train, X_val, Y_train, Y_val= train_test_split(X_train,Y_train  , test_size=0.2, shuffle=False)


# In[3]:


df


# In[4]:


## 데이터의 min , max, mean, std 값 구하기.
dataset_stats = X_train.describe()
dataset_stats = dataset_stats.transpose()

## data normalization
## data normalization
def min_max_norm(x):
  return (x - dataset_stats['min']) / (dataset_stats['max'] - dataset_stats['min'])

def standard_norm(x):
  return (x - dataset_stats['mean']) / dataset_stats['std']

if min_max_normalization_flag==True:
    min_max_norm_train_data = min_max_norm(X_train)
    min_max_norm_val_data = min_max_norm(X_val)
    min_max_norm_test_data = min_max_norm(X_test)

    data_gen_train=tf.keras.preprocessing.sequence.TimeseriesGenerator(min_max_norm_train_data.values.tolist(), Y_train.values.tolist(),
                                                                        length=seq_length, sampling_rate=1,
                                                                        batch_size=batch_size)
    data_gen_val=tf.keras.preprocessing.sequence.TimeseriesGenerator(min_max_norm_val_data.values.tolist(), Y_val.values.tolist(),
                                                                       length=seq_length, sampling_rate=1,
                                                                       batch_size=batch_size)
    data_gen_test=tf.keras.preprocessing.sequence.TimeseriesGenerator(min_max_norm_test_data.values.tolist(), Y_test.values.tolist(),
                                                                       length=seq_length, sampling_rate=1,
                                                                       batch_size=batch_size)
else:
    data_gen_train = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_train.values.tolist(),Y_train.values.tolist(),
                                                                   length=seq_length, sampling_rate=1,
                                                                   batch_size=batch_size)
    data_gen_val = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_val.values.tolist(),Y_val.values.tolist(),
                                                                   length=seq_length, sampling_rate=1,
                                                                   batch_size=batch_size)
    data_gen_test = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_test.values.tolist(),Y_test.values.tolist(),
                                                                        length=seq_length, sampling_rate=1,
                                                                        batch_size=batch_size)


# In[5]:


input_Layer = tf.keras.layers.Input(shape=(seq_length, x_data_dim))
x = tf.keras.layers.LSTM(20,return_sequences=True, activation='tanh')(input_Layer)
x = tf.keras.layers.LSTM(40,return_sequences=True, activation='tanh')(input_Layer)
x = tf.keras.layers.LSTM(60,return_sequences=True, activation='tanh')(input_Layer)
x = tf.keras.layers.Dense(20, activation='relu')(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
Out_Layer= tf.keras.layers.Dense(1, activation=None)(x)
model2 = tf.keras.Model(inputs=[input_Layer], outputs=[Out_Layer])
model2.summary()


loss_function= tf.keras.losses.mean_squared_error
optimize= tf.keras.optimizers.Adam(learning_rate=0.001)
metric= tf.keras.metrics.mean_absolute_error
model2.compile(loss = loss_function,
              optimizer = optimize,
              metrics = [metric])
es = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience= 10)
cp = keras.callbacks.ModelCheckpoint('./predict_stock_2.h5',monitor = 'val_loss',save_best_only = True)

history2 = model2.fit(data_gen_train,
                    validation_data = data_gen_val,
                    steps_per_epoch = len(X_train)/batch_size,
                    epochs = 50,
                      callbacks = [es,cp],
                    validation_freq = 1
)

print(model2.evaluate(data_gen_test))


# In[7]:


input_Layer = tf.keras.layers.Input(shape=(seq_length, x_data_dim))
x = tf.keras.layers.GRU(20, return_sequences=True,activation='tanh')(input_Layer)
x = tf.keras.layers.GRU(40, return_sequences=True,activation='tanh')(input_Layer)
x = tf.keras.layers.GRU(60, return_sequences=True,activation='tanh')(input_Layer)
x = tf.keras.layers.Dense(20, activation='relu')(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
Out_Layer= tf.keras.layers.Dense(1, activation=None)(x)
model3 = tf.keras.Model(inputs=[input_Layer], outputs=[Out_Layer])
model3.summary()


loss_function= tf.keras.losses.mean_squared_error
optimize= tf.keras.optimizers.Adam(learning_rate=0.001)
metric= tf.keras.metrics.mean_absolute_error
model3.compile(loss = loss_function,
              optimizer = optimize,
              metrics = [metric])
es = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience= 10)
cp2 = keras.callbacks.ModelCheckpoint('./predict_stock_3.h5',monitor = 'val_loss',save_best_only = True)

history3 = model3.fit(data_gen_train,
                    validation_data = data_gen_val,
                    steps_per_epoch = len(X_train)/batch_size,
                    epochs = 50,
                      callbacks = [es,cp2],
                    validation_freq = 1
)

print(model3.evaluate(data_gen_test))


# In[15]:


test_data_X, test_data_Y=data_gen_test[0]
prediction_3gru=model3.predict(test_data_X).flatten()
prediction_3lstm=model2.predict(test_data_X).flatten()
Y_test=test_data_Y.flatten()

predict = [prediction_3gru, prediction_3lstm]
his = [history2, history3]
name = [ 'gru3','lstm3']

# visual_y=[]
# visual_pre_y=[]
# for i in range(len(prediction_Y)):
#     label = Y_test[i]
#     prediction = prediction_Y[i]
#     print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
#     visual_y.append(label)
#     visual_pre_y.append(prediction)

# time = range(1, len(visual_y) + 1)
# plt.plot(time, visual_y, 'r', label='true')
# plt.plot(time, visual_pre_y, 'b', label='prediction')
# plt.title('stock prediction')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.legend()
# plt.show()

# loss2 = history2.history['loss']
# val_loss2 = history2.history['val_loss']
# loss3 = history3.history['loss']
# val_loss3 = history3.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss2,loss3, 'bo', label='Training loss')
# plt.plot(epochs, val_loss2,val_loss3, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

plt.figure(figsize = (20,20))
for index,(prediction_Y ,history) in enumerate(zip(predict,his)):
  visual_y=[]
  visual_pre_y=[]
  for i in range(len(prediction_Y)):
    label = Y_test[i]
    prediction = prediction_Y[i]
    visual_y.append(label)
    visual_pre_y.append(prediction)
  plt.subplot(2,1,1)
  time = range(1, len(visual_y) + 1)
  if index ==0:
    plt.plot(time, visual_y, 'r', label='true')
  plt.plot(time, visual_pre_y, label=f'{name[index]}_prediction')
  plt.title('stock prediction')
  plt.xlabel('time')
  plt.ylabel('value')
  plt.legend()
  
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(1, len(loss) + 1)
  plt.subplot(2,1,2)
  plt.plot(epochs, loss, label=f'{name[index]}_Training loss')
  plt.plot(epochs, val_loss, label=f'{name[index]}_Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
plt.show()


# ## Single layer LSTM 실험결과
# ![1.png](attachment:1.png)

# ## 단어로 감성분류하기

# In[9]:


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import time


# ### 데이터셋 구성
# - 각 단어에 대한 정답을 부정(0), 긍정(1)로 정의해서 데이터셋을 구성

# In[10]:


x_train_words = ['good', 'bad', 'amazing', 'so good', 'bull shit',
                 'awesome', 'how dare', 'very much', 'nice', 'god damn it',
                 'very very very happy', 'what the fuck']
y_train = np.array([1, 0, 1, 1, 0,
                    1, 0, 1, 1, 0,
                    1, 0], dtype=np.int32)


# In[17]:


#부정의 예시
index = 1
print('word: {}\nlabel: {}'.format(x_train_words[index],y_train[index]))


# In[18]:


#긍정의 예시
index = 0
print('word: {}\nlabel: {}'.format(x_train_words[index],y_train[index]))


# ## 텍스트데이터 처리를 위한 Tokenizer사용

# In[19]:


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[20]:


tokenizer = Tokenizer(char_level=True)


# In[22]:


tokenizer.fit_on_texts(x_train_words)


# In[23]:


num_chars=len(tokenizer.word_index)+1
print('number of characters: {}'.format(num_chars))


# In[24]:


tokenizer.word_index


# In[25]:


x_train_tokens=tokenizer.texts_to_sequences(x_train_words)


# In[26]:


index=2
print('text: {}'.format(x_train_words[index]))
print('tokens: {}'.format(x_train_tokens[index]))


# In[27]:


x_train_seq_length = np.array([len(tokens) for tokens in x_train_tokens], dtype=np.int32)
num_seq_length = x_train_seq_length


# In[28]:


max_seq_length = np.max(num_seq_length)
print(max_seq_length)


# In[29]:


pad = 'pre'
#pad = 'post'


# In[31]:


x_train_pad = pad_sequences(sequences=x_train_tokens, maxlen=max_seq_length,
                           padding=pad, truncating=pad)
#최대 길이만큼 0을 패팅으로 덮어줌


# In[32]:


index = 7
print('text : {}\n'.format(x_train_words[index]))
print('token : {}\n'.format(x_train_tokens[index]))
print('pad : {}\n'.format(x_train_pad[index]))


# In[35]:


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(),idx.keys()))
print(inverse_map)


# In[39]:


def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]

    text = "".join(words)

    return text


# In[40]:


index = 10
print("original text : \n{}\n".format(x_train_words[index]))
print("tokens: \n{}\n".format(x_train_tokens[index]))
print("tokens to string: \n{}".format(tokens_to_string(x_train_tokens[index])))


# In[41]:


batch_size = 4
max_epochs = 50
num_units = 16
num_classes = 2 
initializer_scale = 0.1
learning_rate = 1e-3


# In[43]:


#tf.data로 data pipline 생성
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_pad,x_train_seq_length,y_train))
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(batch_size=batch_size)
print(train_dataset)


# In[45]:


model = tf.keras.Sequential([
            layers.Embedding(num_chars, num_chars, embeddings_initializer='identity', trainable=False),
            layers.SimpleRNN(units=num_units),
            layers.Dense(units=num_classes, activation='sigmoid')
])


# In[53]:


optimizer=tf.keras.optimizers.Adam(learning_rate)
loss_obj=tf.keras.losses.BinaryCrossentropy(from_logits=False)
mean_loss=tf.keras.metrics.Mean('loss')
loss_history = []


# ### tf.GradientTape를 이용한 학습 진행

# In[54]:


total_steps = int(len(x_train_words)/ batch_size * max_epochs)

for (step, (seq_pad, seq_length, labels)) in enumerate(train_dataset.take(total_steps)):
    start_time = time.time()
    with tf.GradientTape() as tape:
        logits = model(seq_pad)
        loss_value = loss_obj(tf.one_hot(labels, depth=num_classes), logits)

    mean_loss(loss_value)
    loss_history.append((mean_loss.result().numpy()))
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))

    if step % 3 == 0:
        clear_output(wait=True)
        duration = time.time() - start_time
        examples_per_sec = batch_size/float(duration)
        epochs = batch_size * step / float(len(x_train_words))
        print("epochs : {:.2f}, step : {}, loss: {:g}, ({:.2f} examples/sec; {: .3f} sec/batch".format(epochs+1, step, loss_value, examples_per_sec, duration))
print("training finished!")


# In[55]:


loss_history = np.array(loss_history)
plt.plot(loss_history, label='train')


# 모델 평가

# In[56]:


train_dataset_eval = tf.data.Dataset.from_tensor_slices((x_train_pad, x_train_seq_length, y_train))
train_dataset_eval = train_dataset_eval.batch(batch_size=len(x_train_pad))


# In[57]:


loss_object = tf,keras.losses.CategoricalCrossentropy()
acc_object = tf.keras.metrics.CategoricalAccuracy()
val_acc_object = tf.keras.metrics.CategoricalAccuracy()


# In[58]:


val_mean_loss = tf.keras.metrics.Mean("loss")
val_mean_accuracy = tf.keras.metrics.Mean('accuracy')


# In[63]:


for (step, (seq_pad, seq_length, labels)) in enumerate(train_dataset.take(1)):
    predictions = model(seq_pad, training=False)
    val_loss_value = loss_object(tf.one_hot(labels, depth=num_classes), predictions)
    val_acc_value = val_acc_object(tf.one_hot(labels, depth=num_classes), predictions)

    val_mean_loss(val_loss_value)
    val_mean_accuracy(val_acc_value)

    print("valid loss : {: .4g}, valid accuracy : {: .4g}%".format(val_mean_loss.result(),
                                                                   val_mean_accuracy.result() * 100))


# In[64]:


for (step, (seq_pad, seq_length, labels)) in enumerate(train_dataset_eval.take(1)):
    logits = model(seq_pad)
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)


# In[65]:


predictions


# In[66]:


for x , y in zip(seq_pad, predictions):
    if y.numpy() == 1:
        print("{}: positive".format(tokens_to_string(x.numpy())))
    else:
        print("{}: negative".format(tokens_to_string(x.numpy())))


# ## Subclassing API 구경하기

# In[ ]:


class MyNeuralNetwork(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        super(MyNeuralNetwork, self).__init__()

        self.layer1 = layers.Dense(hidden_size, activation = activation.sigmoid)

    def call(self, inputs):
        # forward propagation수행
        a1 = self.layer1(inputs)
        output = self.layer2(a1)

        return output


# ### 문제 1
# tf.kreas.Model을 상속받아 MyNeuralNetwork 클래스를 만들고 다음의 신경망을 설계하시오.
# - 4개의 은닉층과 1개의 출력층으로 구성
# - 은닉층의 뉴런의 수는 100개, 출력층의 뉴런의 수는 2개
# - 은닉층의 활성화함수는 시그모이드, 출령층의 활성화함수는 선형
# 

# In[89]:


import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import layers
get_ipython().system('pip install pydot')
x = tf.convert_to_tensor([[0, 1]], dtype=tf.float32)


# In[90]:


class MyNeuralNetwork(keras.Model):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()

        self.dense1 = layers.Dense(units=100, activation=activations.sigmoid)
        self.dense2 = layers.Dense(units=100, activation=activations.sigmoid)
        self.dense3 = layers.Dense(units=100, activation=activations.sigmoid)
        self.dense4 = layers.Dense(units=100, activation=activations.sigmoid)
        self.dense5 = layers.Dense(units=2, activation=activations.linear)
    
    def call(self, x):
        a1 = self.dense1(x)
        a2 = self.dense2(a1)
        a3 = self.dense3(a2)
        a4 = self.dense4(a3)
        y = self.dense5(a4)

        return y


# In[91]:


# 커스텀 모듈 호출
model = MyNeuralNetwork()
y = model(x)  # model.forward(x)
print(y)


# ### 문제 2
# tf.kreas.Model을 상속받아 MySequentialNeuralNetwork 클래스를 만들고 다음의 신경망을 설계하고, plot_model을 통해 layers를 보이시오.
# - 99개의 은닉층과 1개의 출력층으로 구성
# - 은닉층의 뉴런의 수는 100개, 출력층의 뉴런의 수는 2개
# - 은닉층의 활성화함수는 시그모이드, 출령층의 활성화함수는 선형

# In[92]:


class MySequentialNeuralNetwork(keras.Model):
    def __init__(self):
        super(MySequentialNeuralNetwork, self).__init__()

        self.dense_layers = keras.Sequential()
        for i in range(99):
            self.dense_layers.add(layers.Dense(units=100, activation=activations.sigmoid))
        self.dense_layers.add(layers.Dense(units=2, activation=activations.linear))
    
    def call(self, x):
        y = self.dense_layers(x)

        return y


# In[93]:


model = MySequentialNeuralNetwork()
y = model(x)  # model.call(x)
print(y)


# In[94]:


tf.keras.utils.plot_model(model.dense_layers)


# ## XOR문제 학습하기

# In[ ]:


class XOR(keras.Model):
    """XOR Network"""
    def __init__(self):
        super(XOR, self).__init__()
        # 층을 구성
        self.dense_layers = keras.Sequential()
        self.dense_layers.add(layers.Dense(units=300, activation=activations.relu))
        self.dense_layers.add(layers.Dense(units=300, activation=activations.relu))
        self.dense_layers.add(layers.Dense(units=2, activation=activations.linear))
    
    def call(self, x):
        # forward propagation 수행
        y = self.dense_layers(x)
        return y


# In[ ]:


targets = tf.convert_to_tensor([0, 1, 1, 0], dtype=tf.int32)
targets = tf.one_hot(targets, depth=2)


# In[ ]:


tf.random.set_seed(777)

n_step = 10001  # 총 학습 스텝

# Data 세트 만들기
inputs = tf.convert_to_tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=tf.float32)
targets = tf.convert_to_tensor([1, 0, 0, 1], dtype=tf.int32)
targets = tf.one_hot(targets, depth=2)

# 모델 생성
model = XOR()

# 손실함수
loss_function = losses.BinaryCrossentropy(from_logits=True)

# 옵티마이져
optimizer = optimizers.SGD(learning_rate=0.001)

best_loss = 999
# n_step 동안 학습을 진행한다.
for step in range(n_step):
    # -- 훈련단계 --
    train_loss = 0
    
    
    # 정답을 작성해주세요.
    with tf.GradientTape() as tape:
        # (1) 순방향전파
        
        # (2) 손실값 계산
        
        # (3) 역방향전파(Back Propagation)
        
    
    
    # (4) 옵티마이저로 매개변수 업데이트
    
    
    # 훈련단계 손실값 기록(모든 데이터에 손실값의 평균을 합친다.)
    train_loss += loss.numpy()
    if train_loss < best_loss:
        best_loss = train_loss
        model.save_weights('./XOR_model')
    if step % 1000 == 0:
        print(f"[{step+1}] Loss: {train_loss:.4f}")
        print(model(inputs).numpy())
        print(tf.argmax(model(inputs), axis=1).numpy())


# In[ ]:




