#!/usr/bin/env python
# coding: utf-8

# In[2]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')


# ## keras로 RNN구현하기

# model.add(SimpleRNN(hidden_size))  #가장 간단한 형태

# 추가 인자를 사용할 때
# model.add(SimpleRNN(hidden_size, input_shape = (timesteps, input_dim)
# 
# 다른 표기
# model.add(SimpleRNN(hidden_size, input_length=M, input_dim=N))

# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN


# In[10]:


model = Sequential()
model.add(SimpleRNN(3, input_shape = (2, 10)))
#model.add(SimpleRNN(3, input_length=2, input_dim=10)) 와 위줄의 코드는 동일하다

model.summary()  #batch_size를 안줘서 None으로 뜬다


# In[11]:


model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10)))
model.summary()


# In[12]:


model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10), return_sequences=True))
model.summary()   
#return_sequences를 참으로 하면 time steps의 값이 나온다
#(이번 경우는 과거 2일치를 준다는 것 (batch_unput_shape에 넣은 2를 반환한다) 저장된 값을 준다는 것이다)


# ## 파이썬으로 RNN구현하기

# $$ h_t = tanh(W_x X_t + W_h h_{t-1} +b) $$

# In[13]:


#hidden_state = 0 # 초기 은닉 상태를 0으로 초기화
#for input_t in input_length: # 각 시점마다 입력을 받는다.
#    output_t = tanh(input_t, hidden_state_t) # 각 시점에 대해서 입력과 은닉 상태를 가지고 연산
#    hidden_state_t = output_t # 계산 결과는 현재 시점의 은닉 상태과 된다.


# In[14]:


import numpy as np

timesteps=10     #10일치 값
input_dim = 4    #입력으로 들어가는 것이 4차원이라는 뜻
hidden_size = 8  #8차원이라는 뜻, Dense층을 8로 하는 것과 비슷한 뜻

#입력에 해당되는 2D 텐서
inputs = np.random.random((timesteps, input_dim))

#초기 은닉 상태는 0벡터로 초기화
hidden_state_t = np.zeros((hidden_size,))


# In[15]:


print(hidden_state_t)  #8의 크기를 가지는 hidden_state, 현재는 초기 hidden state로 모든 차원이 0의 값을 가짐


# In[16]:


Wx = np.random.random((hidden_size, input_dim)) #(8,4) 2D텐서 생성 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size)) # (8,8)크기의 2D텐서 생성. hidden state에 대한 가중치
b = np.random.random((hidden_size,)) #(8,)크기의 1D텐서 생성. 편향(bias)


# In[17]:


print(np.shape(Wx))  #hidden state x 입력의 차원, 입력이 될 x(입력값)이랑 곱해져서 h_t로 들어갈 것임
print(np.shape(Wh))  #hidden state x hidden state size  현재 상태에 더할 가중치
print(np.shape(b))   #hudden state size    가중치와 같이 더해질 bias


# $$ h_t = tanh(W_x X_t + W_h h_{t-1} +b) $$

# In[21]:


total_hidden_states = []

#메모리 셀 동작
for input_t in  inputs:   #각 시점에 따라서 입력값이 입력이 됨
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t)+b)
    # Wx + wt * Ht-1 + b
    total_hidden_states.append(list(output_t))  #각 시점의 은닉 상태의 값을 계속해서 축적
    print(np.shape(total_hidden_states))  #각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
    hidden_state_t = output_t
print('\n',hidden_state_t.shape)

total_hidden_states = np.stack(total_hidden_states, axis=0)
#(timestep, output_dim)
print('\n', total_hidden_states)


# ## 더 깊은 RNN

# In[23]:


model = Sequential()
model.add(SimpleRNN(hidden_size, input_length =10, input_dim=5, return_sequences=True))
#return_sequence를 참으로 줬으니 각 시점(timesteps)마다의 아웃풋값이 나올 것이다, False를 하면 이전 시점것들은 안보겠다는 뜻으로
#중간의 10으로(10일치 값) 준 timesteps가 0(안보이게 표시되는것)이 될 것이다
model.add(SimpleRNN(hidden_size, return_sequences=True))


# In[24]:


model.summary()


# ## BiLSTM

# In[25]:


from tensorflow.keras.layers import Bidirectional


# In[26]:


timesteps = 10
input_dim = 5


# In[29]:


model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim))))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))


# ```
# - Embedding을 사용하며, 단어 집합(Vocabulary)의 크기가 5,000이고 임베딩 벡터의 차원은 100입니다.
# - 은닉층에서는 Simple RNN을 사용하며, 은닉 상태의 크기는 128입니다.
# - 훈련에 사용하는 모든 샘플의 길이는 30으로 가정합니다.
# - 이진 분류를 수행하는 모델로, 출력층의 뉴런은 1개로 시그모이드 함수를 사용합니다.
# - 은닉층은 1개입니다.
# ```

# ![Embedding%205000%20RNN%20%EC%98%88%EC%8B%9C%20%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EA%B0%92%28%EA%B0%9C%EC%88%98%29-2.png](attachment:Embedding%205000%20RNN%20%EC%98%88%EC%8B%9C%20%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EA%B0%92%28%EA%B0%9C%EC%88%98%29-2.png)

# ## 임의의 입력으로 SimpleRNN 생성

# In[31]:


from tensorflow.keras.layers import LSTM


# In[37]:


train_x = [[[0.1, 4.2, 1.5, 1.1, 2.8],
           [1.0, 3.1, 2.5, 0.7, 1.1],
           [0.3, 2.1, 1.5, 2.1, 0.1],
           [2.2, 1.4, 0.5, 0.9, 1.1]]]
train_x = np.array(train_x, dtype=np.float32)
print(train_x.shape)  #단어 벡터의 차원은 5, 문장의 길이 4  (batch_size, timesteps, input_dim)


# In[38]:


rnn = SimpleRNN(3)
# rnn=SimpleRNN(3, return_sequences=False, return_state=False)와 동일하다
hidden_state = rnn(train_x)

print('hidden state : {}, shape : {} '.format(hidden_state, hidden_state.shape))


# In[39]:


rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_x)

print('hidden state : {}, shape : {}'.format(hidden_states, hidden_states.shape))


# In[41]:


rnn = SimpleRNN(3, return_sequences=True, return_state=True)
hidden_states, last_states = rnn(train_x)

print('hidden states : {}, shape : {}'.format(hidden_states, hidden_states.shape))
print('last shidden state : {}, shape : {}'.format(last_states, last_states.shape))


# In[42]:


rnn = SimpleRNN(3, return_sequences=False, return_state=True)
hidden_state, last_state = rnn(train_x)

print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
print('last shidden state : {}, shape : {}'.format(last_states, last_states.shape))


# ## LSTM 이해하기

# In[43]:


lstm = LSTM(3, return_sequences=False, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_x)

print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape : {}'.format(last_state, last_state.shape))
print('last cell state : {}, shape : {}'.format(last_cell_state, last_cell_state.shape))  
#gradient vanishing 문제 때문에 마지막 셀 상태까지 보여준다


# In[45]:


lstm = LSTM(3, return_sequences=True, return_state=True)
hidden_states, last_states, last_cell_states = lstm(train_x)

print('hidden states : {}, shape : {}'.format(hidden_states, hidden_states.shape))
print('last hidden states : {}, shape : {}'.format(last_states, last_states.shape))
print('last cell states : {}, shape : {}'.format(last_cell_states, last_cell_states.shape))


# ## LSTM 예제

# In[47]:


x = np.array([[1,2, 3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              [8,9,10],
              [9,10,11],
              [10, 11, 12],
              [20, 30, 40],
              [30, 40, 50],
              [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x.shape, y.shape


# In[48]:


x = x.reshape((x.shape[0], x.shape[1],1))
print(x.shape)


# In[50]:


model = keras.Sequential()
model.add(keras.layers.LSTM(20, activation='relu', input_shape=(3,1)))
model.add(keras.layers.Dense(5,activation='relu'))
model.add(keras.layers.Dense(1))


# In[51]:


model.compile(optimizer='adam',loss='mse')


# In[52]:


es = keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[es])


# In[53]:


x_test = np.array([25,35,45])   #predict용 데이터
x_test = x_test.reshape((1,3,1))
pred = model.predict(x_test)
pred


# ## 네이버 주가 예측

# In[54]:


import time
import requests
import pandas as pd
import numpy as np 
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras 


# In[55]:


stock_price = pd.DataFrame()  #주가 데이터를 저장할 dataframe


# In[56]:


stock_number = '035420'
pages = 50


# In[57]:


for page in range(1, pages+1):
    url = f'https://finance.naver.com/item/sise_day.nhn?code={stock_number}&page={page}'
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36'}
    response = requests.get(url, headers=headers)
    time.sleep(0.5)
    html = BeautifulSoup(response.text, "lxml")

    table = html.select("table")

    juga = pd.read_html(str(table))
    juga = juga[0].dropna()
    stock_price = pd.concat([stock_price,juga], axis=0)


# In[58]:


stock_price = stock_price.reset_index(drop=True)


# In[59]:


stock_price.tail()


# In[60]:


stock_price['날짜'] = pd.to_datetime(stock_price['날짜'])


# In[61]:


stock_price['날짜'].head()


# In[62]:


stock_price.tail()


# In[63]:


plt.figure(figsize=(16,9))
sns.lineplot(y=stock_price['종가'], x=stock_price['날짜'])
plt.show()


# In[78]:


#min max

scaler = MinMaxScaler()
scale_cols=['시가', '고가', '저가', '거래량']
scaled = scaler.fit_transform(stock_price[scale_cols])

scaled_stock = pd.DataFrame(scaled)


# In[80]:


scaled_stock.columns = scale_cols


# In[82]:


scaled_stock


# In[83]:


end_price= np.log1p(stock_price['종가'])


# In[87]:


scaled_stock['종가'] = end_price


# In[88]:


scaled_stock


# In[94]:


train = scaled_stock[:-30]   #30만 냅두고 나머지를 트레인 데이터로 활용 -30 -> 뒤에서 30번까지
test = scaled_stock[-30:]


# In[95]:


train.shape, test.shape


# In[103]:


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data)- window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


# In[104]:


train_feature = train[['시가', '고가', '저가', '거래량']]
train_label = train['종가']

test_feature = test[['시가', '고가', '저가', '거래량']]
test_label = test[['종가']]


# In[105]:


train_feature, train_label = make_dataset(train_feature, train_label, 10)

# train, validation set생성
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 10)


# In[106]:


train_feature.shape[1], train_feature.shape[2]


# In[107]:


train_feature.shape


# In[108]:


model = keras.Sequential([
                          keras.layers.LSTM(16, input_shape=(10, 4), activation='relu'),
                          keras.layers.Dense(1)
])


# In[109]:


model.compile(
    optimizer = 'adam',
    loss= 'mse',
)


# In[110]:


epoch = 500
batch_size = 64

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
cp = keras.callbacks.ModelCheckpoint('predict_juga_2.h5', monitor='val_loss', save_best_only=True)


# In[113]:


history = model.fit(x_train, y_train,
                    epochs= epoch,
                    batch_size = batch_size,
                    validation_data = (x_valid, y_valid),
                    callbacks=[es, cp])


# In[115]:


#최고의 모델을 가져와서 test
loaded_models = keras.models.load_model('predict_juga_2.h5')


# In[116]:


pred = loaded_models.predict(test_feature)


# In[117]:


pred = np.expm1(pred)


# In[119]:


plt.figure(figsize=(12, 9))
plt.plot(np.expm1(test_label), label='actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()


# In[ ]:




