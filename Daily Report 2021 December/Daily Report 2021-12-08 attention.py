#!/usr/bin/env python
# coding: utf-8

# ## Bahdanau Attention

# - Bahdanau Attention
# $$ Score_{alignment} = W * tanh(W_{decoder} * H_{decoder} + W_{encoder} * H_{encoder}) $$

# In[1]:


import tensorflow as tf


# In[2]:


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_decoder = tf.keras.layers.Dense(units)
        self.W_encoder = tf.keras.layers.Dense(units)
        self.W_combine = tf.keras.layers.Dense(1)
    
    def call(self, H_encoder, H_decoder):
        print("[H_encoder shape :", H_encoder.shape)

        H_encoder = self.W_encoder(H_encoder)
        print("[W_encoder X H_encoder shape :", H_encoder.shape)

        print("\n[H_decoder shape:", H_decoder.shape)
        H_decoder = tf.expand_dims(H_decoder, 1)
        H_decoder = self.W_decoder(H_decoder)

        print("[W_decoder X H_decoder] shape :", H_decoder.shape)

        score = self.W_combine(tf.nn.tanh(H_decoder+H_encoder))
        print("[Score Alignment] shape :", score.shape)

        attention_weights = tf.nn.softmax(score, axis= 1)
        print("\n 최종 weight : \n", attention_weights.numpy())

        context_vector = attention_weights * H_decoder
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[3]:


W_size = 100
print("Hidden State를 {0}차원으로 Mapping\n".format(W_size))

attention = BahdanauAttention(W_size)

enc_state = tf.random.uniform((1,10,512))
dec_state = tf.random.uniform((1,512))

_ = attention(enc_state, dec_state)


# In[4]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# ## Loung Attention

# $$ Score(H_{target},H_{source}) = H_{target}^T * W_{combine} * H_{source})$$

# In[5]:


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W_combine = tf.keras.layers.Dense(units)

    def call(self, H_encoder, H_decoder):
        print("[H_encoder] shape: ", H_encoder.shape)

        WH = self.W_combine(H_encoder)
        print("[W_encoder X H_encoder] shape :", WH.shape)

        H_decoder = tf.expand_dims(H_decoder, 1)
        alignment = tf.matmul(WH, tf.transpose(H_decoder, [0, 2, 1]))
        print("[Score_alignmnet] Shape :", alignment.shape)

        attention_weights = tf.nn.softmax(alignment, axis=1)
        print("\n 최종 weight : \n", attention_weights.numpy())

        attention_weights = tf.squeeze(attention_weights, axis=-1)
        context_vector = tf.matmul(attention_weights, H_encoder)

        return context_vector, attention_weights


# In[6]:


emb_dim = 512

attention = LuongAttention(emb_dim)

enc_state = tf.random.uniform((1,10,512 ))
dec_state = tf.random.uniform((1, 512))

_ = attention(enc_state,dec_state)


# ## Softmax

# In[7]:


import numpy as np
import matplotlib.pyplot as plt


# In[8]:


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


# In[9]:


predicted_logit = np.array([-2,1.5,-1,0.5,2])
predicted_prob = softmax(predicted_logit)


# In[10]:


plt.figure(figsize=[18,3])
plt.subplot(1,2,1) # 왼쪽 그림
plt.title('predicted logit')
plt.bar(np.arange(len(predicted_logit)), predicted_logit)
plt.grid()

plt.subplot(1,2,2) #오른쪽 그림
plt.title('predicted probability')
plt.bar(np.arange(len(predicted_prob)),predicted_prob)
plt.grid()
plt.show()


# ## Cross-entropy

# $$ CE(t, y) = - Σ^{k}{k=1}$$

# - t: one-hot vector
# - y probability
# - k : number of categories

# In[11]:


def cross_entropy(t, y):
  eps = 1e-8
  ce = -np.sum(t*np.log(y + eps))
  return ce


# In[12]:


predicted_logit = np.array([-2,1.5,-1,0.5,2])
predicted_prob = softmax(predicted_logit)
target_prob = np.array([0,0,0,0,1])


# In[13]:


plt.figure(figsize=[18,3])
plt.subplot(1,2,1) # 왼쪽 그림
plt.title('predicted probability')
plt.bar(np.arange(len(predicted_prob)), predicted_prob)
plt.grid()
plt.ylim([0,1])

plt.subplot(1,2,2) #오른쪽 그림
plt.title('target probability')
plt.bar(np.arange(len(target_prob)),target_prob)
plt.grid()
plt.ylim([0,1])
plt.show()


# In[14]:


print('Cross-entropy : ', cross_entropy(target_prob, predicted_prob))


# In[15]:


predicted_logit = np.array([-2,1.5,-1,0.5,8])
predicted_prob = softmax(predicted_logit)
target_prob = np.array([0,0,0,0,1])


# In[16]:


plt.figure(figsize=[18,3])
plt.subplot(1,2,1) # 왼쪽 그림
plt.title('predicted probability')
plt.bar(np.arange(len(predicted_prob)), predicted_prob)
plt.grid()
plt.ylim([0,1])

plt.subplot(1,2,2) #오른쪽 그림
plt.title('target probability')
plt.bar(np.arange(len(target_prob)),target_prob)
plt.grid()
plt.ylim([0,1])
plt.show()


# In[17]:


print('Cross-entropy : ', cross_entropy(target_prob, predicted_prob))


# ## 양방향 LSTM + 어텐션 메커니즘 (IMDB 리뷰데이터)

# In[18]:


from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[19]:


vocab_size = 10000
(x_train,y_train), (x_test, y_test) = imdb.load_data(num_words = vocab_size)


# In[20]:


print('리뷰의 최대길이 : {}'.format(max(len(l) for l in x_train)))
print('리뷰의 평균 길이 : {}'.format(sum(map(len, x_train))/len(x_train)))


# In[21]:


max_len = 500
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)


# - Bahdanau Attention
# $$ Score_{alignment} = W * tanh(W_{decoder} * H_{decoder} + W_{encoder} * H_{encoder}) $$
# - Bahdanau Attention
# $$ Score_{alignment} = V * tanh(W_{1} * key + W_{2} * query) $$

# In[22]:


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query):
        # query size = (batch size, hidden size)
        # hidden_with_time_axis = (batch size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score = (batch size, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))) ###

        # attention_weights = (batch size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis =1)

        # context vector shape after sum = (batch size, hidden size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[23]:


from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os


# In[24]:


sequence_input = Input(shape = (max_len, ), dtype = 'int32')
embedded_sequences = Embedding(vocab_size, 128, input_length=max_len, mask_zero=True)(sequence_input)
lstm = Bidirectional(LSTM(64, dropout = 0.5, return_sequences = True))(embedded_sequences) #dropout은 오버피팅/과적합 방지를 위해 사용한다 0.5는 절반을 버린다는 것
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)


# In[25]:


print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)


# In[26]:


state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])


# In[27]:


attention = BahdanauAttention(64)
context_vector, attention_weights = attention(lstm, state_h)


# In[28]:


dense1 = Dense(20, activation='relu')(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs= sequence_input, outputs=output)


# In[29]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=1)

# ## 시간이 너무 오래 걸리는 관계로 스킵
# ### 다른 사람의 결과를 확인한 결과 약 86퍼센트의 테스트 정확도를 보여준다

# print("\n 테스트 정확도 : %.4f" % (model.evaluate(x_test, y_test)[1])) # 소수점 4번 째 자리까지 표시

# ## seq2seq with attention 스페인-영어 번역기

# ### 데이터 준비하기

# In[30]:


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import time
import re
import os
import io


# In[31]:


path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)


# In[32]:


path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"


# In[33]:


with open(path_to_file, 'rt', encoding='UTF8') as f:
    raw = f.read().splitlines()

print("Data Size: ", len(raw))
print("Example :")

for sen in raw[0:100][::20]: print(">>", sen)


# ### 데이터 전처리 : 정제하기

# In[34]:


def preprocess_sentence(sentence, s_token=False, e_token=False):
    # 소문자 변경
    sentence = sentence.lower().strip()

    # 1. 문장 부호를 \1
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    # 2. [ ] --> 공백
    sentence = re.sub(r'[" "]+', " ", sentence)
    # 3. 모든 알파벳, 문장기호를 제외한 것들을 공백으로 바꿔주세요.
    sentence = re.sub(r"[^a-zA-Z?!.,]+", " ", sentence)

    sentence = sentence.strip()

    if s_token:
        sentence = '<start> ' + sentence

    if e_token:
        sentence += ' <end>'

    return sentence 


# In[35]:


enc_corpus = []
dec_corpus = []

num_examples = 30000

for pair in raw[:num_examples]:
    enc, dec = pair.split("\t")

    enc_corpus.append(preprocess_sentence(enc))
    dec_corpus.append(preprocess_sentence(dec, s_token=True, e_token=True))

print("English :", enc_corpus[100])
print("Spanish :", dec_corpus[100])


# ### 데이터 전처리 : 토큰화

# In[38]:


def tokenize(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus)
    
    tensor = tokenizer.texts_to_sequences(corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    
    return tensor, tokenizer


# In[42]:


#정제된 텍스트를 tokenize()함수를 사용해 토큰화한 후 텐서로 변환
enc_tensor, enc_tokenizer = tokenize(enc_corpus)
dec_tensor, dec_tokenizer = tokenize(dec_corpus)


# In[43]:


# 훈련데이터와 검증데이터를 8:2 분리하기
# 토크나이즈 클래스의 텐서가 훈련데이터고 목표는 스페인어이기 때문에 디코더 텐서가 타겟이다
enc_train, enc_val, dec_train, dec_val = train_test_split(enc_tensor,dec_tensor, test_size=0.2)


# In[44]:


# index_word를 활용하여 english vocab size 반환
# index_word를 활용하여 spanish vocab size 반환

print('English vocab size : ',len(enc_tokenizer.index_word))
print('Spanish vocab size : ',len(dec_tokenizer.index_word))


# In[45]:


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.w_dec = tf.keras.layers.Dense(units)
        self.w_enc = tf.keras.layers.Dense(units)
        self.w_com = tf.keras.layers.Dense(1)

    def call(self, h_enc, h_dec):
        # query size = (batch size, hidden size)
        # hidden_with_time_axis = (batch size, 1, hidden size)
        #hidden_with_time_axis = tf.expand_dims(query, 1)

        # score = (batch size, max_length, 1)
        #score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))) ###
        h_enc = self.w_enc(h_enc)
        h_dec = tf.expand_dim(h_dec, 1)
        h_dec = self.w_dec(h_dec)
        score = self.w_com(tf.nn.tanh(h_dec + h_enc))

        # attention_weights = (batch size, max_length, 1)
        #attention_weights = tf.nn.softmax(score, axis =1)
        attn = tf.nn.softmax(score, axis = 1)

        # context vector shape after sum = (batch size, hidden size)
        #context_vector = attention_weights * values
        #context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vec = attn * h_enc
        context_vec = tf.reduce_sum(context_vec, axis=1)

        #return context_vector, attention_weights
        return context_vec, attn


# ![gru%20encoder%20and%20decoder.jpg](attachment:gru%20encoder%20and%20decoder.jpg)

# In[ ]:


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units)
        
    def call(self, x):
        print('입력 shape : ',x.shape)

        x = self.embedding(x)
        print("Embedding Layer를 거친 shape ", x.shape)

        output = self.gru(x)
        print("GRU shape의 output shape:", output.shape)        
        
        return output


# In[ ]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences= True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = tf.keras.layers.Softmax(axis=-1)
        # todo

    def call(self, x, h_dec, enc_out):
        print("입력 shape :", x.shape)

        x = self.embedding(x)
        print("Embedding Layer을 거친 shape :", x.shape)

        context_v = tf.repeat(tf.expand_dims(context_v, axis=1), repeats=x.shape[1], axis=1)
        x = tf.concat([x, context_v], axis= -1)
        print("Context Vector가 더해진 shape :", x.shape)

        x = self.gru(x)
        print("GRU Layer의 Output layer:", x.shape)

        output = self.fc(x)
        print("Decoder의 최종 Output shape :", output.shape)
        # todo

        return out, h_dec, atten


# In[ ]:


BATCH_SIZE= 64
src_vocab_size = len(enc_tokenizer.index_word)+1
tgt_vocab_size = len(dec_tokenizer.index_word)+1

units = 1024
embedding_dim = 512

encoder = Encoder(src_vocab_size, embedding_dim, units)
decoder = Decoer(tgt_vocab_size, embedding_dim, units)

#sample input
sequence_len = 30

sample_enc = tf.random_uniform((BATCH_SIZE, sequence_len))
sample_output = encoder(sample_enc)

print('Encdoer Output : ', sample_output.shape)

sample_state = tf.random.uniform((BATCH_SIZE, units))
sample_logits, h_dec, attn = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_state, sample_output)

print('Decoder output :', sample_logits.shape)
print('Decoder Hidden State :', h_dec.shape)
print('Attention :')

