# -*- coding: utf-8 -*-
"""2021-12-03.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_rAqirAmlvnZUjKR54a27DGj5PmyN7e5
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

sentences = ['nice great best amazing','stop lies', 'pitiful nerd','excellent word','supreme quality','bad','highly respectable']
y_train = [1,0,0,1,1, 0,1]

t = Tokenizer()
t.fit_on_texts(sentences)
vocab_size = len(t.word_index) + 1     #단어 별로 알아서 토큰화가 된다
print(vocab_size)

#텍스트를 시퀀스 형태로 바꿔준다 = 인코딩
x_encoded = t.texts_to_sequences(sentences)
print(x_encoded)

#최대 길이를 구해서 그 보다 작은 것들의 남은 공간은 0으로 매핑을 해준다
max_len = max(len(l) for l in x_encoded) #위에 보면 원소의 갯수 4가 최대가 된다
print(max_len)

#위에서 구한 최대 길이 이하의 것들은 0으로 패딩을 준다
x_train = pad_sequences(x_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)
print(x_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding,Flatten

model = Sequential()
model.add(Embedding(vocab_size, 4,input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) #긍정 또는 부정이라 0,1이어서 덴스층은 1

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs= 100, verbose=1) # 0 1 2

"""## 네이버 영화 리뷰 감성분석
### Word2Vec
"""

!pip install konlpy

import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

"""## 데이터 준비"""

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('/content/ratings_train.txt')
test_data = pd.read_table('/content/ratings_test.txt')

train_data. head()

# Commented out IPython magic to ensure Python compatibility.
from konlpy.tag import Mecab
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab/
!bash install_mecab-ko_on_colab190912.sh

tokenizer= Mecab()

def tokenize_and_remove_stopwords(data, stopwords, tokenizer):
    result = []

    for sentence in data:
        curr_data = []
        curr_data = tokenizer.morphs(sentence) # 형태소기반으로한 토큰화
        curr_data = [word for word in curr_data if not word in stopwords] # 불용어 제거

        result.append(curr_data)
    return result

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

def load_data(train_data, test_data, num_words=10000):
  train_data.drop_duplicates(subset=['document'], inplace=True)
  test_data.drop_duplicates(subset=['document'], inplace=True)

  train_data = train_data.dropna(how='any')
  test_data = test_data.dropna(how='any')

  x_train = tokenize_and_remove_stopwords(train_data['document'], stopwords, tokenizer)
  x_test = tokenize_and_remove_stopwords(test_data['document'], stopwords, tokenizer)

  words = np.concatenate(x_train).tolist()
  counter = Counter(words)
  counter = counter.most_common(10000-4)

  vocab = ['<PAD>','<BOS>','<UNK>','<UNUSED'] + [ket for ket, _ in counter]
  word_to_index = {word:index for index, word in enumerate(vocab)}

  def wordlist_to_Indexlist(wordlist):
    return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>']for word in wordlist]

  x_train = list(map(wordlist_to_Indexlist, x_train))
  x_test = list(map(wordlist_to_Indexlist, x_test))

  return x_train, np.array(list(train_data['label'])), x_test, np.array(list(test_data['label'])), word_to_index

x_train, y_train, x_test, y_test, word_to_index = load_data(train_data, test_data)
print(x_train[0])

index_to_word = {index: word for word, index in word_to_index.items()}

def get_encoded_sentence(sentece, word_to_index): #한 문장
    return [word_to_index['<BOS>']]+ [word_to_index[word] if word in word_to_index else word_to_index['<UNK'] for word in sentence.split()]

def get_encoded_sentences(sentences, word_to_index): #여러 문장
  return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])

def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]

get_decoded_sentence(x_train[10], index_to_word)

"""## 모델 구성을 위한 데이터 분석 및 가공"""

total_data_text = list(x_train) + list(x_test)
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)

print('문장길이 평군: ',np.mean(num_tokens))
print('문장길이 최대: ',np.max(num_tokens))
print('문장길이 표준편차: ',np.std(num_tokens))

# 최대길이 (평균 + 2 * 표준편차)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad sequences maxlen :', maxlen)
print('전체 문장의 {}%가 maxlen설정값 이내에 포함됩니다.'.format(np.sum(num_tokens < max_tokens)/len(num_tokens)*100))

x_train = keras.preprocessing.sequence.pad_sequences(x_train, value = word_to_index['<PAD>'], padding='pre', maxlen = maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, value = word_to_index['<PAD>'], padding='pre', maxlen = maxlen)

print(x_train.shape)
print(x_test.shape)

"""## 모델 구성 및 validation 구성"""

import os
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Input
from sklearn.model_selection import train_test_split

vocab_size = 10000
word_vector_dim = 256 # 워드 벡터의 차원 수

# 1. RNN버전

model_rnn = keras.Sequential()
model_rnn.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model_rnn.add(keras.layers.LSTM(16, activation='relu'))
model_rnn.add(keras.layers.Dense(16, activation='relu'))
model_rnn.add(keras.layers.Dense(1, activation='sigmoid'))


# 2. 1D-CNN

model_cnn = keras.Sequential()
model_cnn.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model_cnn.add(keras.layers.Conv1D(16, 3, activation='relu'))
model_cnn.add(keras.layers.MaxPool1D(2))
model_cnn.add(keras.layers.Conv1D(16, 3, activation='relu'))
model_cnn.add(keras.layers.GlobalAveragePooling1D())
model_cnn.add(keras.layers.Dense(8, activation='relu'))
model_cnn.add(keras.layers.Dense(1, activation='sigmoid'))
#각 모델을 각각 다른 변수에 저장해주세요!

model_rnn.summary()

model_cnn.summary()

x_val = x_train[:50000]
y_val = y_train[:50000]

partial_x_train = x_train[50000:]
partial_y_train = y_train[50000:]

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 15
history_rnn = model_rnn.fit(partial_x_train, partial_y_train, epochs = epochs, batch_size=512, validation_data =(x_val, y_val), verbose=1)

# CNN1D학습
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = model_cnn.fit(partial_x_train, partial_y_train, epochs = epochs, batch_size=512, validation_data =(x_val, y_val), verbose=1)

result_rnn = model_rnn.evaluate(x_test, y_test, verbose=2)
result_cnn = model_cnn.evaluate(x_test, y_test, verbose=2)

history_rnn_dic = history_rnn.history
history_cnn_dic = history_cnn.history

"""### 오버피팅이 발생된다. 그러므로 데이터를 늘리거나 히든 사이즈를 늘려야한다"""

acc = history_rnn_dic['accuracy']
val_acc = history_rnn_dic['val_accuracy']
loss = history_rnn_dic['loss']
val_loss = history_rnn_dic['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history_cnn_dic['accuracy']
val_acc = history_cnn_dic['val_accuracy']
loss = history_cnn_dic['loss']
val_loss = history_rnn_dic['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""# 학습된 embedding 레이어 분석"""

import os

word2vec_file_path = 'word2vec.txt'
f = open(word2vec_file_path, 'w')
f.write('{} {} \n'.format(vocab_size-4, word_vector_dim))

vectors = model_rnn.get_weights()[0]
for i in range(4, vocab_size):
  f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()

from gensim.models.keyedvectors import Word2VecKeyedVectors

word_vector = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
vector = word_vector['짜증']
vector

word_vector.similar_by_word("짜증")

word_vector.similar_by_word("슬픔")

word_vector.similar_by_word("자주포")

"""### 한국어 word2vec 임베딩을 활용해서 성능 개선"""

import gensim

word2vec_path = '/content/drive/MyDrive/Colab Notebooks/ko.bin'
word2vec = gensim.models.Word2Vec.load(word2vec_path)
vector = word2vec['감동']
vector

word2vec.similar_by_word('재미')

mecab = Mecab()

def sentiment_predict(new_sentence):
    import re
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    t = Tokenizer()
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]','',new_sentence)
    new_sentence = mecab.morphs(new_sentence)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = t.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)
    score = float(model_rnn.predict(pad_new))

    if (score > 0.5): # 긍정
        print("{:.2f}% 확률로 긍정 리뷰 입니다. \n".format(score*100))
    else:
        print("{:.2f}% 확률로 부정 리뷰 입니다. \n".format((1-score)*100))

sentiment_predict('이 영화 꿀잼 ㅋㅋㅋㅋ짱짱짱')

sentiment_predict('재미없다')

"""## 네이버 쇼핑 리뷰 감성 분류하기

- 총 200,000개 리뷰로 구성
- 평점이 5점 만점에 1, 2, 4, 5인 리뷰들로 구성된 데이터
- 3점인 리뷰는 긍부정 유무가 애매해서 제외
- 평점이 4, 5인 리뷰에 긍정 -->1
- 평점이 1, 2인 리뷰에 부정 -->0
"""

# Commented out IPython magic to ensure Python compatibility.
from konlpy.tag import Mecab
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab/
!bash install_mecab-ko_on_colab190912.sh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")

!pwd

cd ../

!pwd

cd ../

!pwd

cd ../

urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")

total_data = pd.read_table('ratings_total.txt',names=['ratings','reviews'])
print('전체 리뷰 갯수 : ',len(total_data))

total_data[:5]

"""### 훈련데이터와 테스트데이터를 분리"""

total_data['label'] = np.select([total_data.ratings > 3],[1],default=0) #raings가 3 이상 인 것에 1로 지정
total_data[:5]

total_data['ratings'].nunique()

total_data['reviews'].nunique() # 특이값/고유 값 갯수 확인 이 경우에는 오직 1개만(똑같지 않은) 리뷰의 갯수

total_data['label'].nunique()

total_data.drop_duplicates(subset=['reviews'],inplace=True) #중복된 리뷰 삭제
print('중복을 제거한 샘플의 수 : ',len(total_data))

print(total_data.isnull().values.any())

train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
print('훈련용 리뷰의 갯수 : ', len(train_data))
print('테스트용 리뷰의 갯수 : ',len(test_data))

"""### 레이블(이 경우에는 0과 1/부정과 긍정)의 분포 확인"""

train_data['label'].value_counts().plot(kind='bar')

print(train_data.groupby('label').size().reset_index(name='count'))

"""### 데이터 정제하기"""

train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")

train_data['reviews'].replace('', np.nan, inplace=True)

print(train_data.isnull().sum())

print(test_data.groupby('label').size().reset_index(name='count'))

# test data
# 중복 제거
print(test_data['reviews'].nunique())
test_data.drop_duplicates(subset=['reviews'],inplace=True) #중복된 리뷰 삭제
print('중복을 제거한 샘플의 수 : ',len(test_data))

# test data
# 정규표현식을 이용하여 한글 외 문자 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
# 공백을 null 변경
test_data['reviews'].replace('', np.nan, inplace=True)
# Null값 제거
test_data = test_data.dropna(how='any')
print(train_data.isnull().sum())

# test_data 갯수 반환
print('처리를 한 테스트 샘플의 수 : ',len(test_data))

"""### 토큰화"""

mecab= Mecab()
print(mecab.morphs('이런 상품도 상품인가요? 허허허'))

"""### 불용어 제거"""

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

"""### 단어와 길이 분포 확인하기"""

negative_words = np.hstack(train_data[train_data.label==0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label==1]['tokenized'].values)

negative_word_count = Counter(negative_words)
print(negative_word_count.most_common(20))

positive_words_count = Counter(positive_words)
print(positive_words_count.most_common(20))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이 ;', np.mean(text_len))

text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이 ;', np.mean(text_len))

train_data.head()

test_data.head()

x_train = train_data['tokenized'].values
y_train = train_data['label'].values
x_test = test_data['tokenized'].values
y_test = test_data['label'].values

"""## 정수 인코딩"""

t = Tokenizer()
t.fit_on_texts(x_train)

threshold = 2
total_cnt = len(t.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in t.word_counts.items():
    total_freq = total_freq + value

    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집한 (vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀단어의 수 : %s'%(threshold-1, rare_cnt))
print('단어 집합에서 희귀단어의 비율 :', (rare_cnt/total_cnt)*100)
print('전체 등장 빈도에서 희귀단어 등장 빈도 비율 :', (rare_freq/total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 : ',vocab_size)

original_vocab_size = vocab_size + rare_cnt -2
print('원래 vocab size :', original_vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

print(x_train[:3])
print(x_test[:3])

"""## 패딩"""

print('리뷰의 최대 길이:', max(len(l) for l in x_train))
print('리뷰의 평균 길이 :', sum(map(len, x_train))/len(x_train))
plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of samples')
plt.xlabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt +1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율 : %s'%(max_len, (cnt/len(nested_list))*100))

max_len = 80
below_threshold_len(max_len, x_train)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print(x_train.shape)
print(x_test.shape)

"""## 모델 만들기"""

from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_size = 128

# Embedding
model_gru = Sequential()
model_gru.add(Embedding(vocab_size, 100))
model_gru.add(GRU(hidden_size))
model_gru.add(Dense(1, activation='sigmoid'))

#GRU
#Dense

#es 얼리 스탑핑
#mc
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#model.compile
model_gru.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
history_gru = model_gru.fit(x_train, y_train, epochs=1, callbacks=[es, mc], batch_size= 60, validation_split=0.2)

#history = model.fit
model_gru.evaluate(x_test, y_test)[1]

def sentiment_predict(new_sentence):
    #new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]','',new_sentence)
    new_sentence = mecab.morphs(new_sentence)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)
    score = float(model_gru.predict(pad_new))

    if (score > 0.5): # 긍정
        print("{:.2f}% 확률로 긍정 리뷰 입니다. \n".format(score*100))
    else:
        print("{:.2f}% 확률로 부정 리뷰 입니다. \n".format((1-score)*100))

sentiment_predict('이 상품은 진짜 너무너무 좋아요!')

sentiment_predict('이 상품은 진짜 너무너무 별로옝!')

sentiment_predict('이제 수업이 끝난 것 같아요!')