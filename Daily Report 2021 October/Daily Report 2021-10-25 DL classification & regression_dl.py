# -*- coding: utf-8 -*-
"""20211025 DL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T5bC_L8aiN0ylmYQdx039ktqZ7Yv73xF
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

test1 = [1,2,3]
test2 = [10,20,30]

t1 = tf.Variable(test1, dtype=tf.float32)
t2 = tf.Variable(test2, dtype=tf.float32)

with tf.GradientTape() as tape:
  t3 = t1+t2

gradient = tape.gradient(t3, [t1,t2])
print(gradient[0])  #0번 그러니까 t1에 대한 미분이니 t2의 값이 나온다
print(gradient[1])   #1번에 대한 미분이니 t1의 값이 나온다

t1 = tf.constant (test1, dtype=tf.float32)  #constant 즉, 상수값이기 때문에 특별한 키워드나 명령이 없으면 학습이 안된다.
t2 = tf.Variable (test2, dtype=tf.float32)

with tf.GradientTape() as tape:
  tape.watch(t1)     #상수 constant를 학습 가능하게 해주는 키워드 tape.객체
  t3=t1+t2

gradients = tape.gradient(t3, [t1, t2])
print(gradients[0])
print(gradients[1])

"""Gradient tape를 이요한 linear regression"""

## data 선언
x_data =[[1.],[2.],[3.],[4.]]
y_data =[[1.],[3.],[5.],[7.]]

plt.scatter(x_data, y_data)

## 평균 0, 분산 1의 파라미터의 정규분포로 부터 값을 가져옴.
# 학습을 통해 업데이트가 되어 변화되는 모델의 파라미터인 w,b를 의미한다.
W=tf.Variable(tf.random.normal((1,1),mean=0, stddev=1.0))
b=tf.Variable(tf.random.normal((1,1),mean=0, stddev=1.0))
#lr=tf.constant(0.0001)
lr = 0.01
history = np.zeros([2000,3], 'float32')

w = np.array(W)
w= w.reshape(1)
B = np.array(b)
B=B.reshape(1)

plt.plot(x_data, y_data, 'o')
plt.plot([0,4], [b, (w*4 +B)], 'r-')

w_trace = []
b_trace = []

'''
for epoch in range(2000):
  total_error = 0

  for x,y in zip(x_data, y_data):
    with tf.GradientTape() as tape:
      y_hat = W + x + b
      error = (y_hat - y)**2

    gradients = tape.gradient(error, [W, b])  #미분해서 업데이트 시켜야하는 값을 대괄호 안에 넣어준다

    W= tf.Variable(W - lr * gradients[0])
    b= tf.Variable(b-lr * gradients[1])

    w_trace.append(W.numpy())
    b_trace.append(b.numpy())

    visual_error = tf.square(error)
    total_error = total_error + visual_error

  print("epoch : ",epoch, 'error :',total_error/len(x_data))
  history[epoch,:] = [(total_error/len(x_data))[0], W[0], b[0]]
'''

for epoch in range(200):
    total_error = 0

    for x, y in zip(x_data, y_data):
        with tf.GradientTape() as tape:   #여기서 W + x +b를 미분하고 싶은 것이고 
            y_hat = W * x + b   #W는 가중치 weight , b는 치우침 bias
            error = (y_hat - y) **2   #error가 타겟
        
        gradients = tape.gradient(error, [W, b])   #W와 b의 미분값=error를 알고 싶어서 한 것

        W = tf.Variable(W - lr * gradients[0])  #경사하강법 gradents가 -면 +기울기로 나올 것이고
        b = tf.Variable(b - lr * gradients[1]) #gradents가 +면 -기울기로 업데이트가 될 것이다 이것이 경사하강법 

        w_trace.append(W.numpy())  #여기에 W와 b를 업데이트 한다 사실 필요없는 과정인데 이해를 위해 여기에 업데이트를 시키고 그래프를 그린 것이다
        b_trace.append(b.numpy())

        visual_error = tf.square(error)
        total_error = total_error + visual_error

    print("epoch : ", epoch, "error :", total_error/len(x_data))
    history[epoch,:] = [(total_error/len(x_data))[0], W[0], b[0]]

w = np.array(W)
w = w.reshape(1)
B = np.array(b)
B = B.reshape(1)

plt.plot(x_data, y_data, 'o')
plt.plot([0, 4], [b, (w*4 + B)], 'r-')

#학습이 끝난 후 W와 B로 예측
print(history)
print("W :", W)
print("b:", b)
print("input 3", tf.add(tf.matmul([[3.]], W), b))
print("input 4", tf.add(tf.matmul([[4.]], W), b))

#loss function
plt.plot(history[:,0])   #epoch 1번부터 2000까지(error값) 그래프를 그린것
plt.title('loss function')

a=np.array(w_trace)
a = a.reshape(800,1)

plt.plot(a)

x_data = [[2.,0.,7.], [6.,4.,2.], [5.,2.,4.],[8.,4.,1]]
y_data = [[75], [95], [91], [97]]
test_data=[[5.,5.,5.]]
print(len(x_data),len(x_data[1]))  # 행크기 , 열크기

"""Multi regression"""

model =tf.keras.Sequential()
## tf.keras를 활용한 perceptron 모델 구현. 
## 모델 만들기 위해 sequential 매서드를 선언. 이를 통해 모델을 만들 수 있다.
model.add(tf.keras.layers.Dense(1,input_dim=3)) # hidden layer가 1이다
# 선언된 모델에 add를 통해 쌓아감. , 현재는 입력 변수 갯수 3, perceptron 1개.

model.summary() ## 설계한 모델 프린트

"""dense가 1인 이유는 히든레이어가 1개/구해야하는 값이 1개이기 때문이다
shape이 1인 이유는 x_data의 행렬의 모양과는 관계없이 예측해야하는 값이 1이기 때문(1x1행렬/스칼라)
위의 Param 값이 4개인 이유는 w1x + w2x+w3x+b(bias) 다 합쳐서 4개라는 뜻임 위의 경우는 x_data의 3x1행렬의 원소 3개와 b값의 총 갯수를 뜻하는 것이다
"""



# 모델 loss, 학습 방법 결정하기
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)      ### 경사 하강법으로 global min 에 찾아가는 최적화 방법 선언.
loss=tf.keras.losses.mse  ## 예측값 과 정답의 오차값 정의. mse는 mean squre error로 (예측값 - 정답)^2 를 의미
metrics=tf.keras.metrics.mae  ### 학습하면서 평가할 메트릭스 선언 mse는 mean_absolute_error |예측값 - 정답| 를 의미

# 모델 컴파일하기
model.compile(loss=loss,optimizer=optimizer, metrics=[metrics])

# 모델 동작하기
model.fit(x_data,y_data,   epochs=200,   batch_size=4)

"""epoch에서 1/1인 이유는 x_data의 갯수가 4인데 batch도 4이니 4개를 4개씩 보는 것이니 1/1이 된다, batch를 2로 하면 2/2가 될 것이다."""

# 결과를 출력합니다.
print(model.weights)
print(" test data [5.,5.,5.] 예측 값 : ", model.predict(test_data))

## data 선언
x_data = [[0.,0.], [0.,1.], [1.,0.],[1.,1.]]
y_data = [[0.], [1.], [1.], [1.]]
test_data=[[0.3, 0.3]]

## tf.keras를 활용한 perceptron 모델 구현.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=2))
model.summary() ## 설계한 모델 프린트

# 모델 loss, 학습 방법 결정하기
optimizer=tf.keras.optimizers.SGD(lr=0.01)  ### 경사 하강법으로 global min 에 찾아가는 최적화 방법 선언.
loss=tf.keras.losses.mse  ## 예측값 과 정답의 오차값 정의. mse는 mean squre error로 (예측값 - 정답)^2 를 의미
metrics=tf.keras.metrics.mae ### 학습하면서 평가할 메트릭스 선언

# 모델 컴파일하기
model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

# 모델 동작하기
model.fit(x_data, y_data, epochs=200, batch_size=4)

# 결과를 출력합니다.
print(model.weights)
print(" test data [0.3, 0.3] 예측 값 : ", model.predict(test_data))
if model.predict(test_data)>0.5:
    print(" 합격 " )
else:
    print(" 불합격 ")

## data 선언
x_data = [[5.], [30.], [95.], [100.], [265.], [270.], [290.], [300.],[365.]]
y_data = [[0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.]]
test_data= [[7.]]
test_data2= [[80.]]
test_data3= [[110.]]
test_data4= [[180.]]
test_data5= [[320.]]

## tf.keras를 활용한 perceptron 모델 구현.
model = tf.keras.Sequential() ## 모델 선언
model.add(tf.keras.layers.Dense(1, input_dim=1, activation='sigmoid'))   #시그모이드 함수 적용, 시그모이드 함수에 영향을 받아 비선형적으로 값을 받는다
model.summary()

# 모델 loss, 학습 방법 결정하기
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
loss=tf.keras.losses.mse
metrics=tf.keras.metrics.binary_accuracy  #이진 분류법 0아니면 1이나온다

# 모델 컴파일하기 - 모델 및 loss 등 구조화한 모델을 컴퓨터가 동작 할수 있도록 변환
model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

# 모델 동작하기
model.fit(x_data, y_data, epochs=200, batch_size=9)

# 결과를 출력합니다.
print(" test data [7.] 예측 값 : ", model.predict(test_data))
print(" test data [80.] 예측 값 : ", model.predict(test_data2))
print(" test data [110.] 예측 값 : ", model.predict(test_data3))
print(" test data [180.] 예측 값 : ", model.predict(test_data4))
print(" test data [320.] 예측 값 : ", model.predict(test_data5))

# 모델 loss, 학습 방법 결정하기
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
loss=tf.keras.losses.categorical_crossentropy
metrics=tf.keras.metrics.binary_accuracy  #이진 분류법 0아니면 1이나온다

# 모델 컴파일하기 - 모델 및 loss 등 구조화한 모델을 컴퓨터가 동작 할수 있도록 변환
model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

# 모델 동작하기
model.fit(x_data, y_data, epochs=200, batch_size=9)

# 결과를 출력합니다.
print(" test data [7.] 예측 값 : ", model.predict(test_data))
print(" test data [80.] 예측 값 : ", model.predict(test_data2))
print(" test data [110.] 예측 값 : ", model.predict(test_data3))
print(" test data [180.] 예측 값 : ", model.predict(test_data4))
print(" test data [320.] 예측 값 : ", model.predict(test_data5))

## data 선언
x_data =[[0.,0.],[0.,1.],[1.,1.],[1.,1.]]
y_data =[[0.],[1.],[1.],[0.]]
test_data = [[0.5, 0.5]]

## tf.keras를 활용한 perceptron 모델 구현.
model = tf.keras.Sequential() ## 모델 선언
model.add(tf.keras.layers.Dense(4, input_dim=2,activation='sigmoid'))
model.add(tf.keras.layers.Dense(5, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,  activation='sigmoid'))

model.summary()

# 모델 loss, 학습 방법 결정하기
#optimizer=tf.keras.optimizers.SGD(lr=0.5)
optimizer=tf.keras.optimizers.SGD(lr=1.5)
loss=tf.keras.losses.binary_crossentropy
metrics=tf.keras.metrics.binary_accuracy


# 모델 컴파일하기
model.compile(loss=loss, optimizer=optimizer,metrics=[metrics])

# 모델 동작하기
model.fit(x_data, y_data, epochs=2000, batch_size=4)

print(" test data [0.5 0.5] 예측 값 : ", model.predict(test_data))

import pandas as pd
from google.colab import files

file_uploaded = files.upload()

house_price_data = pd.read_csv('./house_price_of_unit_area.csv')

print(house_price_data.info())

x_data = house_price_data.copy()
tf.random.set_seed(777)  #랜덤 시드를 고정해줌

y_data = x_data.pop("house price of unit area")

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=5,))
model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(300, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation=None))

model=tf.keras.Model(inputs =[input_layer], outputs=[output_layer])
model.summary()

optimizer=tf.keras.optimizers.SGD(learning_rate=0.04) ### 경사 하강법으로 global min 에 찾아가는 최적화 방법 선언.
loss=tf.keras.losses.mean_squared_error  ## 예측값 과 정답의 오차값 정의.
metrics=tf.keras.metrics.RootMeanSquaredError() ### 학습하면서 평가할 메트릭스 선언언

model.compile(loss =loss, optimizer= optimizer, metrics=[metrics])

result = model.fit(x_data, y_data, epochs=100, batch_size=100)

print(result.history.keys())

### history에서 loss key를 가지는 값들만 추출
loss = result.history['loss']

### loss그래프화
epochs = range(1, len(loss) + 1)
plt.subplot(211)  ## 2x1 개의 그래프 중에 1번째
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

### history에서 root_mean_squared_error key를 가지는 값들만 추출
rmse = result.history['root_mean_squared_error']
epochs = range(1, len(rmse) + 1)
### mean_absolute_error를 그래프화
plt.subplot(212)  ## 2x1 개의 그래프 중에 2번째
plt.plot(epochs, rmse, 'r-', label='Training rmse')
plt.title('Training rmse')
plt.xlabel('Epochs')
plt.ylabel('rmse')
plt.legend()

print("\n Test rmse: %.4f" % (model.evaluate(x_data, y_data)[1]))
plt.show()

print(model.evaluate(x_data,y_data))
print("\n Test rmse : %.4f" % (model.evaluate(x_data, y_data)[1]))

