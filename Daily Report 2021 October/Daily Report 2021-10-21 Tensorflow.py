#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install tensorflow')


# In[14]:


import tensorflow as tf
import numpy as np


# In[23]:


b=tf.Variable(0)
for i in range(1,11):
  b=b+i
print("다 더한값 : ",format(b))

a=tf.constant(2)
c=tf.constant(5)
d=tf.constant(3)

mul = tf.multiply(c,a)
add = tf.add(d,a)
sub = tf.subtract(mul,add)
print("사칙연산 수행 : ",format(sub))


# In[9]:


x1=tf.Variable(5)
x2=tf.Variable(3)
x3=tf.Variable(2)

add_result = tf.add(x2,x3)
mul_result = tf.multiply(x1,x3)
sub_result = tf.subtract(mul_result,add_result)

print(sub_result)


# In[24]:


# 랜덤 변수 선언하기

# 평균이 0이고, 분산이 1인 파라미터의 정규분포로부터 값 구하기
x1 = tf.Variable(tf.random.normal((1,1), mean=0, stddev=1.0))
x2 = tf.Variable(tf.random.normal((1,1), mean=0, stddev=1.0))
x3 = tf.Variable(tf.random.normal((1,1), mean=0, stddev=1.0))

add_result = tf.add(x2, x3) # 3+2
mul_result = tf.multiply(x1, x3)
sub_result = tf.subtract(mul_result, add_result)

print("add_result = {}".format(add_result))
print("mul_result = {}".format(mul_result))
print("sub_result = {}".format(sub_result))


# In[25]:


# 다차원 변수 선언하기

x1 = tf.Variable([[1,2], [3,4]])

print("x1 = {}".format(x1))
print("x1 = {}".format(x1.shape))


# In[27]:


x1_reshape=tf.reshape(x1,[4,1])

print("x1 reshaped",format(x1_reshape.shape))
print("x1_reshape=",format(x1_reshape))


# In[41]:


matrix_A=tf.constant([[2,2,4,],[1,1,6],[1,3,8]])
matrix_B=tf.constant([[4,3,3,],[2,1,6],[1,2,8]])
matrix_A


# In[42]:


matrix_A_re=tf.reshape(matrix_A,[3,3])
matrix_B_re=tf.reshape(matrix_B,[3,3])
matrix_A_re


# In[43]:


matrix_mul=tf.multiply(matrix_A_re,matrix_B_re)
print("matirx_A x matrix_B = ",format(matrix_mul))


# In[44]:


matrix_AB = tf.matmul(matrix_A,matrix_B)
print(matrix_AB)
print(matrix_mul)
matrix_AandB = tf.matmul(matrix_A_re,matrix_B_re)
print(matrix_AandB)


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


plt.imshow(matrix_mul)
plt.show()


# In[60]:


x_data = tf.constant([[1.],[2.],[3.],[4.]])
y_data = tf.constant([[5.],[6.],[7.],[8.]])

print(x_data)
print(y_data)


# In[68]:


#평균 0 ,분산 1의 파라미터의 정규분포로 부터 값을 가져옴,
#학습을 통해 업데이트가 되어 변화되는 모델의 반환값을 w,b라고 함
W=tf.Variable(tf.random.normal((1,1), mean=0, stddev=1.0))
b=tf.Variable(tf.random.normal((1,1), mean=0, stddev=1.0))



print("w : ",w)
print('b : ',b)


# In[69]:


for j in range(len(x_data)):
    #data * weight 작성
    WX= tf.matmul([x_data[j]], w)
    
    #bias add 작성
    y_hat = tf.add(WX, b)
    
    lr=tf.constant(0.0001)
    print("y_data : ",y_data[j],"prediction : ",y_hat)


# In[72]:


x_data = tf.constant([[1.],[2.],[3.],[4.]])
y_data = tf.constant([[5.],[6.],[7.],[8.]])
lr=tf.constant(0.0001)
histroy = np.zeros([2000,3],'float32')


# In[73]:


for i in range(2000):  ## 에폭 여기서는 전체 데이터를 2000개를 본다는 것
    total_error = 0

    for j in range(len(x_data)): ## 배치 1
        ## data * weight
        WX =tf.matmul([x_data[j]], W) #[1.]*[1x1 : w:init - 여기서는 위의 랜덤값]
                                      #[2.]*[w 업데이트]
        ## bias add
        y_hat = tf.add(WX, b)  #y=WX+b

        ## 정답인 Y와 출력값의 error 계산
        error =tf.subtract(y_data[j],y_hat)  #실제(정답)-예측값

        ## 경사하강법으로 W와 b 업데이트.
        ## 도함수 구하기
        diff_W =tf.multiply(x_data[j],error) #error* x의합(summation)
        diff_b = error  #b는 편향성이기때문에 그냥 error를 업데이트하면됨

        ##  업데이트할 만큼 러닝레이트 곱
        diff_W =tf.multiply(lr,diff_W)
        diff_b =tf.multiply(lr, diff_b)  #lr *error

        ## w, b 업데이트
        W =tf.add(w, diff_W)  # w + delta(변화량) w +(lr*x*error)
        b =tf.add(b, diff_b)    #b + delta(변화량) b +(lr*error)
        #######

        ## 토탈 에러.
        visual_error = tf.square(tf.subtract(y_hat, y_data[j]))
        total_error = total_error + visual_error

    ## 모든 데이터에 따른 error 값
    print("epoch: ", i, "error : ", total_error/len(x_data))
    histroy[i,:] = [(total_error/len(x_data))[0],w[0],b[0]]


# In[75]:


#학습이 끝난 후 w와 b로 예측하기
print(histroy)
print('w: ',w)
print('b: ',b)
print('input 3: ',tf.add(tf.matmul([[3.0]],w),b))
print('input 4: ',tf.add(tf.matmul([[4.0]],w),b))


# In[86]:


## data 선언
x_data =[[1.],[2.],[3.],[4.]]
y_data =[[1.],[3.],[5.],[7.]]

test_data=[[4.]]


# In[83]:


## tf.keras를 활용한 perceptron 모델 구현.
model =tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_dim=1))#Dense가 wx+b 즉 w,b를 학습가능
# 선언된 모델에 add를 통해 쌓아감. , 현재는 입력 변수 갯수 1, perceptron 1개., inpu
model.summary() ## 설계한 모델 프린트


# In[84]:


#모델의 옵션 지정
# 모델(여기서는 Dense) loss, 학습 방법 결정하기
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01) ### 경사 하강법으로 global min 에 찾아가는 최적화 방법 선언.
loss= tf.keras.losses.mse ## 예측값 과 정답의 오차값 정의. mse는 mean squre error로 (예측값 - 정답)^2 를 의미
metrics= tf.keras.metrics.mae    ### 학습하면서 평가할 메트릭스 선언 mae는 mean_absolute_error |예측값 - 정답| 를 의미

# 모델 컴파일하기
model.compile(loss=loss,optimizer=optimizer, metrics=[metrics])
#여기서 옵션으로 ( loss='SGD',optimaizer='mse',metrics='mae')도 가능

# 모델 동작하기
model.fit(x_data,y_data,   epochs=2000, batch_size= 1)


# In[87]:


#결과를 출력
print(model.weights)
print('test data[4.] 예측값 : ',model.predict(test_data))


# In[ ]:




