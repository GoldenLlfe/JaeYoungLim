#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keyword
print(keyword.kwlist)


# In[ ]:





# In[8]:


a


# In[9]:


print(10+20)

10*30


# In[10]:


a =10
b=20
a+b


# In[11]:


# 10+20
a="asdf"
print(a)


# In[12]:


a


# %%markdown
# 테스트

# In[16]:


print(333, 666, 999, "Gutentag")
print()
print("ㅎㅇ", "내", "이름은", "홍길동이야!")


# In[20]:


#뻘짓입니다-주석연습
print("심심하다")


# #chap02
#     ##01.자료형

# # CHAP 02
# ## 01.자료형

# In[22]:


# type()함수는 자료형을 읽는다
print(type(1))
print(type("hell"))
print(type(True))
print('Hello', "Nice to meet you")


# In[34]:


# 이스케이프(\) 문자열
print('"잘 있어."라고 그는 말했습니다.')
print('\'잘 가.\'라고 그녀는 생각했습니다.')
print()
# \t 탭 이스케이프 문자열
print("이름\t나이\t지역")
print("홍길동\t250\t개성")
print("고길동\t45\t쌍문동")
print("YGL\t28\t학동")
print()
#줄 바꾸기 문자열(\n)
print("Cogito\nergo\nsum")
print()
#따옴표 3번 반복으로 줄 바꾸기
print("""Cogito
ergo
sum""")


# In[39]:


#문자열 연산자
print("안녕"+"하세요")
print()
print(3*"제발")
print()
print(3*"제발"+" 살려주세요")


# In[42]:


#문자열 인덱스
a="Gutentag"

#제로 인덱스는 0부터 셈을 하는데 여기서 0은 첫번째를 뜻하기 때문에 맨 앞 글자만 출력한다
print(a[0], "\t" , "Gutentag[0]", "Gutentag"[0])
print("a[2] : ", a[3], "\t" , "Gutentag[-1] : ", "Gutentag"[-1]) #-1은 마지막 위치


# In[45]:


#문자열 자르기 [시작:끝] -> 시작은 포함, 끝은 포함하지 않음
print("a[1:3] : ", a[1:3], "\t" , "Gutentag[3] : ", "Gutentag"[3])
print(a[:]) #전체 문자열
print(a[2:]) #2번째 인덱스부터 끝까지
print(a[:3]) #처음부터 3번째 인덱스 전가지
print(a[9]) #error index 범위를 벗어남


# In[49]:


#len()함수는 문자열의 길이를 읽는다
print(len("안녕하세요"))


# In[54]:


#숫자형 연산자: + - * /  //  % **
a=5 ; b=2
print("a+b = ", a+b)
print("a-b = ",a-b)
print("a/b = ", a/b) #자동으로 float형으로 바꾼다
print("a*b = ", a*b)
print("a//b = ", a//b) #나눗셈후 소수점을 버리고 정수만 출력
print("a%b = ", a%b) #몫만 출력
print("a**a = ", a**b) #제곱

#숫자형 연산자의 우선순위
print(2 + 2 - 2 * 2 / 2 *2) #혼자공부하는파이썬 69쪽 참고 너무 길다... 곱셈과 나눗셈이 우선순위이며 왼쪽부터 계산을 한다 그래서 2*2를 먼저하고 그후에 2로 나누고 또 2를 곱한뒤에 맨 앞에 2+2를 한 것에 뺄셈을 한다 결과는 0
print((2 + 2 - 2) * 2 / 2 *2) #알던 것처럼 괄호 안부터 계산을 한뒤 곱셈 나눗셈을 한다.괄호는 계산 실수를 방지하는 아주 좋은 것이니 습관을 들이는 것이 좋다.


# In[60]:


#변수 선언과 할당
pi = 3.14159265
r=10

#변수 참조
print("원주율 =", pi)
print("반지름 =",r)
print("원의 둘레 =", 2*pi*r) #원의 둘레
print("원의 넓이 =", pi*r*r) #원의 넓이


# In[62]:


print("pi : ", pi)
pi = "pi"
print("pi : ", pi) #변수는 데이터 타입과 무관하게 다른 데이터 타입으로 재사용 가능


# In[64]:


#복함 대입연산자 += -= /= */ **=
number = 100
number += 30 # 130
number *= 2 # 260
number /= 10 # 26
print(number)

string = "Hello"; string += '!'; string += '?' ; print(string)
string *=2; print(string)


# In[71]:


#대입연산자 : 변수 = 값  ->값을 변수에 입력한 것임
# 문 1) 숫자 변수와 문자열 변수를 선언하여 숫자변수에 10을, 문자열 변수에 string을 입력
#2) 두변수의 값을 출력
#3) 문자열 변수를 숫자 변수의 수만큼 반복해서 출력
string 

# 1번문제
a = "string"
b = 10

print()
# 2번문제
print("a =", a)
print("b = ", b)

#3번문제
print(a*b)


# In[88]:


# input(): 입력을 받아 변수에 저장해서 사용 기본적으로 문자열 자료형이라 수를 이용하려면 int(input())을 통해 숫자로 형병환을 해야한다
input()


# In[83]:


# 문제 : input()함수를 이용하여 키보드에서 문자열 (임의), blank (1), 숫자(1)을 입력받아
#문자와 숫자를 각각 변수에 저장한 후 입력된 문자열을 입력한 숫자만큼 반복해서 출력
#문자를 처음부터 -2 인덱스까지 저장, 숫자는 (-1)로 잘라서 각각의 변수에 저장 후 실행
#숫자가 문자로 인식 int(숫자형변수) -> int로 변환 됨
var = input() #키보드로부터 입력받은 자료를 var에 저장
var2 = var[0 : 5] #문자열만 var2 변수에 저장
var3 = int(var[-1]) #마지막 입력한 숫자를 va3에 저장

print("입력받은 문자열을 입력한 숫자만큼 반목하면 ", var2*var3)


# In[90]:


#두 개의 숫자를 입력받아 변수에 저장한 후
#1.키보드로부터 변수를 입력 받아 변수에 저장
#2.슬라이싱으로 숫자를 각각의 변수에 입력
#3.타입 변환 후 실행
#두 숫자의 +, *, /, -의 결과를 출력
num1=int(input())
num2 = int(input())
print()
print("num1+num2 = ", num1+num2)
print("num1-num2 = ", num1-num2)
print("num1*num2 = ",num1*num2)
print("num1/num2 = ",num1/num2)
print()

print("(num1+num2) * num1 / num2 - num2 = ", (num1+num2) * num1 / num2 - num2)


# In[100]:


#혼자 공부하는 파이썬 90페이지의 5번 문제 풀기
str_input = input("원의 반지름 입력 > ")
str_input = float(str_input)
num_input = float(str_input)#교제에 있어서 그대로 베꼈지만 필요없는 코드임
pi = 3.14
print()
print("반지름: ",str_input)
print("둘레: ", 2*pi*str_input)
print("넓이: ", pi*str_input**2)

