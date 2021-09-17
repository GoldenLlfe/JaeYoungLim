#!/usr/bin/env python
# coding: utf-8

# Python의 자료형: 숫자, 문자열, 불리언
# Input(), print(), int(), float(), str()
# 식별자 : 키워드 사용 불가, 문자로 시작, 숫자 포함 가능, 대소문자 구별
# 변수와 함수를 구별하는 방법 : 함수는 식별자 뒤에 ()괄호가 있다
# 클래스 BeautifulSoup() : 캐멀타입의 식별자
# 
# 문자 slicing하는 방법 : 인덱스를 변수[] 대괄호 안에 숫자를 넣어 이용

# In[7]:


#문제) 문자를 입력받아 2번째부터 5글자만 잘라서 화면에 출력하세요
char = input("6글자 이상의 문자를 입력하세요 >> ")
print(char[1:6]) #파이썬의 index는 0부터 시작
#문제 입력받은 문자열의 마지막 3글자를 출력하세요
print(char[-3:])


# In[19]:


#format() 함수 : 중괄호 포함한 문자열 뒤에 마침표 찍고 format()함수를 사용하되, 중괄호 개수와 format함수 안 매개변수의 개수는 반드시 같아야함.
num = int(input())
num_to_char = "{}".format(num)
print(num_to_char)
print(type(num_to_char))


# In[20]:


input_data = input("두 수를 입력하세요 > ")
input_num1 = input_data[0]
input_num2 = input_data[2]
print(input_data.format(input_num1, input_num2))


# In[5]:


#교제 95페이지
#정수
output_1 = "{:d}".format(51)
#특정 칸에 출력하기
output_2 = "{:4d}".format(52) #4칸
output_3 = "{:8d}".format(53) #8칸
#빈칸을 0으로 채우기
output_4 = "{:04d}".format(54) #4칸 빈곳은 0으로
output_5 = "{:04d}".format(-55) #음수4칸 빈곳은 0으로

print("기본")
print(output_1)
print("#특정 칸에 정수 출력")
print(output_2)
print(output_3)
print("#빈칸을 0으로 채우기")
print(output_4)
print(output_5)


# In[2]:


#소수점 아래 자릿수 지정하기
output_a = "{:10.3f}".format(12.345)
output_b = "{:10.2f}".format(12.345)
output_c = "{:10.1f}".format(12.345)

print(output_a)
print(output_b)
print(output_c)


# In[6]:


#문자열 관련 함수 :upper(), lower(), strip(), lstrip(), rstrip()
char = "   파이썬은 재미있는 프로그래밍 툴   "
print(char.upper())
print(char.lower())
print("-{}-".format(char))
print("-{}-".format(char.strip()))
print("-{}-".format(char.lstrip()))
print("-{}-".format(char.rstrip()))
print()
print(char.isalnum()) #알파벳인지 숫자인지 확인


# In[13]:


#find(),rfind(), in, split() 써보기

char = "가나 다라 마바사"
print("index : ", char.find("나"),char[char.find("나")]) #단어가 나오는 첫번째 index를 return결과로 출력
print("index : ", char.rfind("가"),char[char.find("가")])

print("마바" in char) #원하는 단어가 문장에 있는지 확인
print("아자" in char)

char = char.split()
print("char : ", char)

char2 = "아자,차카,타파하"
char2.split(',') #,로 단어를 분리


# In[31]:


#문제. 키보드에서 두 숫자를 입력받아 두 숫자의 나눈 결과을 계산해서 화며너에 출력
#자리수는 소수점 미만 2자리까지 표현해서 출력

num = input("두 수를 입력하세요 > ")
split_num = num.split()  #입력받은 값을 공백을 기준으로 나눈다
integer1 = float(split_num[0]) #split을 했을때 한 0번째 개체를 실수형으로 저장
integer2 = float(split_num[1])
print("첫 번째 숫자는 : ",integer1)
print("두 번째 숫자는 : ",integer2)
print("두 숫자를 나눈 결과 : ","{:5.3f}".format(integer1/integer2))
print("두 숫자를 나눈 결과 : ",float(integer1/integer2))
"{:5.3f}".format(integer1/integer2)


# In[32]:


x = 10
under_20 = x < 20
print("under_20 : ", under_20)
print()

#문) 키보드에서 두 수를 입력받아 두 수가 같으면 True를 다르면False를 출력
num = input("두 수를 입력하세요 > ")
split_num = num.split()
first_int = int(split_num[0])
second_int = int(split_num[1])

first_int == second_int


# In[ ]:


#if조건문
#   들여쓰기로 문장 처리
if True:
    print("조건문 처리")
    print("조건문 처리2")
if False:
    print("조건문 처리3")
    print("조건문 처리4")
print("조건문 나옴")

if (10 == 100) and (20 < 100):
    print(" and False")
if (10 == 100) or (20 < 100):
    print(" or True")


# In[49]:



#문제) 1.두 수를 입력받아 두 수가 0보다 크면 두 수의 합을 출력
#2. 두 수중 큰 수에서 작은 수를 뺀 결과 출력
#3. 두 수중 0가 없으면 작은 수를 큰수로 나누고 나머지 값 출력

num = input()
split_num = num.split()
num1 = int(split_num[0])
num2 = int(split_num[1])
print()
print("num1 = ", num1)
print("num2 = ", num2)

print("1번 문제")
if num1 > 0 and num2 > 0:
print("두 수가 0보다 작습니다")
print("두 수를 합친 값 = ",num1+num2)
print("2번 문제")
if num1 > num2:
print("num1이 num2보다 큽니다")
print("num1에서 num2를 뺀 값 = ",num1-num2)
num1big = float(num1 // num2)
if num2 > num1:
print("num2가 num1보다 큽니다")
print("num2에서 num1를 뺀 값 = ", num2-num1)
num2big = float(num2 // num1)
print("3번 문제")
if num1big:
print("num1을 num2로 나눈 나머지 값 = ", num1big)
if num2big:
print("num2를 num1으로 나눈 나머지 값 = ", num2big)


# In[61]:


#datetime 패키지 사용 import 패키지명, import 패키지 as 약어(aloas 명)
# 패키지명.함수명
import datetime as dt


now = dt.datetime.now()
print("현재 시각은", now.year,"년", now.month,"월", now.day, "일", now.hour , "시",now.minute,"분",now.second,"초"," 입니다.")


# In[95]:


#문제) 1.두 수를 입력받아 두 수가 0보다 크면 두 수의 합을 출력
#2. 두 수중 큰 수에서 작은 수를 뺀 결과 출력
#3. 두 수중 0가 없으면 작은 수를 큰수로 나누고 나머지 값 출력
#if - else문으로 수정

num = input()
split_num = num.split()
num1 = int(split_num[0])
num2 = int(split_num[1])
print()
print("첫 번째 수 = ", num1)
print("두 번째 수= ", num2)

print("1번 문제")
if num1 > 0 and num2 > 0 :
    print("두 수가 0보다 작습니다")
    print("두 수를 합친 값 = ",num1+num2)
else:
    print("두 수가 0보다 작은수 입니다. 다시 실행 후 수를 입력해 주세요.")
print("2번 문제")
if num1 > num2:
    print("첫 번째 수가 두 번째 수보다 큽니다")
    print("첫 번째 수에서 두 번째 수를 뺀 값 = ",num1-num2)
    num1big = float(num1 // num2)
elif num2 > num1:
    print("두 번째 수에 첫 번째 수를 뺀 값 = ", num2-num1)
    num2big = float(num2 // num1)
else:
    print("두 번째 수는 첫 번째 수보다 작습니다.")
print("3번 문제")
if num1big:
    print("첫 번째 수를 두 번째 수로 나눈 나머지 값 = ", num1big)
else:
    print("두 번째 수를 첫 번째 수로 나눈 나머지 값 = ", num2big)


# In[80]:


#if, if~ else, if~elif~ ... else
#성적을 입력 받아 등급을 부여해서 출력
#60점 미만은 F, 61~70 D, 71~80 C, 81~90 B, 91~100 A

score = int(input("성적을 입력해 주세요 > "))
if score > 90 :
    print("A등급 입니다")
elif 80 < score :
    print("B등급 입니다")
elif score > 70 :
    print("C등급 입니다")
elif score > 60 :
    print("D등급 입니다")
else:
    print("F등급 입니다")


# In[92]:


#문제) 숫자 연산자기호 수자를 입력받아 연산자 기호가 '+'면 두 숫자의 합을
# '-'이면 두 수의 차를, '*'이면 두 수의 곱을, 아니면 "기호 오류" 문자 출력

input_num = input("식을 입력하세요 : ")
num1 = int(input_num.split()[0])
sym = input_num.split()[1]
num2 = int(input_num.split()[2])
equ = "{} {} {} = " .format(num1,sym,num2)

if sym == '+':
    print(equ, num1 + num2)
elif sym == '-':
    print(equ, num1-num2)
elif sym == '*':
    print(equ, num1*num2)
elif sym == '/':
    print(equ, "{:10.3f}".format(num1/num2))
else:
    print("기호 오류")


# In[101]:


# list : [ , , , ...] ->여러개의 자료의 집합
# list 추가 :list명.append(), list명.insert(index,추가데이터), list명.extend(추가할리스트,
# list삭제 : list명.pop(), del list명(index)
# ㅣlist 안의 요소 삭제 : list명.pop(index)
#list 값으로 삭제 : list명.remove(값)
#list의 모든 값 삭제 : list명.clear()
a_list = ["abc", "test", 123, True, 78, "Hello"]
print(a_list[0])
print(a_list[0][0]) #0번째의 0번째 자료
print(a_list[-1][2]) #마지막 자료의 2번째 자료

b_list = [1,2,3]; c_list = [4,5,6]
print(b_list + c_list) #list 연산 + 두 개의 리스트 결합, * 반복
print((b_list + c_list)*2)

a_list.append("add")  #append는 요소를 추가하는 것 ->list의 마지막에 요소 추가
print(a_list)
a_list.insert(2, "insert data")
print("a_list에 데이터 추가 ",a_list)
print("a_list의 길이 ",len(a_list))
b_list.extend(c_list) #b_list에 c_list의 값을 추가하고 기존의 데이터를 변경함
print(b_list)


# In[113]:


# list 추가 :list명.append(), list명.insert(index,추가데이터), list명.extend(추가할리스트)
# list삭제 : list명.pop(), del list명[ndex]
# ㅣlist 안의 요소 삭제 : list명.pop(index)
#list 값으로 삭제 : list명.remove(값)
#list의 모든 값 삭제 : list명.clear()

#문제)여러 개의 데이터를 입력 받아 리스트로 저장한 후
# 1. 마지막 데이터를 출력 후 삭제
# 2. 3번째 인덱스의 값을 출력
# 3. 추가로 여러 개의 데이터를 입력 받아 기존의 리스트에 추가 후 출력
# 4. 처음 입력한 데이터의 리스트 변수를 clear

mylist = input("데이터를 입력하세요 : ").split()
print("입력한 데이터를 저장한 리스트 >> ",mylist)
print("1.마지막 데이터를 출력 후 삭제")
print("삭제될 마지막 데이터 : ",mylist.pop())
print()
print("2. 3번째 인덱스의 값을 출력")
print(mylist[3])
print()
print("3. 추가로 여러 개의 데이터를 입력 받아 기존의 리스트에 추가 후 출력")
mylist.extend(input("추가 데이터를 입력하세요 : ").split())
print(mylist)
print()
print("4. 처음 입력한 데이터의 리스트 변수를 clear")
print("clear 전 리스트 : ", mylist)
mylist.clear()
print("clear 후 리스트 : ",mylist)


# In[115]:


d_list = [999, 666, "고길동", "홍길동", "YGL", "Pyt"]

# 반복문 >> for 변수 in 반복할 자료:
#              처리문

for a in d_list:
    print(a)
print()
for b in range(3):
    print("반복 : ", b)
print()
for character in "Hello":
    print(character, ' - ')


# In[117]:


# 키보드로부터 입력을 받아 리스트에 저장 후
# 입력한 자료가 숫자면 합을 구해서 출력하세요
#1. input 자료를 리스트로 추가
#2. 리스트의 자료를 하나씩 비교 (숫자인지) 반복문 실행
#3. 숫자이면 합을 구함
#4.반복문이 끝난 후 합을 출력

a_list = input("데이터를 입력하세요 :").split()
total = 0  #total을 0으로 초기화/선언
for var in a_list:
    if var.isnumeric():  #변수.isnumeric() 을 통해 자료가 숫자인지 확인
        total += int(var) #+=대입 연산자로 total에 덧셈을 함
print("total : ", total)


# In[128]:


#숫자를 입력받아 해당하는 숫자의 구구단을 출력

num = int(input("구구단을 만들 수를 입력하세요 : "))
for gugu in range(9):  #gugu 변수에 코드를 9번 반복하겠다는 반복문
    print("{} * {} = {}".format(num, gugu+1, num * (gugu+1)))
    


# 21년 9월 17일 정리
# 1. 숫자와 문자열의 다양한 기능 : " ".format()
# 2. 문자열 구성 파악하기 is함수명() ->결과는 True False
# 3. 문자열 자르기 : split() -> 문자열.split(원하는 기준(공백이면 공백을 기준으로 자른다))
# 4. 조건문 : if 조건식: , if ~ else, if ~ elif 조건식: ~ elif... else: *else는 조건식이 안붙고 그냥 else: 로 써야한다.
# 5. 날짜, 시간 함수 : datetime패키지, import 의미
# 6. list -> 관련함수 append(), extended(), pop(), insert(), remove(), clear(), del list명[인덱스]
# 7. 반복문 -> for 변수 in 반복할 것(리스트/문자열/등등)
