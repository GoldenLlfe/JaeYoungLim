#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 팩토리얼 반복문 교제 229쪽
def factorial(n):  #함수 선언
    output=1      # output 변수 선언
    for i in range(1,n+1):   #반복문으로 숫자를 돌린다
        output *= i
    return output   #값을 리턴


# In[ ]:


counter = 0   #변수 선언
def fibonacci(n):     #피보나치 수열 함수 선언
    #어떤 피보나치 수를 구하는지 출력합니다
    print("피보나치를 구합니다.".format(n))
    global counter
    counter += 1
    #피보나치 수를 구합니다.
    if n == 1:
        return 1
    if n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
    


# In[ ]:


#팩토리얼
print("1! : ", factorial(1))
print("2! : ", factorial(2))
print("3! : ", factorial(3))


# In[ ]:


#피보나치 수열
fibonacci(10)
print("-----")
print("피보나치 10을 계산에 활욜된 덧셈 횟수는 {}번 입니다.".format(counter))


# In[ ]:


fibonacci(30)
print("-----")
print("피보나치 10을 계산에 활욜된 덧셈 횟수는 {}번 입니다.".format(counter))


# In[ ]:


#피보나치 수열 계산  함수
# 메모 변수를 만든다
dictionary = {
    1: 1,
    2: 2,
    
}

count = 0
#함수를 선언
def fibonacci2(n):
    global count
    count += 1
    if n in dictionary:   #메모 되어 있으면 메모된 값 리턴
        return dictionary[n]
    else:
        output = fibonacci2(n-1) + fibonacci2(n-2)
        dictionary[n] = output
        return output


# In[ ]:


print("피보나치(10) : ", fibonacci2(10))
print("피보나치(20) : ", fibonacci2(20))
print("피보나치(30) : ", fibonacci2(30))
print("피보나치(40) : ", fibonacci2(40))


# In[ ]:


#평탄화 함수
def flatten(data):
    #결과를 저장할 리스트
    output = []
    #리스트 각각의 item을 가져와서 작업
    for item in data:
        #데이터의 타입이 list라면 flatten함수를 호출, 아니면 output에 리시트 추가
        if type(item) == list:
            output += flatten(item)
        else:
            output.append(item)
    return output


# In[ ]:


#평탄화 함수 사용
#리스트가 포함된 리스트 작성
example = [[1,2,3],[4,[5,6]],7,[8,9]]
print("원래 리스트", example)
print("평탄화 된 리스트", flatten(example))


# In[ ]:


#튜플
[a,b] = [10, 20]
c,d= (30,40,50),(1,2,3)

print("원래 요소들",a,b,c,d)
print(type(c))
print(type(d))
a =[40]
b=[50]
c=50,60,70
d=100,200,300
print("바뀐 요소들, 하지만 튜플은 안 바뀐다",a,b,c,d)
print(type(c))


# In[ ]:


#문제) 두 수를 입력받아 큰 수에서 작은 수를 빼는 함수 작성 (num1-num2)
# 함수 정의
def minus_func(num1, num2):
    return num1 - num2

def diff_func(num1, num2):
    if num1 < num2:       #num1이 num2보다 크면 자리를 바꿔서 큰수에서 작은 수를 뺄 수 있게 해준다.
        return (num2, num1)
    else:
        return (num1, num2)


# In[ ]:


input_num = input("수 입력 : ").split()
(num1, num2) = diff_func(int(input_num[0]), int(input_num[1]))
print(minus_func(num1,num2))


# In[ ]:


#맵과 필터 함수
def power(item):
    return item * item
def under_3(item):
    return item < 3

list_input_a = [1,2,3,4,5,6,]

output_a = map(power, list_input_a)
print("맵함수 실행 결과",output_a)
print(list(output_a))
print()

output_b = filter(power, list_input_a)
print("필터함수 실행 결과",output_b)
print(list(output_b))


# In[ ]:


#람다 함수
output_a = map(lambda x:x*x,list_input_a)
print(output_a)


# 결과가 좀 이상하지만 그 위의 것과 결과가 같은 것을 볼 수 있다.

# In[ ]:


#file을 이용한 데이터 저장
#open,read or write, close
#opem(파일명, 모드)
#  w ->  기존의 파이링 존재하면 기존 데이터 삭제 후 새로 생성, 존재하지 않으면 새로 생성한다
# a -> 마지막에 추가
# r -> 읽기만 가능하고 존재하지 않으면 작동하지 않는다

file = open("test file.txt","w")  #file open
file.write("Hello python programming")    #file write
file.close()

file = open("test file.txt", "r")
do = file.read()
do
print(do)
file.close()


# In[ ]:


file = open("test file.txt", "a")
file.write("#!adding new content!@")

file = open("test file.txt", "r")
do = file.read()
print(do)
file.close()


# In[ ]:


#파일 열고 수정 닫고 응용
#맵과 필터 함수
def power(item):
    return item * item
def under_3(item):
    return item < 3

list_input_a = [1,2,3,4,5,6,]

file = open("map and file test.txt", "w")

output_a = map(power, list_input_a)
print("맵함수 실행 결과",output_a,file=file)
print(list(output_a))
print()

output_b = filter(power, list_input_a)
print("필터함수 실행 결과",output_b)
print(list(output_b))

file.close()


# In[ ]:



#랜덤하게 100명 키와 몸무게 만들기

import random

korean = list("가나다라마바사아자차카타파하김이유임철진구누두루무부수우주추쿠투푸후")

with open("random weight and height.txt", "w") as file:
    for i in range(100):
        #랜덤한 값으로 변수를 생성
        name = random.choice(korean) + random.choice(korean) +random.choice(korean)
        weight = random.randrange(50, 120)
        height = random.randrange(150, 195)
        
        #텍스르 파일을 생성후 내용을 쓰고 with 구문 덕에 자동으로 close까지 된다.
        file.write("{}, {}, {}\n".format(name, weight, height))


# In[ ]:


with open("random weight and height.txt", "r") as file:
    for line in file:
        (name, weight, height) = line.split()

file = open("random weight and height.txt", "r")
do = file.read()      
do


# In[ ]:


#제너레이터 함수 만들기 : yield 키워드를 넣으면 일반 함수가 제너레이터 함수가 된다
def test():
    print("함수호출")
    return 1

print("시작 00 : ")
next(test())  #위의 함수에서 yield가 아니라 return을 써서 함수 실행이 안된다


# In[ ]:


#제너레이터 함 만들기 : yield 키워드를 넣으면 일반 함수가 제너레이터 함수가 된다
def test():
    print("함수호출")
    yield 1
    print("함수호출2")
    yield 2

print("시작 00 : ",next(test()))
print("시작 11 : ",next(test()))
next(test())  #위의 함수에서 yield가 아니라 return을 써서 함수 실행이 안된다


# In[ ]:


#제너레이터 함수 만들기 : yield 키워드를 넣으면 일반 함수가 제너레이터 함수가 된다
def test():
    print("함수호출")
    yield 1
    print("함수호출2")
    yield 2

output = test()
print("시작 00 : ",next(output))  #넥스트 함수를 사용해야 제너레이터 함수 호출
print("시작 11 : ",next(output))
next(test())  #위의 함수에서 yield가 아니라 return을 써서 함수 실행이 안된다


# In[ ]:


#제너레이터 함수 만들기 : yield 키워드를 넣으면 일반 함수가 제너레이터 함수가 된다
def test():
    print("함수호출")
    yield 1
    print("함수호출2")
    yield 2

output = test()
print("시작 00 : ",next(output))  #넥스트 함수를 사용해야 제너레이터 함수 호출
print("시작 11 : ",next(output))
print("시작 2 : ",next(output))
next(output)  #더이상 반복 부분이 없어서 에러가 뜬다


# In[ ]:


numbers = list(range(1, 10 + 1))

print("홀수만 출력")
print(list(filter(lambda numbers: numbers%2 != 0,numbers)))
print("3이상 7미안 추출")
print(list(filter(lambda numbers:7>numbers>=3,numbers)))
print("제곱해서 50미만 추출")
print(list(filter(lambda numbers: (numbers*numbers)<50,numbers)))


# In[ ]:


with open("lambda test.txt","w") as file:
    numbers = list(range(1, 10 + 1))

    print("홀수만 출력",file=file)
    print(list(filter(lambda numbers: numbers%2 != 0,numbers)),file=file)
    print("3이상 7미안 추출",file=file)
    print(list(filter(lambda numbers:7>numbers>=3,numbers)),file=file)
    print("제곱해서 50미만 추출",file=file)
    print(list(filter(lambda numbers: (numbers*numbers)<50,numbers)),file=file)


# In[ ]:


#조건문으로 예외처리
pi = 3.14
user_input_a = input("정수 입력 > ")
#사용자의 입력이 숫자인지 호가인
if user_input_a.isdigit():
    user_input_a = int(user_input_a)
    
    print("원의 반지름 : ",user_input_a)
    print("원의 둘레 : ",user_input_a*2*pi)
    print("원의 넓이 : ", user_input_a*user_input_a*pi)
else:
    print("정수를 입력하지 않았어요.")


# In[ ]:


#조건문으로 예외처리
pi = 3.14
user_input_a = input("정수 입력 > ")
#사용자의 입력이 숫자인지 호가인
if user_input_a.isdigit():
    user_input_a = int(user_input_a)
    
    print("원의 반지름 : ",user_input_a)
    print("원의 둘레 : ",user_input_a*2*pi)
    print("원의 넓이 : ", user_input_a*user_input_a*pi)
else:
    print("정수를 입력하지 않았어요.")


# In[ ]:


#조건문으로 예외처리
pi = 3.14
#사용자의 입력이 숫자인지 호가인
try:
    user_input_a = int(input("정수 입력 > "))
    
    print("원의 반지름 : ",user_input_a)
    print("원의 둘레 : ",user_input_a*2*pi)
    print("원의 넓이 : ", user_input_a*user_input_a*pi)
except:
    print("정수를 입력하지 않았어요.")


# In[ ]:


#조건문으로 예외처리 try excepty else 구문 활용
pi = 3.14
#사용자의 입력이 숫자인지 호가인
try:
    user_input_a = int(input("정수 입력 > "))
except:
    print("정수를 입력하지 않았어요.")
else:
    print("원의 반지름 : ",user_input_a)
    print("원의 둘레 : ",user_input_a*2*pi)
    print("원의 넓이 : ", user_input_a*user_input_a*pi)
finally:
    print("무조건 실행")


# 예외 처리 구문에는 try에 except, execpt else, except finally, 그리고 finally 총 가지로 크게 구분 지을 수 있다.

# In[ ]:


def write_func(filename, text):
    try:
        file = open(filename, "w")
        return file   # 리턴이 중간에 존재 /수정 전 return
        file.write(text)   #실행이 안된다
    
    
    except:
        print("file error")
    
    finally:   #중간에 리턴이 되어도 무조건 실행
        print("try end ...")
        file.close()

file=write_func("file_close.txt","Hello file")
file.closed


# In[ ]:


#키보드에서 파일명을 입력받아 r 모드로 열어서 파일의 내용을 출력
# 없으면 w 모드로 open해서 "new file"을 파일에 저장 후 파일 close
# try except finally 활용


while True:
    filename = input("피일명 입력 : ")
    try:
        file = open(filename, "r")
        
        print(file.read())
    except Exception as err:
        print("file error : ", err) #예외 에러 메세지만 출력
        file = open(filename, "w")
        file.write("new file")
    finally:
        file.close()

    print("file.closed : ",file.closed)


# In[ ]:


list_number = [55, 273, 32, 72, 100]

try:
    #입력을 받는다
    number_input = int(input("정수 입력 : "))
    #리스트의 요소를 출력한다
    print("{}번째 요소: {}".format(number_input, list_number[number_input]))
    예외발생()
except ValueError as exception:
    #valueerror가 발생하는 경우
    print("정수를 입력해주세요")
    print(type(exception),exception)
except IndexError as exception:
    #indexerror가 발생하는 경우
    print("리스트의 인덱스를 벗어났어요")
    print(type(exception), exception)
except Exception as exception:
    #indexerror나 valueerror가 아닌 다른 예외가 발생하는 경우
    print("미리 파악하지 못한 예외가 발생했습니다.")
    print(type(exception), exception)


# In[ ]:


number = 10

if number > 0:
    raise NotImplementedError


# In[ ]:


'''
파일명을 입력받아 w모드로 open한 후
이름과 성적을 입력받아 파일에 저장
이름에 end가 입력되면 file을 close한 후
r모드로 파일을 다시 open
파일에서 자료를 읽어 list에 저장한 후 
키보드로부터 검색할 이름을 검색 한 후 있으면 이름과 성적 출력
없으면 not found error 출력
파일을 open하는 함수 file_open()작성
'''
#파일명 입력
#파일open함수 출력
#자료 입력 (이름 성적) 이름에 end 입력되면 입력 종료
#파일 close
#파일 open 함수 호출
#자료를 가져와서 변수에 저장
#검색할 이름 입력
#자료검색 : 존재하면 출력 , 업서으면 not found 출력

#파일을 여는 함수
def openfile(filename, filemode):
    file = open(filename, filemode)
    return file

#파일에 자료를 입력하는 함수
def writefile(file):
    while True:
        name = input("이름 입력: 'end'입력시 종료 >")
        if filename == 'end':
            break

        while True:
            try:
                grade=int(input("성적 입력 > "))
                break
            except:
                print("수를 입력")
        context = name + str(grade)
        file.write(context)





# In[3]:


#파일 open
def file_open(file_name, file_mode):
    file = open(file_name, file_mode)
    return file

def file_write(file): #자료 입력 [이름, 성적], 이름에 end 입력되면 입력 종료
    while True:
        name = input("이름 입력 : 'end'입력시 종료 > ")
        if name == 'end':
            break
            
        while True:
            try:
                grade = int(input("성적 입력 > "))
                break
            except:
                print("숫자를 입력하세요")
        context = name + ' , ' + str(grade) +'\n'
        file.write(context) #file에 자료 저장


# In[4]:


file_name = input("파일명 입력")
file = file_open(file_name, "a") #파일 오픈 함수 호출
file_write(file) #파ㅣㄹ에 저장하는 함수 호출
file.close() #파일 close

#파일에서 자료를 가져와서 변수에 저장
names,grades = [],[]
with file_open(file_name, "r") as file:
    for item in file:
        values = item.split(" , ")
        names.append(values[0])
        grades.append(int(values[1]))
        
#검색할 이름 입력
name = input("find name > ")
#자료를 검색 :존재하면 출력, 업서으면 not found 출력
if name in names:
    print("{} : {}".format(name,grades[names.index(name)]))
else:
    print("not found")


# In[1]:


'''
파일명을 입력받아 w모드로 open한 후
이름과 성적을 입력받아 파일에 저장
이름에 end가 입력되면 file을 close한 후
r모드로 파일을 다시 open
파일에서 자료를 읽어 list에 저장한 후 
키보드로부터 검색할 이름을 검색 한 후 있으면 이름과 성적 출력
없으면 not found error 출력
파일을 open하는 함수 file_open()작성
'''
#파일명 입력
#파일open함수 출력
#자료 입력 (이름 성적) 이름에 end 입력되면 입력 종료
#파일 close
#파일 open 함수 호출
#자료를 가져와서 변수에 저장
#검색할 이름 입력
#자료검색 : 존재하면 출력 , 업서으면 not found 출력

#파일을 여는 함수
def openfile(filename, filemode):
    file = open(filename, filemode)
    return file

#파일에 자료를 입력하는 함수
def writefile(file):
    while True:
        name = input("이름 입력: 'end'입력시 종료 >")
        if name == "end":
            break

        while True:
            try:
                grade=int(input("성적 입력 > "))
                break
            except:
                print("수를 입력")
        context = name + str(grade)
        file.write(context)


# In[5]:


#파일명 입력
#파일open함수 출력
#자료 입력 (이름 성적) 이름에 end 입력되면 입력 종료
#파일 close
#파일 open 함수 호출
#자료를 가져와서 변수에 저장
#검색할 이름 입력
#자료검색 : 존재하면 출력 , 업서으면 not found 출력

filename = input("파일명 입력")
file = openfile(filename, "a")
writefile(file)
file.close()

name,grade = [],[]
with openfile(filename, "r") as file:
    for item in file:
        values = item.split()
        name.append(values[0])
        grade.append(int(values[1]))

search_name = input("검색할 이름 : ")
                   
if search_name in name:
    print("{} : {}".format(name,grade[name.index]))
else:
    print("못 찾음")

