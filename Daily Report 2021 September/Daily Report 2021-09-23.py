#!/usr/bin/env python
# coding: utf-8

# In[13]:


#교제 158페이지 2번문제
numbers = [273, 103, 5, 32, 65, 9, 72, 800, 99]

for number in numbers:
    if number >= 100:
        print("100 이상의 수: ", number)


# In[17]:


list_of_list = [[1, 2, 3],[4, 5, 6, 7],[8, 9]
]

for lining in list_of_list:
    for line2 in lining:
        print(line2)


# In[19]:


#딕셔너리 사용
dictionary = {
    "name": "7D 건조 망고",
    "type": "당절임",
    "ingridients": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
    "origin": "필리핀"
}

#출력
print("name:", dictionary["name"])
print("type:", dictionary["type"])
print("ingridients:", dictionary["ingridients"])
print("origin:", dictionary["origin"])
print()

#값 변경
dictionary["name"] = "8D 건조 망고"
print("name:", dictionary["name"])


# In[23]:


#값 추가
dictionary["new"]="새로운 값 추가"
dictionary


# In[24]:


# 딕셔너리의 값 제거 del

del dictionary["new"]
dictionary


# In[25]:


dictionary["new"] # key error: 딕셔너리에 키 값이 없는 경우 발생


# In[26]:


dictionary = {
    "name": "7D 건조 망고",
    "type": "당절임",
    "ingridients": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
    "origin": "필리핀"
}


# In[41]:


# in 연산자로 딕셔너리 안의 키값을 찾기 - true false값으로 출력됨
key = input(" > 딕셔너리 접근 키 입력 : ")

if key in dictionary:
    print(dictionary[key])
else:
    print("존재하지 않는 키입니다")


# In[32]:


key = input(" > 딕셔너리 접근 키 입력 : ")

if key in dictionary:
    print(dictionary[key])
else:
    print("존재하지 않는 키입니다")


# In[33]:


key = input(" > 딕셔너리 접근 키 입력 : ")

if key in dictionary:
    print(dictionary[key])
else:
    print("존재하지 않는 키입니다")


# In[34]:


dictionary


# In[40]:


key = input(" > 딕셔너리 접근 키 입력 : ")

if key in dictionary:
    print(dictionary[key])
elif key in dictionary["ingridients"]:
    print("ingridients 키에 있는 값입니다")
else:
    print("존재하지 않는 키입니다")


# In[45]:


# get()함수
value1 = dictionary.get("name")
value2 = dictionary.get("asdf")

print("value1 : ", value1)
print("value2 : ", value2)
print()

if value2:
    print("none")
#위의 값은 None 이라 출력이 되지않습니다

print()
if not value2:
    print("none")


# In[46]:


# for 문장을 사용하여 dictionary의 값을 출력

for key in dictionary:
    print(key, " : ", dictionary[key])


# In[65]:


character = {
    "name": "기사",
    "level": "12",
    "items": {
        "sword": "불꽃의 검",
        "armor": "풀플레이트"
    },
    "skil": ["베기", "세게 베기", "아주 세게 베기"]
}

type(character("name"))
type(character("items"))
type(character("skill"))


# In[52]:


# 교제 173 페이지 4번 문제

character = {
    "name": "기사",
    "level": "12",
    "items": {
        "sword": "불꽃의 검",
        "armor": "풀플레이트"
    },
    "skil": ["베기", "세게 베기", "아주 세게 베기"]
}

#for 반복문을 사용합니다
for key in character:
    print("name :", character["name"])
    print("level :", character["level"])
    print("items :", character["items"])
    print("skill :", character["skil"])
    
print()
    
for key2 in character:
    if key2 type("문자열") is str
        print("name :", character["name"])
        elif key2 type([]) is list
        print("skill :", character["skil"])
        elif key2 type({}) is dict
        print("items :", character["items"])
    else:
    print("name :", character["name"])
    print("level :", character["level"])
    print("items :", character["items"])
    print("skill :", character["skil"])


# In[96]:


# 교제 173 페이지 4번 문제

character = {
    "name": "기사",
    "level": 12,
    "items": {
        "sword": "불꽃의 검",
        "armor": "풀플레이트"
    },
    "skill": ["베기", "세게 베기", "아주 세게 베기"]
}

for key in character:
    if type(character[key]) == list:
        for value in character[key]:
            print(key, " : ", value)
    elif type(character[key]) == dict:
        for key1 in character[key]:
            print(key1, " : ", character[key][key1])
    else :
         print(key, " : ", character[key])


# for key in character:
#     if type(character[key]) == list:
#         for key2 in character[key]:
#             print(key, " : ", key2)
#         print("skills :", character[key])
#     elif type(character[key]) == str:
#         for key2 in character[key]:
#             print(key, " : ", character[key][key2])
# else :
#      print(key, " : ", key2)
#      
#      
#    for key in character:
#     if type(character[key]) == list:
#         for key2 in character[key]:
#             print("skill :", character(key))
#         print("skills :", character[key])
#     elif type(character[key]) == str:
#         print("name : ", character[key])
#     elif type(character[key]) == dict:
#         print("items : ", character[key])

# In[99]:


# for 반복문과 범위를 함께 조합해서 사용합니다
for i in range(5):
    print(str(i) + " = 반복 변수")
print()

for i in range(5, 10):
    print(str(i) + " = 반복 변수")
print()

for i in range(0, 10, 2):
    print(str(i) + " = 반복 변수")
print()

for i in range(0, 10, 3):
    print(str(i) + " = 반복 변수")
print()


# In[101]:


data = int(input("범위를 입력하세요 : "))
a = list(range(data))
print(a)
for i in a(0, 10, 3):
    print(str(i) + "= 반복변수")


# In[111]:


array = [273, 4, 43, 108, 99]
#리스트에 반복문 작용
index = 1
for i in array:
    print("{}번 째 반복문 : {}".format(index, i))
    index += 1
print()

#range 함수를 이용하여 범위 지정
for i in range(len(array)):
    print("{}번 째 반복문 : {}".format(i+1, array[i])) #range는 0부터

print()

#reverse로 출력
for i in range(len(array),0,-1):  # -1씩 감소
    print("{}번 째 반복문 : {}".format(i, array[i-1]))
print()

for i in reversed(range(5)):   # 뒤에서 부터 출력
    print("{}번 째 반복문 : {}".format(i, array[i]))
    
print()

for i in reversed(range(len(array))):   # reversed를 이용 array안의 요소 갯수만큼 뒤에서 부터 출력
    print("{}번 째 반복문 : {}".format(i, array[i]))


# In[113]:


### while 조건식: 
#     처리문

while True :   #무한 반복
    print('.',end="")


# In[114]:


i = 0
while i < 30:   #조건식을 탈출하는 구문이 while 처리문 안에 존재해야 함
    print('.', end="")
    i+=1   # i = i+1


# In[115]:


array = [273, 4, 43, 108, 99]
value = 99

# del array[1]: 1인덱스의 리스트 값을 제거
while value in array: #값을 list에서 모두 지우기
    array.remove(value)
    
array


# In[118]:


import time #시간 관련 패키지함수 사용시 import 함

number = 0

print("현재 시각 : ", time.time())
target_time = time.time() + 5
while time.time() < target_time:  #특정 시간동안 프로그램 정지 또는 실행
    number += 1
print("{}초 동안 {}번 반복 처리".format(5,number))


# In[119]:


#문제) 기호 숫자 두 개를 입력 받아 계산하는 프로그램을 작성
# 기호가 # 이면 프로그램 종료
# 기호는 + - * / 는 실행, 다른 기호는 계속해서 진행

i = 0
while i < 10:
    i += 1  # i에 1을 한 번씩 증가시킴
    if i == 8:  #i가 8이면 조건 통과
        break   #i가 8이번 프로그램 종료
    elif (i%2 == 0):  #2로 나눴을때 나머지가 0이면 아래 조건으로
        continue     #프로그램 종료
    else:    #위의 조건문에 따라 2로 나눴을 때 나머지가 0이 안나오면 통과
        print(i)   #i 출력
    print("i ==", i)


# In[ ]:


if value1 != 0 and value2 !=0


# In[128]:


#문제) 기호 숫자 두 개를 입력 받아 계산하는 프로그램을 작성
# 기호가 # 이면 프로그램 종료
# 기호는 + - * / 는 실행, 다른 기호는 계속해서 진행

char_list = ['+', '-', '*', '/']
while True:
    input_char = input("> 부호를 입력하세요 (#이면 종료, +, -, *, /) >")
    if input_char == "#":
        break
    elif input_char in char_list:
        while True:
            input_num = input("두 개의 숫자를 입력하세요 > ").split()
            if len(input_num) == 2:
                break
            
        value1 = int(input_num[0])
        value2 = int(input_num[1])
        if input_char == '+':
            print("{} + {} = ".format(value1, value2), value1 + value2)
        elif input_char == '-':
            print("{} - {} = ".format(value1, value2), value1 - value2)
        elif input_char == '*':
            print("{} * {} = ".format(value1, value2), value1 * value2)
        elif input_char == '/':
            if value2 == 0:
                print("0으로 나눌 수는 없습니다.")
            else:
                print("{} / {} = ".format(value1, value2), value1 / value2)
        else:
            print("알 수 없는 오류.")

print()
print("프로그램 종료")


# In[1]:


#숫자는 무작위로 입력해도 상관없습니다. 교재 188페이지 2번 문제
key_list = ["name", "hp", "mp", "level"]
value_lsit = ["knight", 200, 30, 5]
character = {}

#간단하게는 character["name"]="knight"와 같이 반복하면 되겠다

for i in range(len(key_list)):  #len(key_list) 의 값은 4이기에 range(4)가 되고 0~3의 인덱스가 된다
    character[key_list[i]] = value_lsit[i]
character


# In[9]:


numbers = [12, 108, 99, 66, 7]

#최소값
'''
min_value = numbers[0]
for value in numbers:
    if min_value > value:
        min_value = value
min_value
'''

print(sum(numbers))
print(min(numbers))
print(max(numbers))
print(list(reversed(numbers)))
print(numbers[ : : -1])
print(sum(numbers, start=2), sum(numbers, start=0))


# In[14]:


#enumerate 함수
numbers = [12, 108, 99, 66, 7]
print(list(enumerate(numbers)))
print(numbers)

for i in range(len(numbers)):
    print("{}번째 요소 : {}".format(i, numbers[i]))
    
print("\nenumerate")
for i, value in enumerate(numbers):   #enumeratre list안의 인덱스와 값을 출력
    print("{}번째 요소 : {}".format(i,value))


# In[18]:


#문제) 키보드로부터 숫자를 임의의 갯수로 입력받아 리스트에 저장 한 후
#  print("{}번째 요소 : {}".format(i,value)") 형식으로 결과출력
#  전체 숫자의 최소값과 최대값, 합계를 구하세요
# 1.문자열 space로 분리하여 리스트화, 문자를 숫자로 변환
# 2.출력 형식대로 합계 최소값 최대값 구함

input_num = list(input("임의의 숫자를 입력하세요 : ").split())

for i,value in enumerate(input_num):  # enumerate 함수가 i에 인덱스가 value에 input_num의 리스트 요소가 들어가게 한다.
    input_num[i] = int(value)  # 인덱스와 요소를 순서대로 맞춰준다
    print("{}번째 요소 : {}".format(i,value))
    
print("최소값 : {}\t 최대값 : {}\t 합계 : {}\n".format(min(input_num), max(input_num), sum(input_num)))


# In[20]:


character = {
    "name": "기사",
    "level": 12,
    "items": {
        "sword": "불꽃의 검",
        "armor": "풀플레이트"
    },
    "skill": ["베기", "세게 베기", "아주 세게 베기"]
}

for key, element in character.items():  # . 을 찍고 탭을 누르면 쓸 수있는 함수를 알려준다
    if type(element) == list:  # 수정 전 if type(character[key]) == list:
        for value in element:   # for value in character[key]:
            print(key, " : ", value) # character[key]를 element로 대체해서 가독성이 좋아짐.
    elif type(element) == dict:
        for key1 in element:
            print(key1, " : ", element[key1])
    else :
         print(key, " : ", element)


# In[26]:


# list 내포 : list 안에 for 문장을 사용 (if, true false들도 사용가능)

array = []
for i in range(0,20,2):
    array.append(i*i)
print(array)
print("\n위의것을 간단히 함\n")
# # list 내포 : list 안에 for 문장을 사용  -> [실행문 for 변수 in 반복]
array1 = [i*i for i in range(0,20,2)] # 반복 자료에서 아이템 1개를 가져와서 i에 입력후 i*i 실행
print(array1)

print("\nif 조건식을 넣어서 만듬\n")
#조건식을 조합
array2 = [i*i for i in range(0,20,2) if i%2 == 0]  # 2로 나눠서 나머지가 0일때 -> 짝수일때 실행
print(array2)


# In[36]:


# 여러 개의 숫자를 입력받아 숫자가 6의 배수면 list_6에,
# 3의 배수면 list_3, 2의 배수면 list_2에 넣어 출력, 단 각 리스트에는 공통된 숫자는 x

array = input("여러 개의 숫자를 입력하세요 : ").split()
# 아래에서 문자를 숫자로 변환
for i, value in enumerate(array):
    array[i] = int(value)
#조건에 맞는 리스트 작성
list_2 = [i for i in array if i%6 and i%2 == 0]
# array -> i -> if 조건문 -> i
list_3 = [i for i in array if i%6 and i%3 == 0]
list_6 = [i for i in array if i%6 == 0]

print("2의 배수 : {}, 3의 배수 : {}, 6의 배수 : {}\n".format(list_2, list_3, list_6))


# # 5장 함수

# In[40]:


'''
def 함수명():
    실행문
'''
def print_func(value, n=2):   #n 부분에 초기값을 넣으면 함수 호출시에 그 자리에 아무것도 안넣어도 실행가능
    for i in range(n):
        print(value)


# In[41]:


print_func("ㅎㅇ",5) #print_func(value="ㅎㅇ", n=5)
print()
print_func("ㅎㅇ")


# In[56]:


def prt_f(value, *values, n=2): # 기본 가변 기본
    for i in range(n):
        for item in values:
            print(value, " (이)가", item, "라고 외칩니다.")
            return "종료1"
    return "종료2"


# In[46]:


prt_f("홍길동", "아버지", "형님")


# In[57]:


a = prt_f("a")
print(a)


# char_list = ['+', '-', '*', '/']
# while True:
#     input_char = input("> 부호를 입력하세요 (#이면 종료, +, -, *, /) >")
#     if input_char == "#":
#         break
#     elif input_char in char_list:
#         while True:
#             input_num = input("두 개의 숫자를 입력하세요 > ").split()
#             if len(input_num) == 2:
#                 break
#             
#         value1 = int(input_num[0])
#         value2 = int(input_num[1])
#         if input_char == '+':
#             print("{} + {} = ".format(value1, value2), value1 + value2)
#         elif input_char == '-':
#             print("{} - {} = ".format(value1, value2), value1 - value2)
#         elif input_char == '*':
#             print("{} * {} = ".format(value1, value2), value1 * value2)
#         elif input_char == '/':
#             if value2 == 0:
#                 print("0으로 나눌 수는 없습니다.")
#             else:
#                 print("{} / {} = ".format(value1, value2), value1 / value2)
#         else:
#             print("알 수 없는 오류.")
# 
# print()
# print("프로그램 종료")

# In[101]:


#문제) 기호 숫자 두 개를 입력 받아 계산하는 프로그램을 작성
# 기호가 # 이면 프로그램 종료
# 기호는 + - * / 는 실행, 다른 기호는 계속해서 진행
# 각각의 연산을 함수로 만들어서 호출 plus_func, minus_func, mul_func, div_func
def input_func():
    while True:
        input_char = input("> 부호를 입력하세요 (#이면 종료, +, -, *, /) >")
        if input_char in char_list:
            return input_char
        else:
            return input_char
def input_num():
    while True:
        input_num = input("두 개의 숫자를 입력하세요 > ").split()
        if len(input_num) == 2:
            return input_num

def plus_func(values):
    return (values[0] + values[1])

def minus_func(values):
    return (values[0] - values[1])

def mul_func(values):
    return (values[0] * values[1])

def div_func(values):
    if values[1] == 0:
        print("0으로 나눌 수는 없습니다.")
        return input_num
    else:
        return (values[0] / values[1])


# In[106]:


char_list = ['+', '-', '*', '/']

while True:
    input_char = input("> 부호를 입력하세요 (#이면 종료, +, -, *, /) >")
    if input_char == "#":
        break
    else:
        values = input_num()
        for i,value in enumerate(values):
            values[i] = int(value)
        if input_char == '+':
            result = plus_func(values)
        elif input_char == '-':
            result = minus_func(values)
        elif input_char == '*':
            result = mul_func(values)
        else:
            if values[1] == 0:
                print("0으로 나눌 수는 없습니다")
                continue
            result = div_func(values)
    print("{} {} {} = {}".format(values[0], input_char, values[1], result))
print()
print("프로그램 종료")

