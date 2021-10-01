#!/usr/bin/env python
# coding: utf-8

# In[1]:


students = [
    { "name": "윤인성", "korean": 87, "math": 98, "english": 88, "science": 95 },
    { "name": "연하진", "korean": 92, "math": 98, "english": 96, "science": 98 },
    { "name": "구지연", "korean": 76, "math": 96, "english": 94, "science": 90 },
    { "name": "나선주", "korean": 98, "math": 92, "english": 96, "science": 92 },
    { "name": "윤아린", "korean": 95, "math": 98, "english": 98, "science": 98 },
    { "name": "윤명월", "korean": 64, "math": 88, "english": 92, "science": 92 }
]


# In[23]:


#객체를 만드는 함수
#딕셔너리를 리턴하는 함수를 선언

def create_student(name, korean, math, english, science):
    return {
        "name": name,
        "korean": korean,
        "math": math,
        "english":english,
        "science":science
    }

students = [
    create_student("윤인성", 87, 98, 88, 95),
    create_student("연하진", 92, 98, 96, 98),
    create_student("구지연", 76, 96, 94, 90),
    create_student("나선주", 98, 92, 96, 92),
    create_student("윤아린", 95, 98, 98, 98),
    create_student("윤명월", 64, 88, 92, 92)
]

#학생 점수의 합을 구하는 함수
def student_get_sum(students):
    return student["korean"]+student["math"]+            student["english"]+student["science"]
#평균을 구하는 함수
def student_get_avg(students):
    return student_get_sum(students) / 4

#출력하는 함수
def student_to_string(students):
    return "{}: \t{} \t{}".format(student["name"], student_get_sum(students), student_get_avg(students))


# In[24]:


#class 와 object

print("이름", "총점", "평균", sep="\t")
for student in students:
    print(student_to_string(students))


# In[5]:


#class로 생성하여 object를 관리
class Student:   #class 클래스이름():
    def __init__(self,name,korean,math,english,science):
        self.name = name
        self.korean = korean
        self.math = math
        self.english = english
        self.science = science
        
    #Student 클래스 a_class.인스턴스, 데이터


    #학생 점수의 합을 구하는 함수
    def get_sum(self):
        return self.korean + self.math + self.english + self.science
    #평균을 구하는 함수
    def get_avg(self):
        return self.get_sum() / 4

    #출력하는 함수
    def to_string(self):
        return "{}: \t{} \t{}".format(self.name, self.get_sum(), self.get_avg())

    def __str__(self):
        return "{}: {}\t {}".format(self.name,                                   self.get_sum(),                                   self.get_avg())
students = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]

for student in students:
    print(student.to_string())


# In[34]:


students = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]
print("students의 데이터 갯수 : ", len(students))


# In[67]:


students = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]

for student in students:
    print(str(student.__str__()))


# In[2]:


#isinstance(인스턴스,클래스)

class Human:
    def __init__(self):
        pass

class Student(Human):
    def __init__(self):
        pass

student = Student()
#인스턴스 확인
print("isinstance(student, Human): ",isinstance(student,Human))

human = Human()


# In[71]:


def __eq__(self, value): 
    return self.get_sum() == value.get_sum()
def __ne__(self, value): 
    return self.get_sum() != value.get_sum()
def __gt__(self, value):
    return self.get_sum() > value.get_sum()
def __ge__(self, value):
    return self.get_sum() >= value.get_sum()
def __lt__(self, value):
    return self.get_sum() < value.get_sum()
def __le____(self, value):
    return self.get_sum() <= value.get_sum()


# In[ ]:


students = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]


# In[6]:


class Student:
    count = 0

    def __init__(self, name, korean, math, english, science):
        # 인스턴스 변수 초기화
        self.name = name
        self.korean = korean
        self.math = math
        self.english = english
        self.science = science
        
        # 클래스 변수 설정 클래스명.변수
        Student.count += 1
        print("{}번째 학생이 생성되었습니다.".format(Student.count))
    

    def get_sum(self):
        return self.korean + self.math + self.english + self.science
    #평균을 구하는 함수
    def get_avg(self):
        return self.get_sum() / 4

    #출력하는 함수
    def to_string(self):
        return "{}: \t{} \t{}".format(self.name, self.get_sum(), self.get_avg())

    def __str__(self):
        return "{}: {}\t {}".format(self.name,                                   self.get_sum(),                                   self.get_avg())
# 학생 리스트를 선언합니다.
students = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]
# 출력합니다.
print()
print("현재 생성된 총 학생 수는 {}명입니다.".format(Student.count))


# In[8]:


class Student:
    count = 0
    students = []
    #클래스 함수 데코레이터 
    @classmethod
    def print(cls):
        print("-----학생 몰록-----")
        print("이름\t총점\t평균")
        for student in cls.students:   #Student.student라고 해도 괜찮지만 매개변수로 받은 cls를 활용함
            print(str(student))
        print("------ ------ -----")
    
    
    def __init__(self, name, korean, math, english, science):
        # 인스턴스 변수 초기화
        self.name = name
        self.korean = korean
        self.math = math
        self.english = english
        self.science = science
        
        # 클래스 변수 설정 클래스명.변수
        Student.count += 1
        Student.students.append(self)
        
    def get_sum(self):
        return self.korean + self.math + self.english + self.science
    #평균을 구하는 함수
    def get_avg(self):
        return self.get_sum() / 4

    def __str__(self):
        return "{}: {}\t {}".format(self.name,                                   self.get_sum(),                                   self.get_avg())
# 학생 리스트를 선언합니다.
students = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]    
Student.print()


# In[9]:


#프라이빗 변수를 활용한 원의 둘레와 넓이를 구하는 프로그램

import math

class Circle:
    def __init__(self, radius):
        self.__radius = radius
    def get_circumference(self):
        return 2 * math.pi * self.__radius
    def get_area(self):
        return math.pi * (self.__radius ** 2)
    
# 원의 둘레와 넓이를 구합니다.
circle = Circle(10)
print("# 원의 둘레와 넓이를 구합니다.")
print("원의 둘레:", circle.get_circumference())
print("원의 넓이:", circle.get_area())
print()

# __radius에 접근합니다.
print("# __radius에 접근합니다.")
print(circle.__radius)   #__radius는 프라이빗 변수라 외부에서 접근이 불가해서 에러가 발생


# In[10]:


#프라이빗 변수를 활용한 원의 둘레와 넓이를 구하는 프로그램

import math

class Circle:
    def __init__(self, radius):
        self.__radius = radius
    def get_circumference(self):
        return 2 * math.pi * self.__radius
    def get_area(self):
        return math.pi * (self.__radius ** 2)
    
    def get_radius(self):   #getter 그냥 프라이빗 변수를 가진 함수, 변수 그 자체인 함수
        return self.__radius
    def set_radius(self):   #setter 프라이빗 변수를 가진 함수를 다른 변수에 넣음
        self.__radius = value
    
# 원의 둘레와 넓이를 구합니다.
circle = Circle(10)
print("# 원의 둘레와 넓이를 구합니다.")
print("원의 둘레:", circle.get_circumference())
print("원의 넓이:", circle.get_area())
print()

# 간접적으로 __radius에 접근합니다.
print("# __radius에 접근합니다.")
print(circle.get_radius())   #__radius는 프라이빗 변수라 외부에서 접근이 불가해서 에러가 발생


# In[ ]:


class Circle:
    def __init__(self, radius):
        self.__radius = radius
    def get_circumference(self):
        return 2 * math.pi * self.__radius
    def get_area(self):
        return math.pi * (self.__radius ** 2)
    
    @property   #데코레이터를 활용한 게터와 세터
    def radius(self):   #그냥 프라이빗 변수를 가진 함수
        return self.__radius
    @radius.setter
    def set_radius(self):   #프라이빗 변수를 가진 함수를 다른 변수에 넣음
        self.__radius = value


# In[11]:


#inheritance 상속
#부모 클래스 선언
class Parent:
    def __init__(self):
        self.value = "테스트"
        print("Parent 클래스의 __init()__ 메소드가 호출됨")
    def test(self):
        print("Parent 클래스의 test()메소드 입니다")

#자식 클래스 선언
class Child(Parent):
    def __init__(self):
        Parent.__init__(self)
        print("Child 클래스의 __init()__메소드가 호출됨")

        
        
#자식 클래스의 인스턴스를 생성하고 부모의 메서드를 호출한다.
child=Child()
child.test
print(child.value)

