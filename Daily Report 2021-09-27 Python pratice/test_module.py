#!/usr/bin/env python
# coding: utf-8

# In[2]:


pi = 3.1415

print()
def moudulename():
    print("## 모듈의 __name__ 출력 ")

    print(__name__)

    print()
def numbe_input():
    output = input("숫자 입력 : ")
    return float(output)

def get_circumference(radius):
    return 2 * pi * radius

def get_circle_area(radius):
    return pi * radius * radius

