# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 11:15:07 2018

@author: Sumanta
"""

# set is unordered , mutable  data type
# no duplicate is not allowed

#add , remove ,  union  , intersections , difference

#empty set

set1 = set()


# how to create set

setAge = {20 , 22, 25, 20, 27}
print(type(setAge))
print(setAge)


setGen = {'m', 'm' ,'f','m','f','f'}
print(setGen)

setGen.add(5)  # add element in set
print(setGen)

setGen.remove('f') # remove element 
print(setGen)

setGen.update([22,25,22,29])  
print(setGen)


unions = setAge | setGen
print(unions)


intersections =  setAge & setGen
print(intersections)

difference1_2 = setAge - setGen
print(difference1_2)


s1 = {1,2,3,4,5}
s2 = {'a' ,'b' ,'c' ,'d' ,'e'}

unions = s1 | s2
print(unions)


intersections =  s1 & s2
print(intersections)

difference1_2 = s1 - s2
print(difference1_2)







