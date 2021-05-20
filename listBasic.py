# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:40:55 2018

@author: Sumanta
"""

# search can be start from the begetive inex in case of list.
# what is the benifit
    # head and tail implementation can be done using negetive index
    # in case of stock market we and capturing every 5 minite time data as row data
    # to find end of day data then use can use negetive index(-1)
# capute the rediaction data wrt the time to forcast the energy generation.
    # timestamp ,   rediation/m2 , panel area , panel/inveter efficiency
    # energy curve is incremental  
    # t1  ,  red1 , panAre1 , efic1
    # t2  ,  red2 , panAre2 , efic2
    # t3  ,  red3 , panAre3 , efic3

# basic methods in python list 
    # append  ,  insert , 
#################  append ###################
# append add a single item / value at the end of the list
    
numList = [1,2,3,4]
numList.append(5)
print(numList)

############## insert ################
# insert add the value at the perticular index

numList.insert(2,9)
print(numList)


# use this list functions in user definded functions
"""
def addItem(item , ind):
    print(" using this function for appending and inserting the item into the list)
    ind=none
    lis.append(item)

"""

############# extend ###########
# extend add the multiple value to the list
numList.extend([15,16,17])
print(numList)

#  Differenct beteween append and extend
# extend can through type error then you try to exend by passing indivisual value 
# so it should be a list.

lisX = [25,27,28]
numList.extend(lisX)
print(numList)


numList.append(lisX)
print(numList)


# extend vs. append
a = [1, 2]
b = [3, 4]
c = [5, 6]
b.append(a)
c.extend(a)
print(b)
print(c)

# concatination ( for concatination we can use  + ) adding two list

newList = numList+[31,32,33]
print(newList)


#delete using del --- 

#del newList   # delete entire list
#print(newList)  # out put --- name 'newList' is not defined

#clear   it will create an empty list

#newList.clear()
#print(newList)  

#pop -- delete the last element for the list

newList.pop()
print(newList)

print("hi")

newList.pop(3)  # delete the value from particutlar index using pop
print(newList)

lisY = ["HELLO" , [1,2,3]]
print(lisY[0][3])  # output  L
print(lisY[1][2])  #output 3

import math
n , lis = 5 , []

for power in range(5):
    lis.append(math.pow(2,power))
    
print(lis)

"""  
def power(no):
    lis=[]
    for n in range(int(no)):
        p=n+1
        print(p)
        lis.append(math.trunc(math.pow(2,p)))
    return lis

num = input("enter the no to find the power of 2 :" )

print("the power result list is " , power(num))


"""
def power(no,val):
    lis=[]
    for n in range(int(no)):
        p=n+1
        print(n)
        lis.append(math.pow(val,p))   # or we can use num**val
    return lis

num = input("enter the no to find the power of 2 :" )

val = input("enter the value for which power need to be counted :")

print("the power result list is " , power(num,int(val)))



# Remove the  words less than 3 chars from the list

import re
sentence =  ['I', 'am' ,'in', 'a', 'bank', 'and', 'saw','brown' ,'fox' ,'jumps', 'over', 'the', 'lazy', 'dog']
sentence= [  re.sub(r'\b\w{1,2}\b', '',i) for i in sentence ]
   
print(sentence)

# or we can do below

print([w for w in sentence if len(w)>2])



###############
a=[[]]*3
a[1].append(5)
print(a)



