# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 21:13:09 2019

@author: Sumanta
"""

'''
The lazy factory is a concept behind the generator and the iterator.
Means they get to work and produce a single value, when you ask elese it will
be idle. 
'''

    # Itertools
    
'''
The iterator is an object that implements the iterator protocol, means
the iterable classes implement both __iter__() and __next__() in the same class.

__iter__() or iter() return self, means return the iterator object.

Every time you call next() , you ask for the next value.
An iterator knows how to compute it, when you call next() on it. 

'''

'''
There is an major dissimilarity between an iterable and iterator.
The iterable data type can use for iterator.
'''

tSet = {1, 2, 3}   # tSet is a set, which is an iterable data type 
type(tSet)          
theIterator = iter(tSet)  # theIterator is an iterator
next(theIterator)
next(theIterator)


aList = [1, 2, 3, 4]   # aSet is a list, which is an iterable data type
type(aList)          
theIterator = iter(aList) # theIterator is an iterator
next(theIterator)
next(theIterator)
next(theIterator)

# Lets create a class to get a series of numbers using iterator

class theSeries(object):
    def __init__(self , low , high):
        self.lower = low
        self.high = high
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.lower > self.high:
            raise StopIteration    # to stop the iteration
        else:
            self.lower += 1    # increament of lower value 
            return self.lower - 1  # to return the orginal lower value which passed in next()
        
numList = theSeries(1, 15)
#print(numList.__next__())
print(list(numList))

'''
The  lists [] , sets {}, dictionary {}, tuple () and strings "" 
container hold data that can be use as iterable.
'''
if 1 in [1,2,3]:
    print('List')

if 4 not in {1,2,3}:
    print('Tuple')
    
if 'a' in "apple":
    print("string")

'''
Itertools Module in python provides a lot of functions to work with iterators.
Those are "count() , cycle() ,groupby() , repeat(), map(), zip(), product() , permutations()
combinations() , combinations_with_replacement() 
'''

#map()  --> map() apply a function to each element of the list

list(map(len, ['abc', 'de', 'fghi']))  # output ===> [3, 2, 4]

#zip --> zip function takes iterable elements as input, and returns iterator

list(map(sum, zip([1, 2, 3], [4, 5, 6])))  # output ===>  [5, 7, 9]


# product()  computes the cartesian product of input iterables

from itertools import product
print( list(product([1,2,3],repeat = 2)) )
# output -->  [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

X = [[1,2,3],[3,4,5]]
print(list(product(*X)))
#output --> [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]

Y = [[1,2,3],[7,8] ,[9,5]]
print(list(product(*Y)))
#output --> [(1, 7, 9), (1, 7, 5), (1, 8, 9), (1, 8, 5), (2, 7, 9), (2, 7, 5), (2, 8, 9), (2, 8, 5), (3, 7, 9), (3, 7, 5), (3, 8, 9), (3, 8, 5)]

# Take two lists  A and B as in put and make there cartesian product AxB

A = map(int,input().split())  # The first line contains the space separated elements "1 2"
B = map(int,input().split())  # The 2nd line contains the space separated elements  "3 4"

for item in product(A,B):
    print(item,end=' ')


# select k things from a set of n things is called a permutation
# means  ordered sample without replacement 
# mathematical formula is   p( n , k )
    
'''
itertools.permutations(iterable[, r])

This tool returns successive  length permutations of elements in an iterable.

If  is not specified or is None, then  defaults to the length of the iterable, 
and all possible full length permutations are generated.

Permutations are printed in a lexicographic sorted order. So, if the input 
iterable is sorted, the permutation tuples will be produced in a sorted order.

'''
from itertools import permutations
print(permutations(['1','2','3']))
# output is <itertools.permutations object at 0x00000216CAB3AF68> 

print(list(permutations(['1','2','3'])) )
#[('1', '2', '3'), ('1', '3', '2'), ('2', '1', '3'), ('2', '3', '1'), ('3', '1', '2'), ('3', '2', '1')]

print(list(permutations(['1','2','3'],2)) )
#[('1', '2'), ('1', '3'), ('2', '1'), ('2', '3'), ('3', '1'), ('3', '2')]

print(list(permutations('abc',3)) )
#[('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]

import itertools as it
       # in mathematically we can say select 3 balls from 5-red, 3-blue, 2-green and 5-yellow balls
bills = [20, 20, 20, 10, 10, 10, 10, 10, 5, 5, 1, 1, 1, 1, 1]

list(it.permutations(bills, 3))

totalPermutation = len(list(it.permutations(bills, 3)))

# Print the permutations of the string  ABCD  in lexicographic sorted order.

from itertools import permutations
strVal = input()                    #  ABCD 2
pattern = strVal.split()[0]
r = int(strVal.split()[1])
#print(list(permutations(pattern,r)) )

for i in sorted(list(permutations(pattern,r))):
    res=''
    res= res.join(i)
    print( res )




# Choice group of k things from a set of n things is called a combination

# means  unordered samples without replacement 
# mathematical formula is   c( n , k )

'''
itertools.combinations(iterable, r) 

This tool returns the  length subsequences of elements from the input iterable.

Combinations are emitted in lexicographic sorted order. So, if the input 
iterable is sorted, the combination tuples will be produced in sorted order.

'''

from itertools import combinations

print(list(combinations('12345',2)) )
#[('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '3'), ('2', '4'), ('2', '5'), ('3', '4'), ('3', '5'), ('4', '5')]

A = [1,1,3,3,3]
print(list(combinations(A,4)) )
#[(1, 1, 3, 3), (1, 1, 3, 3), (1, 1, 3, 3), (1, 3, 3, 3), (1, 3, 3, 3)]


import itertools as it

bills = [20, 20, 20, 10, 10, 10, 10, 10, 5, 5, 1, 1, 1, 1, 1]
    # in mathematically we can say chose 3 balls from 5-red, 3-blue, 2-green and 5-yellow balls
list(it.combinations(bills, 3))

totalNoOfCombination = len(list(it.combinations(bills, 3)))

#  example 2
# Print the combinations of the string  HACK  in lexicographic sorted order.

from  itertools  import combinations

strVal = input()                    #  HACK 2
pattern = sorted(strVal.split()[0])
theR = int(strVal.split()[1])


for r in range(theR):
    r=r+1
    for i in sorted( list(combinations(pattern,r)) ):
        res=''
        res= res.join(i)
        print( res )


# combinations_with_replacement means unordered samples with replacement 
#  Choice group of k things from a set of n things with replacement 
#  mathematical formula for this   c( n+k-1 , k )

'''
itertools.combinations_with_replacement(iterable, r) 
This tool returns  length subsequences of elements from the input iterable 
allowing individual elements to be repeated more than once.

Combinations are emitted in lexicographic sorted order. So, if the input 
iterable is sorted, the combination tuples will be produced in sorted order.
'''

from itertools import combinations_with_replacement

print(list(combinations_with_replacement('12345',2)) )
#[('1', '1'), ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '2'), ('2', '3'), ('2', '4'), ('2', '5'), ('3', '3'), ('3', '4'), ('3', '5'), ('4', '4'), ('4', '5'), ('5', '5')]

A = [1,1,3,3,3]
print(list(combinations(A,2)) )
#[(1, 1), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (3, 3), (3, 3), (3, 3)]

import itertools as it

bills = [20, 20, 20, 10, 10, 10, 10, 10, 5, 5, 1, 1, 1, 1, 1]

list(it.combinations_with_replacement(bills, 3))

totalComWithRep = len(list(it.combinations_with_replacement(bills, 3)))


#example 2
# Print the combinations_with_replacement of the string  HACK  in lexicographic sorted order.

from  itertools  import combinations_with_replacement as cwr

strVal = input()                    #  HACK 2 
pattern = sorted(strVal.split()[0])
theR = int(strVal.split()[1])


for i in sorted( list(cwr(pattern,theR)) ):
    res=''
    res= res.join(i)
    print( res )


# Example 3

# Two numbers "b"  and "c" chosen in random with replacement from first 9 natural
# numbers. Find the probability that " (x^2) + bx + c > 0 " for all x belongs to R







    # Generators

'''
There is two way to create generator. 
    Using generator funciton and generator expression we can create genrator

Instade of return statement in a function if the body of a function contains 
yield, the function automatically becomes a generator function.

Generator create a stream of values. When we iterate generator fuction 
through a loop, the 'next()' gives single value every time whcih return 
by 'yield'.

When you should use yield instead of return in Python?
Return sends a specified value back to its caller whereas Yield can produce 
a sequence of values. We should use yield when we want to iterate over a 
sequence, but don’t want to store the entire sequence in memory.

 
'''

# normal function having return statement

def someFunction(a):
    result = []
    for x in a:
        result.append(x)
    return result


val = [1,3,5,7,9]

someFunction(val)

# Replace the normal fuction to genrator

def someGenerator():
    for x in range(len(val)):
        yield val[x]
        

sg =  someGenerator()
        
for i in range(5):
    print( next(sg) )

# if we will run the next() one more time it will throw 'StopIteration' error
# as there is no more value present in the variable.

#QUS--> A Python program to generate squares of 1 to 10 using generator 
  
def integers():
    i = 1
   #while True:  """Infinite sequence of integers."""
    while i < 10:
        yield i
        i = i + 1

def squares():
    for i in integers():
        yield i * i

sqr =  squares()

'''
next(sqr) # we can execute the next(), to get square value from 1 to 9 
next(sqr)
next(sqr)
next(sqr)
next(sqr)
next(sqr)
next(sqr)
next(sqr)
next(sqr)

next(sqr) # if we will run the next() one more time it will 
          # throw 'StopIteration' error. 
'''
for i in range(9):
    print( next(sqr) )


'''
Generator Expressions is nothing but the list comprehensions 
 
'''

agen = (x*x for x in range(10))
agen  # <generator object <genexpr> at 0x0000021506CB3938>

print( tuple(agen) )

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
res = [nums[i*2:(i+1)*2] for i in range(len(nums)//2)]
print(res)


'''
We can use the generator expressions as arguments to various functions.

'''
    
sumOfElements =  sum( (x for x in range(4, 10)) )
print(sumOfElements)



# Traditional way to read a file  

def catFile(fileName):
    for f in fileName:
        for line in open(f , 'r'):
            print(line)
            
# Traditional way  to grep line if the pattern present

def grepPatternLine(pattern , fileName):
    for f in fileName:
        for line in open(f , 'r'):
            if pattern in line:
                print(line)
            

# Using generator grep line if the pattern present
                
def readfiles(filename):
    for line in open(filename,'r'):
        yield line

def grepLine(pattern, lines):
    return (line for line in lines if pattern in line) # return a lines tuple 

def printlines(lines):
    i = 0
    for line in lines:
        i+=1
        print(line)
    print("\n total no of line having parttern is  " , i)

def theMain(pattern, filename):
    lines = readfiles(filename)
    patternLines = grepLine(pattern, lines)
    printlines(patternLines)
   

import os
os.chdir("E:/datascienceNml/DataScienceInPy/BasicPythonForDS/")
theMain('generator','Iterators_Generators.py')



    
















