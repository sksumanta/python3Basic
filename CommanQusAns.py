
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:05:06 2019

@author: Sumanta
"""

# How to reverse a string

def reverseString(s):
    s1 = ''.join(reversed(s))
    return s1

print(" reverse string is " ,  reverseString('sumanta@123'))

# Best way to reverse a string is 

def revSlic(s):
    return s[::-1]

print(" reverse string is " ,  revSlic('sumanta@123'))

# How to print duplicate characters from a string

givenStr = "initializing"

uniqChar = set(list(givenStr))

for i in uniqChar:
    #print(i , givenStr.count(i))
    if givenStr.count(i) > 1:
        print(i)

# if two strings are anagram or not , means two sorted strings are equal or not
        
def check(s1, s2): 
    # the sorted strings are checked  
    if(sorted(s1)== sorted(s2)): 
        #print(sorted(s1))
        #print(sorted(s2))
        print("The strings are anagrams.")  
    else: 
        print("The strings aren't anagrams.")          
            
s1 ="listen"
s2 ="silent" 
check(s1, s2) 

# find all the permutations of a given string?

def permute(a, r, l=0): 
    if l == r: 
        print( "".join(str(a) ) )
    else: 
        for i in range(l, r + 1): 
            a[l], a[i] = a[i], a[l] 
            print("first i, l, r, a[i], a[l]",i, l, r, a[i], a[l])
            permute(a, r, l + 1) 
            print("sec i, l, r, a[i], a[l]",i, l, r, a[i], a[l])
            a[l], a[i] = a[i], a[l] # backtrack 
            
# Driver program to test the above function 
string = "ABC"

n = len(string) 
a = list(string) 
permute(a, n-1)  
                       # or
                        
from itertools import permutations
print(list(permutations(a)) )

# check if a string contains only digits?

   
                    # Iterators
# What is the difference between iterable and iterator?
'''
Iterable is an object, which one can iterate over.
When Iterables are passed to iter() method they generates Iterator.

Iterator is an object, which iterate over an iterable object 
using __next__() method to returns the next item of the object.

To make Iterator the class of an object needs either a method __iter__, 
which returns an iterator, or a __getitem__ method with sequential 
indexes starting with 0.

'''

# Give some examples of iterables 
'''
List , tuple , String are the iterables.
 
'''

# How for loop act as an iterator
'''
For loop calls  iter() on the iterable object. If call is sucess then
iter() return an iterator object to __next__()  to access the element
of the object, one at a time. when __next__() falls it raise a 
" StopIteration  " exception and for loop will terminate.

'''

# Check object is iterable or not

def chkIter(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

for ele in [8.9, 34, 'spyder', [3,4,5], (2,4,6), {'a','g','t'}, {'a':7}]:
    print(ele , " is iterable " , chkIter(ele))

# how to convert a list into iterator and access each element from that?

givenList = [4, 5, 6, 7, 8]

iterObj = iter(givenList)  # it will create the iterator object

while True:
    try:
        print(iterObj.__next__())
    except StopIteration:
        break

# Remove empty strings from list of strings

givenList =  ['', 'egg', '', 'is', 'best', '']

iterObj = iter(givenList)

newList=[]
while True:
    try:
        x=iterObj.__next__()
        if '' == x :
            pass
        else:
            print()
            newList.append(x)
    except StopIteration:
        break
print(newList)

        # using filter() method Remove empty strings from list of strings

newList = list(filter(None , givenList))
print(newList)


# Remove all digits from a list of strings

givenList = ['alice1', 'bob2', 'cara3', '4geeks', '3for' ]

def getStr(lists):
    newList = ["".join(x for x in ele if (not x.isdigit()) ) for ele in givenList ]
    return newList

print(getStr(givenList))
    
            # using regular expression  re 
'''
re.sub(Source_pattern,repl_pattern,source_string) used to replace substrings. 
'''
import re
def getStr(lists):
    pattern = '[0-9]'
    newList = [re.sub(pattern,'',i) for i in givenList]
    return newList

print(getStr(givenList))


#  Remove last character in list of strings 

givenList = ['Manjeets', 'Akashs', 'Akshats', 'Nikhils'] 

newList = list(map(lambda x : x[:-1] , givenList) )

print(newList)            
            
            
# remove i’th character from string

givenStr = "Demonstrating"

def removeIth(no):
    newStr= "".join(givenStr[:no-1])+givenStr[no:]
    return newStr

print(removeIth(4))

# Remove tuple from list of tuples if first element of tuple not containing any character

givenList = [(', ', 12), ('...', 55),
        ('-egg', 115), ('spyder', 11)]

import re

newList = []
def chkTuple(ele):
    if re.search(r'\w',ele[0]):
        newList.append(ele)
    return newList

for e in givenList:
    chkTuple(e)
    
print(newList)

    
# Remove consecutive duplicates from list

import itertools as it

givenList =  [1, 4, 4, 4, 5, 6, 7, 9, 2, 2, 2, 4, 3, 3, 9]

print([ ele[0] for ele in it.groupby(givenList) ])
    
 
# Find frequency of each character in String

givenStr = "initializing"

uniqChar = set(list(givenStr))
for i in uniqChar:
    print(i , givenStr.count(i))
    
# Remove the characters repetade more than once in the string 

givenStr = "initializing"

newstr=''
for i in givenStr:
    if 1 == givenStr.count(i):
        newstr = newstr+i
print(newstr)

# Return maximum occurring character in an input string

givenStr = "individual_life"

newList=[]
counts=[]
uniqStr = set(list(givenStr))
for i in uniqStr:
    counts.append(givenStr.count(i))
    newList.append([i , givenStr.count(i)])


maxCount = sorted( counts, reverse=True )[0]
newDict=dict(newList)

for ele in newDict:
    if newDict[ele] == maxCount:
        print(ele)
    
# Count the no of vlowels present in a string with there position.

givenStr = "individual_life"

vowel = ['a','e','i','o','u']

add=0

for ind in range(len(givenStr)):
    for vow in vowel:
        if givenStr[ind] == vow:
            print(vow ," postion is ", ind+1)    
    
for vow in vowel:
    add = add + givenStr.count(vow)
print("total no of vowels " , add)


# count no of vlowels present for all possible non repeted substrings of given string.

'''
ex ==>  str = “abc”
Substrings of “abc” are = {“a”, “b”, “c”, “ab”, “bc, “abc”}
count of vowels is 3 (‘a’ occurred 3 times)

Hints -->  no of possible substring = ( n * (n+1) ) / 2 where n is no of chars
but remove the repeted substring

'''
 
givenStr = "danceing"

vowel = ['a','e','i','o','u']

def subStr():
    strList=[]    
    n= len(givenStr)
    for i in range(0,n):
        if i == 0:
            strList.append(givenStr)
        else:
            strList.append(givenStr[:n-i])
            strList.append(givenStr[i])
            #print(i ,"          " ,   strList)
    return set(strList)

#print(subStr())
            
def countVowel():
    newList=subStr()
    add = 0
    for i in vowel:
        for ele in newList:
            print('count of ', i ,' in ',ele ,' is ', ele.count(i))
            add = add + ele.count(i)
    return add

print(countVowel())
    
# count duplicates tuples in a list

givenList = [('a', 'b'), ('a', 'e'), ('b', 'x'), ('b', 'x'), 
                                             ('a', 'e'), ('b', 'x'),]
#givenList =  [(0, 5), (6, 9), (0, 8)]

newList=[]
for ele in givenList:
     add=0
     for itm in givenList:
         if ele[0]==itm[0] and ele[1] == itm[1]:
             add =  add+1
     #print(ele , add)
     newList.append((ele,add))

#print(set(newList))

if len(set([secInd[1] for secInd in sorted(newList)])) > 1:
    for itm in set(newList):
        print(itm[0], itm[1])
else:
    print("no duplicate present")

    # Best way to do the above example is use    get() of dictionary
    
newDict ={}
for ele in givenList:
    newDict[ele] = newDict.get(ele,0)+1
    
print(newDict)    

# Remove all duplicates and permutations in nested list

givenList =  [[-11, 0, 11], [-11, 11, 0], [-11, 0, 11], 
                              [-11, 2, -11], [-11, 2, -11], [-11, -11, 2]]

import itertools as it
def removeDuplicate():
    newList=[]
    for ele in it.groupby(givenList):
        #print(sorted(ele[0]))
        newList.append(tuple( sorted(ele[0]) ) )
    #print(newList)
    return newList
   
selectOne = set( removeDuplicate() ) 
print(selectOne)

            # using comprehension
            
selectOne =  set( tuple( sorted(ele[0]) )  for ele in it.groupby(givenList) )
        
print(selectOne)    
    
    
# Natural Language Processing (NLP)

'''
Using Natural Language Processing (NLP) computer process and analyze large 
amounts of natural language data. 
NLP perform Tokenization process to perform the activity.

Tokenization is the process of tokenizing or splitting paragraph into
sentence, sentence into words.

To perform Tokenization we need below steps 

    Text into sentences using tokenization
    Sentences into words using tokenization
    Sentences using regular expressions tokenization ( wordPunctTokenization,
                        whiteSpaceTokenization)
'''

# How  sent_tokenize works

'''
The sent_tokenize() function uses an instance of PunktSentenceTokenizer 
from the nltk.tokenize.punkt module. sent_tokenize() function knows
begining and ending of sentence and punctuation.
 
'''

from nltk.tokenize import sent_tokenize 
  
text = "Hello everyone. Welcome to spyder. You are studying NLP python"
res = sent_tokenize(text)
print(res)


# How word_tokenize works?
'''
word_tokenize() function is a wrapper function that calls tokenize() on 
an instance of the TreebankWordTokenizer class and separate the words 
using punctuation and spaces.

'''

from nltk.tokenize import word_tokenize

text = "Hello everyone. Welcome to spyder. You are studying NLP python"

res = word_tokenize(text)

print(res)


# what is the difference between TreebankWordTokenizer ,   
# and WordPunctTokenizer and RegexpTokenizer and regexp_tokenize 
 
'''
 TreebankWordTokenizer - It doen’t discard the punctuation allow user to 
                                 decide what to do with the punctuations.
 WordPunctTokenizer – It seperates the punctuation from the words.
 RegexpTokenizer and regexp_tokenize  - It seperates the words using 
                                             Regular Expression
'''
from nltk.tokenize import TreebankWordTokenizer 
  
tokenizer = TreebankWordTokenizer() 

text = "Hello everyone! Welcome to spyder. You are studying NLP python."

res =tokenizer.tokenize(text) 

print(res)    


from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()

text = "Hello everyone! Welcome to spyder. You are studying NLP python."

res = tokenizer.tokenize(text)

print(res)

 
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[\w']+") 

text = "Let's see how it's working."

res = tokenizer.tokenize(text)

print(res)


from nltk.tokenize import regexp_tokenize 
  
text = "Let's see how it's working."

res = regexp_tokenize(text, "[\w']+") 

print(res)


# replace a word with asterisks in a sentence

text = "Spyder is a IDE for codeing. People who love codeing can use \
        this IDE to execute their valuables/ideas using programing language"

def censor(text, word): 
    wordList =  text.split()
    stars = '*'*len(word)
    ind = 0
    for w in wordList:
        if w == word:
            wordList[ind] = stars
        #print(wordList,end = "\n\n" )
        result =' '.join(wordList) 
        ind = ind + 1
    return result
        
print(censor(text,'IDE'))


                        # Mathematics Qus and Ans

# QUS
'''
Consider two points, P (Px,Py) and Q (Qx,Qy). We consider the inversion or 
point reflection, R (Rx,Ry), of point P across point  Q to be a 180 degree 
rotation of  point P around Q. Given  N sets of points  P and Q , find R 
for each pair of points and print two space-separated integers denoting 
the respective values of  Rx and Ry on a new line.

Input Format

The first line contains an integer, N , denoting the number of sets of points. 
Each of the N subsequent lines contains four space-separated integers describing 
the respective values of ,Px ,Py ,Qx,Qy and defining points  P=(Px,Py)and Q=(Qx,Qy).

Constraints
    1 < = N <=15 
    -100 <= Px ,Py ,Qx,Qy <=100

Output Format 

For each pair of points  P and Q, print the corresponding respective values 
of  Rx and Ry as two space-separated integers on a new line.

Sample Input

2
0 0 1 1

Sample Output
2 2

'''

# ANS

'''
For each P (Px,Py) and R (Rx,Ry) the Q is the midpoint 
so Q (Qx,Qy)= ( Rx+Px/2 , Ry+Py/2)

Q = R+p/2 => R = 2Q-P => R = (2Qx-Px , 2Qy-Py)

'''

import os
import sys
#
# Complete the findPoint function below.
#
def findPoint(px, py, qx, qy):
    #
    # Write your code here.
    #
    return [(qx*2)-px, (qy*2)-py]


if __name__ == '__main__':
    
    n = int(input())        # take the input 2 

    for n_itr in range(n):
        pxPyQxQy = input().split() # take 4 nos with space separated two times

        px = int(pxPyQxQy[0])

        py = int(pxPyQxQy[1])

        qx = int(pxPyQxQy[2])

        qy = int(pxPyQxQy[3])

        result = findPoint(px, py, qx, qy)
        
        print(result)


# QUS
'''
pigeonhole principle == > if N items are put into M containers, with N > M, 
then at least one container must contain more than one item.
'''

'''
Case 1 : A pair of socks are present, hence exactly 2 draws for the socks to match. 
Case 2 : 2 pair of socks are present in the drawer. The first and the second 
draw might result in 2 socks of different color. The 3rd sock picked will 
definitely match one of previously picked socks. Hence, 3.

Sample Input

2 -  two socks
1 -  no of pairs (first time n =1)
2 -  no of pairs (second time n  = 2)
Sample Output

2
3
'''

def maximumDraws(n):
    return n+1

t = int(input())

for t_itr in range(t):
    n = int(input())
    result = maximumDraws(n)

print(result)



#QUS

"""
At the annual meeting of Board of Directors of Acme Inc, every one starts 
shaking hands with everyone else in the room. Given the fact that any two 
persons shake hand exactly once, Can you tell the total count of handshakes?

Sample Input

2  - no of inputs
1  no of Directors (first time n =1)
2  no of Directors (second time n  = 2)

Sample Output

0
1

It is a combination problem with chooseing 2 person form N => c(N,2)
"""


def maximumDraws(n):
    return (n * (n-1))//2   # we can not use itertools.combination as 'n' is an interger

t = int(input())

for t_itr in range(t):
    n = int(input())
    result = maximumDraws(n)

print(result)


            # or we can use factorial from math module

import math

def nCr(n,r):
    f = math.factorial
    if r > n:
        r=0
    return f(n) / f(r) / f(n-r)

t = int(input())

for t_itr in range(t):
    n = int(input())
    if n < 2:
        result = 0
    else:
        result = nCr(n,2)
    print(result)
    
#QUS

#  A triangle of height 'h', base 'b', having an area of at least 'a'.
# Find the hight of the triangle if  area 'a' and base 'b' is given.

'''    
Sample Input 0

2 2  ----- area 'a' and base 'b' with space separated 

Sample Output 0

2    

Hints ==> Area of triangle is    a = (b * h) / 2

'''


import math
def lowestTriangle(base, area):
    # Complete this function
    res = math.ceil((2*area)/base)
    return res

base, area = input().strip().split(' ')
height = lowestTriangle(base, area)
print(height)


#QUS
'''
Luke is daydreaming in Math class. He has a sheet of graph paper with 'n' rows 
and 'm' columns as army base.  He wants to draw point with a red dot on top of 
its border fence, then it's considered to be supplied.

Explanation 

Luke has four bases in a 2 x 2 grid. If he drops a single package where the walls 
of all four bases intersect, then those four cells can access the package.
    #   #   ------> # is the army base    
      o     ------> o is the draw point
    #   #
'''


def gameWithCells(n, m):
    #
    # Write your code here.
    #
    #print("n%2" , n%2)
    #print("m%2" , m%2)
    return (n+n%2)*(m+m%2)//4

nm = input().split()

n = int(nm[0])

m = int(nm[1])

result = gameWithCells(n, m)

print(result)

#Qus

'''
Find number of unique prime factors in the inclusive range of any number.

Sample Input

6  ----------- no of input 

1
2
3
500
5000
10000000000

Sample Output

0
1
1
4
5
10
'''


def primeCount(n):
    #
    # Write your code here.
    #
    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 
                                      59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    res=1
    c=0
    if n == 1:
        return c
    else:
        for p in prime:
            res=res*p
            if res <= n:
                c = c+1
        return c


q = int(input())

for q_itr in range(q):
    n = int(input())
    result = primeCount(n)
    print("no of prime factors ",result)






