# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 10:21:27 2018

@author: Sumanta
"""

# Tuple is an ordered immmutable  ( read only ) datatype in python
# and use open bracket to create a tuple.

age = (20, 21,23,24,26,27)
print(type(age))
print(age)

# For writing tuple for a single value, you need to include a "comma"
# even though there is a single value. Also at the end you need to write semicolon

tup = (70,);

# packing / creating a tuple 
aTup = ("Suman", 20, "Python Education")

# Unpacking the tuple
(ename, dept, profile) = aTup    # tuple unpacking
print(ename)
print(profile)
print(dept)

# Looping through tuple 
age = (20, 21,23,24,26,27)
for val in age:
    print(val)

# slicing of tuple     
ageCut = age[0:]
print(ageCut)

ageCut = age[1:3]
print(ageCut)

ageCut = age[-4:-1]
print(ageCut)

# reverse the tuple
ageCut = age[::-1]
print(ageCut)

# Advantages of tuple over list
'''
Due to immutable properties of tuple 
    1 - we can use tuple as key in Dectionary.
    2 - tuple is faster than List.
    3 - tuple is write protected.
    4 - Element from a tuple cannot be deleted, but entier tuple can be deleted 
    using 'del' keyword.
'''
                        #Exception
'''
    Python has built-in exception and User-defined exception.
When runtime error occur, Python creates an exception object.
There are plenty of built-in exceptions that are raised when 
corresponding errors occur.

Exception	              Cause of Error
==================      ======================
AssertionError	     Raised when assert statement fails.
AttributeError	     Raised when attribute assignment or reference fails.
EOFError	           Raised when the input() functions hits end-of-file condition.
FloatingPointError	Raised when a floating point operation fails.
GeneratorExit		Raise when a generator's close() method is called.
ImportError	 		Raised when the imported module is not found.
IndexError			Raised when index of a sequence is out of range.
KeyError			     Raised when a key is not found in a dictionary.
KeyboardInterrupt	Raised when the user hits interrupt key (Ctrl+c or delete).
MemoryError			Raised when an operation runs out of memory.
NameError			Raised when a variable is not found in local or global scope.
NotImplementedError	Raised by abstract methods.
OSError				Raised when system operation causes system related error.
OverflowError		Raised when result of an arithmetic operation is too large to be represented.
ReferenceError		Raised when a weak reference proxy is used to access a garbage collected referent.
RuntimeError		Raised when an error does not fall under any other category.
StopIteration		Raised by next() function to indicate that there is no further item to be returned by iterator.
SyntaxError			Raised by parser when syntax error is encountered.
IndentationError	Raised when there is incorrect indentation.
TabError			     Raised when indentation consists of inconsistent tabs and spaces.
SystemError			Raised when interpreter detects internal error.
SystemExit			Raised by sys.exit() function.
TypeError			Raised when a function or operation is applied to an object of incorrect type.
UnboundLocalError	Raised when a reference is made to a local variable in a function or method, but no value has been bound to that variable.
UnicodeError		Raised when a Unicode-related encoding or decoding error occurs.
UnicodeEncodeError	Raised when a Unicode-related error occurs during encoding.
UnicodeDecodeError	Raised when a Unicode-related error occurs during decoding.
UnicodeTranslateError	Raised when a Unicode-related error occurs during translating.
ValueError			Raised when a function gets argument of correct type but improper value.
ZeroDivisionError	Raised when second operand of division or modulo operation is zero.

'''

# TRY EXCEPT block to catch the exception

age = (20, 21,23,24,26,27)
"""
age[0] =100
print(age)   #  we will get the type error as tuple is immutable
"""
try:
    age[0] = 100
    print(age)
except TypeError as te:
    print("Data is immuatble can not be override", te)  # print the message and error type
    
print("after exceprion handling ")

lisAge = list(age)
print(type(lisAge))
lisAge[0] = 100
print(lisAge)


# Program to find the item present in the tuple and devide the item
# with the valide devision factor

theTuple = (20,21,23,24,26,27) 
       
infiVal = True
while infiVal:
    inVal = input("Enter the item to check the item present in tuple or not : ")
    try:
        if type(int(inVal)) == int:
            print("The input value is numeric that is ", inVal)
            if int(inVal) in theTuple:
                print("The value is present in the Tuple ")
    except:
        infiVal = True
    divNo = input("Enter the devision factor : ")
    try:
        if type(int(divNo)) == int:
            print("The input value is numeric that is ", divNo)
            try:
                div = int(inVal)/int(divNo)
                print(" The division of ", inVal ," / " , divNo , " is " , div)
                break
            except ZeroDivisionError as zde:
                print("please divide by number greater than zero")
    except:
        infiVal = True
            
                    # namedtuple()
                    
# namedtuple()  is a tuple where tuple has a specific name. 
# Basically, namedtuples are lightweight object types. 
        
'''
syntax ==>
        variableName = namedtuple('tupleName' , 'var1,var2,.......')
'''        

from collections import namedtuple
allCars = namedtuple('car' , 'Price, Mileage, Colour, Class')

nexon =  allCars(Price='1000000', Mileage=30, Colour = 'Grey' , Class = 'A' )   # assigning namedtuple

print(nexon)

print(nexon.Price)

    
# example -->
# Find the average mark of the students,  the input data is like spreedsheet.

'''
when below program will ask input, provide this as input

5

ID         MARKS      NAME       CLASS     
1          97         Raymond    7         
2          50         Steven     4         
3          91         Adrian     9         
4          72         Stewart    5         
5          80         Peter      6   
'''

from collections import namedtuple

# for the no of students we need to find average
N = int(input())
# take field name as input
fields = input().split()
#print(fields)

total = 0
for i in range(N):
    students = namedtuple('student',fields)
    # taking each row value as input
    field1, field2, field3,field4 = input().split()
    #print(field1, field2, field3,field4)
    student = students(field1,field2,field3,field4)
    #print(student)
    total += int(student.MARKS)
print('{:.2f}'.format(total/N))











