# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:02:51 2018

@author: Sumanta
"""

# why numpy module 
"""
 NumPy is the fundamental package for scientific computing with Python.
 It provides a high-performance multidimensional array object 
 ( for example 
         [
            [1,2,3]
            [5,6,7]  ----- this is two dimentional array having row and column
            [8,9,4]
         ] simlarly we can go for  (N-dimention) more dimentional array in python ) , 
 and tools for working with these arrays.
"""

# how to declare an array

import numpy as np
fstArray= np.array([1,3,5])   # -------------> it is an one dimentional array
print("The containt of the one dimentional array is ", fstArray)

multiArray = np.array([(1,3,5),(7,8,9)])
print("The containt of the two dimentional array is ",multiArray )

# we can create an array by using   arange() , linspace()
"""
syntax of arange() method is
    arange([start, ]stop, [step, ]dtype=None)
Syntax of linspace() method is
    linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
            where    start ---> starting point
                     stop  ---> end point
                     num   ---> number of points/ elements between start and stop
                                         (the point include start and stop)
                     endpoint = True ---> we will inclue the stop point in array
                                             else it will exclude stop from array
                     retstep = False ---> if false the output of array will not 
                                     show the space/difference between each point/element.
    
"""

import numpy as np

singDiArr = np.arange(1,15,2) 
print("the single dimentional array created by arange " , singDiArr)

import numpy as np
linspaceArr = np.linspace(2,3,5)
print("The array created by linspace ",linspaceArr)

linArr = np.linspace(2,4,5,endpoint=False)
print("The array created by linspace with endpoint false ", linArr)

linArrRet = np.linspace(2,4,5,retstep=True)
print("The array created by linspace with retstep true ",linArrRet)

# why array if there is list
"""
The numpy array is more faster than list. Alos numpy array consume less memory than list

""" 

# example to check the memory occupied by  list and array

import numpy as np
import sys

theList = range(1000)

theArray = np.arange(1000)

memList = sys.getsizeof(5)*len(theList)
print("memory occupy by list is ", memList)

memArray = theArray.size * theArray.itemsize
print("memory occupy by array is ", memArray)


# example to check array is faster than list and alos convinient ( easy to do operation)

import numpy as np
import time

sizes = 10000000

lis1 = range(sizes)
lis2 = range(sizes)

startTime = time.time()

add2ListInLoop = [(l1,l2) for l1 , l2 in zip(lis1 , lis2) ]

endTime =  time.time()

listTimeTaken = endTime-startTime

print("Time taken by the list is in sec " ,listTimeTaken )

arr1 = np.arange(sizes)
arr2 = np.arange(sizes)

ArrStartTime = time.time()

add2Array = arr1  + arr2

arrEndTime = time.time()

arrTimeTaken = arrEndTime - ArrStartTime

print("Time taken by the array is in sec  " ,arrTimeTaken )


# how to get the dimention of the array
"""
    To get the dimention of the array we can use  ndim  
"""
import numpy as np

theArr = np.array([3,5,7])

arrDim = theArr.ndim

print("the array dimention is  ", arrDim)

theArr = np.array([
        [3,5,7],
        [2,4,8]
        ] )

arrDim = theArr.ndim

print("the array dimention is  ", arrDim)

# How to get the data type of an array
"""
To get the data type of an array we can use  dtype  attribute
"""

import numpy as np

theArr = np.array([3,5,7])

arrDtype = theArr.dtype

print("the array data type is  ", arrDtype)

theArr = np.array([
        [3,5,7],
        [2,4,8]
        ] )

arrDtype = theArr.dtype

print("the array dimention is  ", arrDtype)

## How to create user definded data type to declare an array

import numpy as np

dTyp = np.dtype([('name', 'S10'),('age', int)]) 

theArr = np.array([("raju",21),("anil",25),("ravi", 17), ("amar",27)], dtype = dTyp) 

print("The array with user defined data type is  ", theArr)

# ('grades', np.float64, (2,) ) ---> float  ,  and i4, i8 --> integer 


# Total size of the array or no of element present in an array we can use   size  attribute
"""
To get the no of element present in an array we can use   size  attribute
 
"""

import numpy as np

theArr = np.array([
         [2,4,6],
         [3,5,7],
         [9,7,5]
        ])

otherArr =  np.array([
         [2,4],
         [3,5],
         [9,7]
        ])

arrSize = theArr.size

print("The size of the array is ", arrSize)


otherArrSize = otherArr.size

print("The size of the array is ", otherArrSize)

# How to check each element size in an array 
# How to check the total size of array in bytes

"""
To get each element size in an array  use  itemsize  attribute 
To get total array size in byte we can use (array size * itemsize) 
"""
import numpy as np

theArr = np.array([
        [2,4,6],
        [3,5,7]
        ])

elementSize = theArr.itemsize

print(" The element size in array ", elementSize)

totalArrSize = theArr.size * theArr.itemsize

print("Total array size in byte ", totalArrSize) 


# How to get the shape of the array  -----> use  shape  attribut 

import numpy as np

theArr = np.array([
         [2,4,6],
         [3,5,7],
         [9,7,5]
        ])

otherArr =  np.array([
         [2,4],
         [3,5],
         [9,7]
        ])

arrShape = theArr.shape

print("The shape of the array is ", arrShape)


otherArrShape = otherArr.shape

print("The shape of the array is ", otherArrShape)

theArr =  np.array([
         [2,4,6],
         [3,5],
         [9,7,5]
        ])

arrShape = theArr.shape

print("The shape of the array is ", arrShape)


# How to reshape the array
"""
reshape means converting row to column and column to row, 
That we can do by using reshape() 

"""

import numpy as np
theArr =  np.array([
         [2,4,6,8],
         [3,5,9,7]
        ])

arrReShape = theArr.reshape(4,2)

print("The shape of the array is ", arrReShape)

import numpy as np
theArr =  np.array([
         [2,4,6,8],
         [3,5,9,7],
         [11,22,33,44]
        ])

arrReShape = theArr.reshape(4,3)

print("The shape of the array is ", arrReShape)


# the np.reshape(k, -1) use to create k row and n-number of columns (same as number of element present).

import numpy as np
newRecord=np.array([8,99,84,0,35.4,0.388,50]).reshape(1, -1)
print(newRecord)

newRec=np.array([[8,99,84,0,35.4,0.388,50],
                    [11,143,94,146,36.6,0.254,51]]).reshape(2, -1)
print(newRec)

# the np.reshape(-1, k) use to create k-number of columns taking k no of elements each time.

import numpy as np
newRecord=np.array([8,99,84,0,35.4,0.388,50]).reshape(-1, 1)
print(newRecord)

newRec=np.array([[8,99,84,0,35.4,0.388,50],
                    [11,143,94,146,36.6,0.254,51]]).reshape(-1, 2)
print(newRec)

# Access element of single dimentional arry
"""
To access single dimentional array , we can use the index of the array.
The index is started from 0.
From backward the index start from -1
 
"""

import numpy as np

singDimArr = np.array([3,5,6,7,8,9]) 

fifthElem = singDimArr[4]

print("The fifth element of the single dimentional array ", fifthElem)

secElem = singDimArr[1]

print("The second element of the single dimentional array ", secElem)

lastElem = singDimArr[-1]

print("The last element of the single dimentional array ", lastElem)


# Access element of multi dimentional array

"""
In case of multi dimentional array the row index starts form 0 and ends with n-1 

each row contains the element, where the index starts from 0 and ends with n-1 with
restpect to row

so for multi dimentional array the index representation is as below
  Example
    0,3  -----------> first row's second element
    2,4  -----------> 3rd row's fourth element
    
"""

import numpy as np

mArr = np.array([
        [3, 5, 2, 4],
        [7, 6, 8, 5],
        [1, 6, 7, 9]
        ])

ind02 = mArr[0,1]

print("second element of zeroth row from the array is ", ind02)

ind33 = mArr[2,2]

print("third element of 3rd row from the array is ", ind33)

indSecRowLastEle = mArr[1,-1]
print("The last element of second row is ",indSecRowLastEle)

indLast = mArr[-1,-1]  #To get the last element of last row

print("The last element of last row is ",indLast)

# Array slicing or Accessing subarray using slicing.

"""
 slicing of single dimentional array uses the below syntax
     x[start:stop:step]  (x is the array )
     
     start -----> start indicates the starting index ( default is  0)
     stop ------> stop indicates the ending index    ( default is size of the array)
     step ------> step indicates the iteration sequence (default is 1)
"""
import numpy as np
singArr = np.arange(10)
print("Single dimentional array is ", singArr)  # output ---- [0 1 2 3 4 5 6 7 8 9]

first5ele = singArr[:5]
print("The first five element from single dimentional array ", first5ele)

after6ele = singArr[6:]

print("The elements after sixth index in single dimention array", after6ele)

subArr = singArr[4:8]

print("The sub array from 4th index to 8th index ", subArr)

altIndx = singArr[::2]

print("Every other element from single dimentional array ", altIndx)

altIndx1 = singArr[1::2]

print("Every other element index start from 1, from single dimentional array ", altIndx1)

revArr = singArr[::-1]

print("The array in reverse order ", revArr)

revArr4 = singArr[4::-1]

print("The reverse array from index 4 ", revArr4) # [4,3,2,1,0]

revArr4 = singArr[6::-2]

print("The reverse every other element from array from index 6 ", revArr4) # [6,4,2,0]

revArr4 = singArr[6:1:-2]

print("The reverse every other element from array from index 6 to index 1", revArr4) # [6,4,2]

# Slicing and accessing sub array from multidimentional array
"""
Slicing and accessing sub array from multidimentional array uses below syntax

x[ rowStart:rowStop:rowStep , colStart:colStop:colStep ]  -----> where x is the array

"""

import numpy as np
multiArr = np.array([
        [15, 17, 14, 18],
        [12,  5, 21,  4],
        [ 7, 16,  8,  9],
        [ 1,  6, 19,  7],
        [22, 33, 44, 55]
        ])
print("The multidimention array is  ", multiArr)
print("The shape of the array is ", multiArr.shape)

twoRows3Cols = multiArr[:2 , :3]
print("two rows, three columns from multidimentional array " , twoRows3Cols) 

allRowEvOthCol = multiArr[: , ::2]
print("All rows, every other column elements from multi dimention array",allRowEvOthCol )

allRowRevCol = multiArr[: , ::-1]
print("All rows, reverse all column in reverse order ", allRowRevCol)

revRowallCol = multiArr[::-1 , :]
print("Reverse all rows, not the column in reverse order ", revRowallCol)

revAllRowCol = multiArr[::-1 , ::-1]
print("reverse all rows and columns ", revAllRowCol)


secRow = multiArr[1]
print("second row of the multidimention array ", secRow)

secCol = multiArr[: , 1]
print("Second cloumn of the multidimentional array ", secCol)

threeCrossTwo = multiArr[:3 , :2]
print("By using first 3 rows and 2 columns create a 3X2 array ",threeCrossTwo )

secEleTrdRow = multiArr[2 , 1]
print("second element of third row is  ", secEleTrdRow)


# creating copy of an array or a subarray

import numpy as np
multiArr = np.array([
        [15, 17, 14, 18],
        [12,  5, 21,  4],
        [ 7, 16,  8,  9],
        [ 1,  6, 19,  7],
        [22, 33, 44, 55]
        ])

copyArr = multiArr.copy()
print("The copied array is  ", copyArr)

subArr = multiArr[:3 , :3]
print("sub array is  " , subArr)

copySubArr = multiArr[:3 , :3].copy()
print("copy of the sub array ",  copySubArr)

# Reshape of array
"""
To reshape an array, the size of the initial array should match with the new 
reshape array.
 
The reshape is very much useful for the conversion of one dimentional array to 
multidimentional array.

"""

import numpy as np

singDiArr = np.arange(2,17,2) 
print("the single dimentional array is " , singDiArr)

reshape4Cro2 = singDiArr.reshape(4,2)
print("the single dimentional array is reshaped to 4X2 dimention",reshape4Cro2)


reshapeToCol = singDiArr.reshape(8,1)
print("row array reshape to column array or column vector " , reshapeToCol)


#Array Concatenation and Splitting

"""
By using  np.concatenate, np.vstack, and np.hstack we can concatenate two arrays.

syntax of concatenate
       concatenate((a1, a2, ...)[, axis=0])   ---> a1 ,a2 ... are arries
                       bydefault axis=0 meaning row wise concatenation
                       means when you concatinate it will create new row
                       due to the default axis=0 for multidimentional array

hstack(tup)  ---> it give the same out put like axis=1
vstack(tup)  ---> it give the same out put like axis=0

dstack(tup) ----> concatinate with respect to axis=2

"""

import numpy as np

fstArr = np.array([1,2,3])
secArr = np.array([5,7,9])

concArr = np.concatenate((fstArr,secArr))
print("The concatenation of two array is ", concArr )

fst2dArr = np.array([
        [1, 2, 3],
        [4, 5, 6]
        ])
sec2dArr = np.array([
        [4, 6, 8],
        [3, 5, 7]
        ])

conc2dArr = np.concatenate((fst2dArr,sec2dArr))
print("The concatenation of two array is ", conc2dArr )

conc2dArrAxi1 = np.concatenate((fst2dArr,sec2dArr),axis=1)
print("concatenate along the second axis=1 ",conc2dArrAxi1)


hstackArr = np.hstack((fstArr,secArr))
print("The concatenation of two array is ",hstackArr)

hstack2dArr = np.hstack((fst2dArr,sec2dArr))
print("The concatenation of two array is ",hstack2dArr)

vstackArr = np.vstack((fstArr,secArr))
print("The concatenation of two array is ",vstackArr)

vstack2dArr = np.vstack((fst2dArr,sec2dArr))
print("The concatenation of two array is ",vstack2dArr)


#splitting of array

"""
array_split ---- Split an array into multiple sub-arrays of equal or near-equal size.
split ---- Split array into a list of multiple sub-arrays of equal size.
        split(arr , split_index_or_section [,axis=0] )
        
        split_index_or_section ---> if index/section is N then array will split 
                                    into N equal arrays.
                                    if index/section is [M,N] then array will split 
                                    into array[:M] , array[M:N] , array[N:]
                                    
hsplit ---- Split array into multiple sub-arrays horizontally 
        hsplit(arr , split_index_or_section) --------> axis = 1
        
vsplit ---- Split multi dimentional array into multiple sub-arrays vertically
        vsplit(arr , split_index_or_section)  -------> axis = 0
        
dsplit ---- Split array into multiple sub-arrays along the 3rd axis (depth).
        dsplit(arr , split_index_or_section)

"""

import numpy as np
singArr =  np.arange(10)

arrSplit = np.array_split(singArr,3)  # ---> create nearly equal size sub array
print("The splited array into multiple sub array ",arrSplit)


import numpy as np
singArr =  np.arange(9)

splitArr = np.split(singArr,3)  # split() will create sub array of equal size
print("the splited array in multiple array " , splitArr)

import numpy as np
singArr = np.arange(10)

splitArr = np.split(singArr,[3,6]) # 0:3 -- fst arr , 3:6 -- sec arr , 6: -- 3rd array 
print("the splited array in multiple array " , splitArr)

import numpy as np
multiArr = np.array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]
       ])


multiArrSplit = np.array_split(multiArr,3)
print("spliting single multidimentional array to three arraies",multiArrSplit)

splitMulArr = np.split(multiArr,2)
print("spliting single multidimentional array to two arraies",splitMulArr )

splitMulArr = np.split(multiArr,[2,4]) # :2 -- first , 2:4 -- 2nd , 4: --- 3rd
print("spliting single multidimentional array to two arraies",splitMulArr )


import numpy as np
singArr =  np.arange(9)
multiArr = np.array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]
       ])

hsplitArr = np.hsplit(singArr,3)
print("split the array using hsplit for axis=1 " , hsplitArr )

hsplitMultiArr = np.hsplit(multiArr , 2) # spliting wrt columns 
print("split the array using hsplit for axis=1 " , hsplitMultiArr) 

hsplitMultiArr = np.hsplit(multiArr , [3]) #spliting wrt columns :3 --fst 3: -- 2nd 
print("split the array using hsplit for axis=1 " , hsplitMultiArr) 


import numpy as np
singArr =  np.arange(9)
multiArr = np.array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]
    ])

vsplitMultiArr = np.vsplit(multiArr , 2) # spliting wrt rows 
print("split the array using hsplit for axis=1 " , vsplitMultiArr) 

#spliting wrt rows :3 --fst 3:5 -- 2nd 5: --3rd
vsplitMultiArr = np.vsplit(multiArr , [3,5]) 
print("split the array using hsplit for axis=1 " , vsplitMultiArr) 


# Iteration over N-dimentional array
"""
By using  np.nditer() we can iterate multi-dimensional arrays

By using nditer we can create transpose of array/ matrix (row to column)

"""

import numpy as np

singArr = np.arange(9)

for ele in np.nditer(singArr):
    print(ele)
    

multiArr = np.array([
       [ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [10, 15, 14, 11],
       [12, 13,  8,  9]
    ])
    
print("iterate multidimentional array")
for mulEle in np.nditer(multiArr):
    print(mulEle)
    
    
##########  Advance method of accessing element of Multi dimentional array ##########
########## This is known as integer indexing

"""
    Case 1
    ========
 When multidimensional array is indexed for accessing element from the 
 source multidimensional array in that case we have to follow below steps.

First we need to check the shape of the multidimensional array which is used
for indexing. The shape should be same else it will through "value error" due 
to shape mismatch.

So now condider an array "Xarr", and we will find the element of the array by
passing multidimensional array as index.
Example -->
Xarr[[0,1,2],[1,2,2]] here the multidimensional array is [[0,1,2],[1,2,2]]

In this case we will get the values correspond to the index set that is
Xarr[0,1] ---- where  0 is the row and 1 is the column
Xarr[1,2] ---- where  1 is the row and 2 is the column
Xarr[2,2] ---- where  2 is the row and 2 is the column
     
"""

import numpy as np
theArr = np.arange(1,10,1).reshape(3,3)
print("the array given is  ",theArr)

print("Get the element by using multidimensional array as index")

theEle = theArr[[0,1,2],[1,0,2]] #used array for indexing shape should be same

print("The elements are  ", theEle) ###   [2,4,9]

"""
case 2 - broadcasting mechanism to access element
=================================
In  broadcasting mechanism permits index arrays to be combined with scalars.
So the scalar value is used for all the corresponding values of the index arrays.

let consider an example Xarr[[0,1,2],2] means the elements of below indexes
Xarr[0,2] ---- where  0 is the row and 2 is the column
Xarr[1,2] ---- where  1 is the row and 2 is the column
Xarr[2,2] ---- where  2 is the row and 2 is the column

"""

import numpy as np
theArr = np.arange(1,10,1).reshape(3,3)

print("the Array is ",theArr)

print("Get element by broadcast mechanism ")

theEle = theArr[[0,1,2],2]
print("The elements are ", theEle)  # [ 3, 6, 9 ]

"""
Index array may be combined with slices

"""

import numpy as np
theArr = np.arange(1,10,1).reshape(3,3)

print("the Array is ",theArr)

theEle = theArr[[0,1,2],1:3]  # index combined with slices
                              # so elements are [0,1],[0,2],[1,1],[1,2],[2,1],[2,2] 
print("The elements are ", theEle)



#################  3D  Array  ###################

import numpy as np

threDarr = np.array( [ 
            [ [2, 3, 4],
              [7, 8, 9]
            ],
            [ [10, 11, 12],
              [15, 25, 35]
            ]
                ] )
print("The 3D matrix is  ", threDarr) 
the0thInd = threDarr[0]
print("The elements of zeroth index ", the0thInd)

the1stInd = threDarr[1]
print("The elements of first index is ", the1stInd)



################ Boolean Array Indexing

import numpy as np 
multiArr = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 

print("The array is ", multiArr)
print('\n')  
# Now we will print the items greater than 5 
 
print('The items greater than 5 are:' )
print( multiArr[multiArr > 5] )


"""
Using boolean array indexing we can filter the NaN (not a number) from array
"""

import numpy as np
theArr = np.array([1,2,np.nan,3,np.nan,np.nan,7])
print("The array is ", theArr)
print('\n')

# to remove nan form array we can use  "isnan"  function in boolean indexing

eles = theArr[~np.isnan(theArr)]

print("The elements without NaN is ", eles)

"""
 to filter out the non-complex elements from array by using boolean indexing
"""

import numpy as np
theArr = np.array([ 1, 2, 3+4j, 4 , 1j+5 , 3.5+2j ])
print("The array is ", theArr)
print('\n')

# to get all the complex nos form array we can use  "iscomplex"  function.

compEle = theArr[np.iscomplex(theArr)]

print("The complex elements are ", compEle)



#################### flat  function ----> syntax  arrName.flat[index] 
"""
The flat function return given index value by converting multi dimensional 
array to one dimensional array in memory. 
"""

import numpy as np 
multiArr = np.arange(12).reshape(3,4) 
print("The array is ", multiArr)
print('\n')
# returns element corresponding to index in flattened array 
ele7thInd = multiArr.flat[7]
print("The 7th index element is ", ele7thInd)

ele7thTo9thInd = multiArr.flat[7:10]
print("The 7th index element is ", ele7thTo9thInd)


####### ravel function ----> syntax arrName.ravel([order='F']) ---> order is optional
"""
The ravel() function convert multidimensional array to single dimensional array

"""

import numpy as np 
multiArr = np.arange(12).reshape(3,4) 
print("The array is ", multiArr)
print('\n')

#return the single dimentional array

singArr = multiArr.ravel()    #  c language type memory allocation

print("the single dimensional array is ", singArr)

singArrF = multiArr.ravel(order='F')  # Fortran language type memory allocation

print("The single dimensional array is ", singArrF)


#############  resize ()
"""
The resize function is use to return a new array with specified size.
If the new size is greater than the original then the extra places will
be filled with repeated copies.
"""

import numpy as np
aArr = np.arange(1,10)
print("The array is ", aArr)

newResizArr = np.resize(aArr , (3,3) )  # here (3,3) indicates 3X3 2D array

print("The new resized array of 3X3 is ", newResizArr)


newResizArr = np.resize(aArr , (3,4) )  # here (3,3) indicates 3X3 2D array
# 3X4 is 12 elements but origial array has 9 element so extra places will
# be filled with repeated copies.
print("The new resized array of 3X4 is ", newResizArr) 

"""
What is the difference between reshape and resize is 
in case of reshape the original array need to be reshaped with the existing
number of elements. But in case of resize the size of the array can be increase
or decrease 

"""

###########   append()  method

"""
To append the input array dimension should match with the existing array.
The append operation for array is not in place so it will create a new array. 

When we specify the axis the append method will create multi dimension array
so value should be in proper multi dimension format.
syntax
    np.append(arr_name,values[,axis])
"""

import numpy as np
aArr = np.arange(2,12).reshape(5,2)

print("The given array is ", aArr)

appendArr = np.append(aArr,[50,55]) # in this case it will create 1D array

print("The appended new array is ", appendArr)

appendArrAxis0 = np.append(aArr,[[23,25]],axis=0) 

print("The new array after append elements along axis 0 ",appendArrAxis0 )

appendArrAxis1 = np.append(aArr , [[33],[44],[28],[34],[63]],axis=1) 

print("The new array after append elements along axis 1 ",appendArrAxis1 )

#############  insert() function
"""
The insert() function inserts values in the input array along the given axis.
The insert operation for array is not in place so it will create a new array. 

syntax --->
    np.insert(arr_name, index , value , axis )
    
"""

import numpy as np

aArr = np.arange(5,17).reshape(6,2)

print("The array is ",aArr)

insertEle = np.insert(aArr , 3 , [99,88])

print("The new array is ", insertEle)

inserEleAxis0 = np.insert(aArr , 5 , [[44,22]], axis=0) 

print("the new array after insert element along axis 0 ", inserEleAxis0)

inserEleAxis1 = np.insert(aArr ,1,11, axis=1)  

print("the new array after insert element along axis 1 ", inserEleAxis1)

############## delete() function
"""
The delete() function use to delete the specified sub array from the 
existing array.
Delete function return a new array.
syntax--->
    np.delete(arr, index, axis=None) ---> The index can be slice, int 
                                                or array of ints 
                                        Indicate which sub-arrays to remove.
                                    ----> axis  can be  0 or 1 default is None
"""

import numpy as np

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

print("The array is ",arr)
eleDel = np.delete(arr, 1 , 0)  # 1 is the row index , 0 is the axis

print("After delete the new array is ", eleDel)

eleDel0 = np.delete(arr, np.s_[::2] , 0)  # here the index is a incremental list of 0,2,4.... and so on
                                          # np.s_[start:end:increment]
print("After delete the new array is ", eleDel0)

eleDel1 = np.delete(arr, np.s_[::2] , 1)

print("After delete the new array is ", eleDel1)


################  transpose
"""
 To invert the transposition for an array of components we need transpose() 
 function. Transposing a 1-D array returns an unchanged view of original array.
 syntax 
         np.transpose(a, axes=None)
"""

import numpy as np

arr = np.arange(12).reshape(3,4) 
print("array is ", arr)
transArr = np.transpose(arr)
print("Transpose of array is " , transArr)

##############  swapaxes 
"""
Interchange two axes of an array.
syntax 
        np.swapaxes(a, axis1, axis2)

"""

import numpy as np
arr=np.array([[1,2,3]])
print("Original array is ", arr)

swapArr= np.swapaxes(arr,0,1)
print("After swap axes the array is ", swapArr)


import numpy as np
arr2D = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
print("Original array is ", arr2D)

swap2dArr = np.swapaxes(arr2D,0,2)
print("After swap axes the array is ", swap2dArr)


##################### Trigonomatric function
"""
By using trigonomaric function we can find out sin , cos , tan value 
for a given array. Also we can calculate arcsin, arcos, and arctan functions
(inverse of sin , cos , tan ........)
As well as we can convert radians to degrees by using degrees() function.
"""

import numpy as np
anglArr = np.array([0,30,45,60,90])

print("The array is ",anglArr)

sinThita = np.sin(anglArr * np.pi/180) # to calculate sin0 use (np.sin())

print("The sin0 value is ",sinThita)
            # [0.         0.5        0.70710678 0.8660254  1.        ]

invOfSin = np.arcsin(sinThita)
print("The inverse of sin0 is " ,invOfSin)
            #  [0.         0.52359878 0.78539816 1.04719755 1.57079633]

chkDegree = np.degrees(invOfSin)
print("Result in degree is ", chkDegree)
            # [ 0. 30. 45. 60. 90.]

# similarl we can find  cos , arcos , tan and atctan 
            
################## around()  function
"""
The around() function return the value rounded to the described precision
Syntax
        around(a [, decimals = 0 ])
"""            

import numpy as np

arr = np.array([1.0,5.55, 123, 0.567, 25.532])

print("The array is ", arr)

roundedArr = np.around(arr)

print("The value after rounded ",roundedArr)

roundedArr1 = np.around(arr, decimals=1)

print("The value after rounded to precision 1 ", roundedArr1)

roundedArr_1 = np.around(arr, decimals=-1)

print("The value after rounded to precision 1 ", roundedArr_1)


##########  floor() function
"""
The floor() function returns the largest integer 
not greater than the input parameter.

"""

import numpy as np

arr = np.array([-1.7, 1.5, -0.2, 0.6, 8, 10])

print("The array is ", arr)

floorArr = np.floor(arr)

print("The value return by floor method is ", floorArr)


############## ceil() function
"""
The ceil function returns the nearest largest integer 
greater than the input parameter.

"""

import numpy as np

arr = np.array([-1.7, 1.5, -0.2, 8, 0.6, 10])

print("The array is ", arr)

ceilArr = np.ceil(arr)

print("The value return by ceil method is ", ceilArr)


#### Broadcasting of array ( adding , substracting , multiplying , deviding )
"""
         
As per mathmtics process we can not add , substract, multiply 
and devide a  " m X n " matrix with another matrix of " 1 X n " or " m X 1 ".
In python by using broadcast mechanisim the array/matrix of " 1 X n " or
"m X 1" converted to  " m  X n" array/matrix by coping the data.
Then the addition , devision , multiplication , devison operations can be 
performed with the matrix.

For example ---> 
        fstArr = [            secArr = [         In this case we can not add, 
                  [1,2,3],              [3,4,7]  sustract , multipy and devide
                  [4,5,6],               ]       both array in python. so by
                  [3,8,7]                        using broadcast mechanisim
                    ]                            the second array "secArr"
will change to     secArr = [
                              [3,4,7],
                              [3,4,7],      now we can do all the operations
                              [3,4,7]       element wise
                              ]

similarly in case of example2 --> 
        fstArr = [            secArr = [     In this case we can not add, 
                  [1,2,3],              [3]  sustract , multipy and devide
                  [4,5,6],              [4]  both array in python. so by
                  [3,8,7]               [8]  using broadcast mechanisim
                    ]                    ]   the second array "secArr"
will change to     secArr = [
                              [3,3,3],
                              [4,4,4],      now we can do all the operations
                              [8,8,8]       element wise
                              ]

"""

import numpy as np

fstArr = np.array([[1,1,1,1],
                   [3,3,3,3],
                   [4,4,4,4],
                   [6,6,6,6]
                     ])

secArr = np.array([2,2,2,2])

secArrs = np.array([
                    [1],
                    [3],
                    [1],
                    [2]
                     ])

print("The both array are \n ", fstArr ,"\n\n\n" ,secArr ,"\n")

## adding these two arrays is

add2Arr = fstArr + secArr    ### we can use  sum() also for addition

print("The sum of the arrays is  \n", add2Arr)

add2Arr = fstArr + secArrs    ### we can use  sum() also for addition

print("The sum of the arrays is  \n", add2Arr)

## subtract of these two arrays are

sub2Arr = fstArr - secArr    ### we can use  subtract() also for addition

print("The sum of the arrays is  \n", sub2Arr)

sub2Arr = fstArr - secArrs    ### we can use  subtract() also for addition

print("The sum of the arrays is  \n", sub2Arr)

## multiplication of these two arrays are 

mul2Arr = fstArr * secArr  ## we can use multiply() also for multiplication

print("The multiplication of the arrays is  \n", mul2Arr)

mul2Arr = fstArr * secArrs

print("The multiplication of the arrays is  \n", mul2Arr)

## Devision of these two arrays are

div2Arr = fstArr / secArr ## we can use divide() also for devision

print("The devision of the two array is \n ", div2Arr)

div2Arr = fstArr / secArrs 

print("The devision of the two array is \n ", div2Arr)


   ##### satistical functions
###### max() function
"""
The max() give the maximum value from the array
"""

import numpy as np

arr = np.array([-1.7, 1.5, -0.2, 8, 0.6, 10])

print("The array is ", arr)

maxArr = arr.max()

print("The maximum value of the array is ", maxArr)

nArr = arr.reshape(3,2)
print("The array is ", nArr)

nMaxArr = arr.max()

print("The maximum value of the array is ", nMaxArr)


####### min()
"""
The min() give the minimum value from the array
"""
import numpy as np

arr = np.array([-1.7, 1.5, -0.2, 8, 0.6, 10])

print("The array is ", arr)

minArr = arr.min()

print("The maximum value of the array is ", minArr)

nArr = arr.reshape(3,2)
print("The array is ", nArr)

nMinArr = arr.min()

print("The maximum value of the array is ", nMinArr) 

############### percentile()  function
"""
Syntax
    percentile(arr, percent, axis)  -----> percent must be between 0 to 100
"""

import numpy as np

arr = np.array([[10, 7, 4], [3, 2, 1]])

print("The array is ", arr)

percArr = np.percentile(arr,50)

print("The percentile of array is ", percArr)   # ---- o/p 3.5

percArr0 = np.percentile(arr,50, axis=0)

print("The percentile of array along 0 axis ", percArr0) # (10+3) / 50 , (7+2) /50 .......

percArr1 = np.percentile(arr,50, axis=1)

print("The percentile of array along 1 axis ", percArr1) # (10+7+4)/50 ......

############  median() function
"""
find the Median of {13, 23, 11, 16, 15, 10, 26}. 
    Put them in order: {10, 11, 13, 15, 16, 23, 26}
    The middle number is 15, so the median is 15.

find the Median of {13, 23, 11, 16, 15, 10, 26,39}.
    Put them in order: {10, 11, 13, 15, 16, 23, 26,39}
    The middle number is (15+16)/2, so the median is 15.5.  

Syntax---->
        median(a, axis=None)   

"""

import numpy as np

arr= np.array([[10, 7, 4], [3, 2, 1]])

print("The array is ", arr)

mediArr = np.median(arr)

print("median of the array is  ", mediArr)  #  (3+4) / 2

mediArr0 = np.median(arr,axis=0)

print("median of array along  axis 0  " , mediArr0 ) # 10+3 /2 , 7+2 /2 ....

mediArr1 = np.median(arr,axis=1)

print("median of array along  axis 1  " , mediArr1 ) # 7 , 2


###########  mean() function
"""
Arithmetic mean is the sum of elements along an axis divided by the 
number of elements.

find the Mean of {13, 23, 11, 16, 15, 10, 26}. 
    sum of the nos = 13+23+11+16+15+10+26 = 114
    total no of elements = 7
    so mean is 114/7 = 16.29
syntax--->
        mean(arr, axis=None) 
"""

import numpy as np

arr = np.array([[1,2,3],[3,4,5],[4,5,6]])

print("The array is ", arr)

meanArr = np.mean(arr)

print("The mean of the array  ", meanArr) #1+2+3+3+4+5+4+5+6 /9 

meanArr0 = np.mean(arr, axis=0)

print("mean of array along  axis 0  " ,meanArr0) # 1+3+4 /3 , 2+4+5 / 3 .....

meanArr1 = np.mean(arr, axis=1)

print("mean of array along  axis 1  ",meanArr1) # 1+2+3 /3 , 3+4+5 / 3 .....

######## average() function
"""
The average of an array is 
        " sum of the product of the each element / weight of elements "

find the Median of {13, 2, 3, 10, 6 , 10 , 2, 2 }
    adding the product of the corresponding elements =13+(2X3)+3+(10X2)+6 = 48
    Sum of the occurence of each element ( means 1 time 13 , 3 time 2 ...)
            = 1 + 3 + 1 + 2 + 1 = 8 
    so average is  48/8 = 6

syntax --->
    average(arr, axis=None ,  weight=None)
"""

import numpy as np

arr = np.array([[1,2,3],[3,4,5],[4,5,6]])

print("The array is ", arr)

averArr = np.average(arr)

print("The average of the array is ", averArr) # 1+2+(2*3)+(2*4)+(2*5)+6 / 9

wt = np.array([3,2,4])

averArr0 =  np.average(arr, axis=0 , weights=wt)

print("The average of the array with respect to weight ", averArr0) # 1*3 + 3*2 + 4*4 / 3+2+4 .......

averArr1 =  np.average(arr, axis=1 , weights=wt)

print("The average of the array with respect to weight ", averArr1) # 1*3 + 2*2 + 3*4 / 3+2+4 .......


###########  Variance  
"""
find the Variance of { 2, 3, 10, 6 , 4 }
        step1--get the sum of the nos =  2+3+10+6+4 = 25
        step2--squar of the sum of the nos = 25 X 25 = 625
        step3--devide the squar result with the no of elements = 625 / 5 = 125
        step4--then find the sum of each elements squre = 2X2 + 3X3 + 10X10 + 6X6 + 4X4 = 165
        step5--then substract step3 from step 4 =  165 - 135.2 =40
        step6--total no of elements  = 5 
        step7--devide step5 by step6 =  40/5 = 8
    So the variance formula is mean(abs(x - x.mean())**2)
Syntax --->
        var(a, axis=None)
"""

import numpy as np

arr = np.array([2, 3, 10, 6 , 4])

variArr = np.var(arr)  # 8.0

print("the variance of the array is ", variArr)

import numpy as np

arr = np.array([[1,2,3],[3,4,5],[4,5,6]])

print("The array is ", arr)

variArr = np.var(arr)  

print("the variance of the array is ", variArr)


########## Standard Deviation
"""
The Standard Deviation is the square root of variance.
    So formula is sqrt(mean(abs(x - x.mean())**2))

syntax -----> 
        std(a, axis=None, dtype=None)
        
"""

import numpy as np

arr = np.array([3,4,5,6,7,8,9])

print("The array is ", arr)

stdArr = np.std(arr)

print("The standard deviation of array is  ", stdArr)

import numpy as np

arr = np.array([[1,2,3],[3,4,5],[4,5,6]])

print("The array is ", arr)

stdArr = np.std(arr)

print("The standard deviation of array is ", stdArr)


#### sort() function
"""
The sort() function use to sort the elements of array by useing different 
sorting algorithim (quicksort , mergesort , heapsort)
Syntax --->
        np.sort(a, axis, kind, order)

"""
"""
quicksort ------------>
    In case of quicksort an element "x" of array as pivot. Put "x" at its 
    correct position in sorted array and put all smaller elements 
    (smaller than x) before "x", and put all greater elements (greater 
    than x) after "x". 
Example for sorting of  10,80,30,90,40,50,70
                                
   step1        10,30,40,50 ------- 70 ---- 90,80  (partition around 70 )
   step2        10,30,40-----50  (around 50)   and     80----90 ( around 80)
   step3        10,30 ------ 40 (around 40)
   step4        10-----30 (arround  30)
   step5        10
   
so sorted array is (10 , 30 , 40 , 50 , 70 , 80 , 90)
"""

"""
Merge Sort ------------------>
        Merge Sort is a Divide and Conquer algorithm. It divides input array 
in two halves, and then sort the two halves and merge them.

Example for Merge Sort of  (38, 27 , 43 , 3 , 9)
    step1       38 , 27              and         43 , 3 , 9  ( devided )
    step2       38      27  (devided)        and     3 , 43  and   9 (devided)
    step3       27,38  (merged)   and    3 and 43 ( devided ) and  9
    step4       27,38    and   3,43 (merged) and 9
    step5       27,38    and   3,9,43  ( merged )
    step6       3,9,27,38,43 ( merged )
"""

"""
Heap  sort ----------------->
        The heap sort is a binary tree sort algorithm. 
"""

import numpy as np

arr = np.array([[3,7,5],[9,1,8]])

print("The array is ", arr)

sortArr = np.sort(arr)  ### sort along the last axis [[3,5,7],[1,8,9]] here last axis is 1

print("The sorted array is ", sortArr)

sortArr0 = np.sort(arr,axis=0)  ### [ [3,1,5],[9,7,8]]

print("The sorted array along axis0  ", sortArr0) 

# ------------------------

import numpy as np

dataType = np.dtype([('name', 'S10'),('age', int)]) 

theArr = np.array([("suman",21),("sksahoo",25),("uma", 17), ("uksahoo",27)], dtype = dataType) 

print("The array is " , theArr)

sortArr = np.sort(theArr,order = 'name')

print("The sorted array is ", sortArr)

######   lexsort()  function

"""
The lexsort() function perform an indirect sort using a sequence of keys.
lexsort(keys[, axis=-1]) ----> keys should be shorted and in form of tuple
                         ----> bydefault axis is the last axis and optional. 
"""

import numpy as np

lstNam = ("sahoo" , "das" ,"nayak")
fstNam = ("sk","uk","pk")

namTupIndex = np.lexsort((fstNam,lstNam))

lexIndexVal = [ fstNam[i] + " : " + lstNam[i] for i in namTupIndex ] 

print(" The sorted value of lexsort is  ", lexIndexVal)


# Note ----> same can be achived by using   argsort() function.
# the argsort() function return sorted indexes of an array for a given axis
 
import numpy as np

lstNam = ("sahoo" , "das" ,"nayak")
fstNam = ("sk","uk","pk")

agrsortVal = [ fstNam[i] + " : " + lstNam[i] for i in np.argsort(lstNam)]

print(" The sorted value of argsort is  ", agrsortVal)


############### NumPy - Matrix Library

#### zeros()  function
"""
The zeros() function of matrix library return a matrix with zeros.
syntax --->
        zeros(shape, dtype=float)
"""

import numpy as np

zerosMatrix = np.zeros((2,2))

print("The zeros matrix is ", zerosMatrix)


####  ones() function

"""
The ones() function of matrix library return a matrix with zeros.
syntax --->
        ones(shape, dtype=float)
"""

import numpy as np

onesMatrix = np.ones((2,3))

print("The zeros matrix is ", onesMatrix)

#####  identity() function

"""
The identity() function of matrix library return a matrix with zeros.
syntax --->
        identity(n, dtype=float)
"""

import numpy as np

identityMatrix = np.identity(3)

print("The zeros matrix is ", identityMatrix)


#####  asarray() function
"""
The asarray() function use to convert from any form to array.
syntax-->    
    asarray(a, dtype=None) ---->  a is the any from that can convert to array

"""

import numpy as np

val = [3,4,5,6]

arr = np.asarray(val)

dtyp = arr.dtype

print("The array is " , arr , " data type is " ,dtyp )


#####  asmatrix() funciton
"""
The asmatrix() function use to convert from any form to array.
syntax-->    
    asmatrix(m, dtype=None) ----> m is the any from that can convert to matrix

"""

import numpy as np

arr = np.array([3,4,5,6])

matrix = np.asmatrix(arr)

print("The matrix is " , matrix )


####################### Linear algebra

#### dot() function 
"""
The dot() function is use for dot product of two arrays.

If both arrays are 1-D arrays, it is inner product of vectors.

which is equivalent to matrix multiplication for 2D array, where first 
matrix dimension will be  2Xm (2D array) and second matrix dimension 
will be 2Xk (2D array).

If either a or b is 0-D (scalar), it is equivalent to multiply and 
using numpy.multiply(a, b) or a * b is preferred.

If a is an N-D array and b is a 1-D array, it is a sum product over the 
last axis of a and b.

If a is an N-D array and b is an M-D array (where M>=2), it is a sum 
product over the last axis of a and the second-to-last axis of b.

syntax ----->
    dot(a, b, out=None) -------> a , b both are the array 
                        -------> if a, b both are 1D array then output is 
                                scalar else it will be an array.
"""

import numpy as np

fst = np.array([[1, 0], [0, 1]])
sec = np.array([[4, 1], [2, 2]])

dotRes =  np.dot(fst,sec)

print("The dot product of the array is ", dotRes)


import numpy as np

fst = np.array([[1, 1], [2, 1]])
sec = np.array([[4, 1,2], [2, 2,3]])

dotRes =  np.dot(fst,sec)

print("The dot product of the array is ", dotRes)


#### vdot()  function 
"""
The dot product of two vector is a scalar. The verctors should be same dimension. 

example --->

        [3,1,-7] . [-13,-11,-10] = (3*-13) + (1*-11) + (-7*-10) 
                                 = (-39) + (-11) + 70
                                 = -50 + 70 
                                 = 20
Syntax ---------->
    vdot(a, b) -----------> where a and b both are the verctors of same dimension
"""

import numpy as np

vec1 = np.array([3,1,-7])

vec2 = np.array([-13,-11,-10])

vecDot = np.vdot(vec1,vec2)

print("the vectors dot product is " , vecDot)


import numpy as np

vec1 = np.array([[3,1,-7],[2,3,-2]])

vec2 = np.array([[-13,-11,-10] ,[3,-2,4]])

vecDot = np.vdot(vec1,vec2)

print("the vectors dot product is " , vecDot)


###### inner() function
"""
The inner() function is to perform sum product of A and B array
of same dimension.

The inner product of two  1D verctor is a scalar.

        [3,1,-7] . [-13,-11,-10] = (3*-13) + (1*-11) + (-7*-10) 
                                 = (-39) + (-11) + 70
                                 = -50 + 70 
                                 = 20
So dot or vdot or inner all functions will give same result for 1D array.

Example sum product of A and B array
    A = [[x1, x2], 
         [x3, x4]]
    B = [[y1, y2],
         [y3, y4]]
    inner(A,B) = [[x1*y1+x2*y2, x1*y3+x2*y4], [x3*y1+x4*y2, x3*y3+x4*y4]]

"""

import numpy as np

fst = np.array([3,1,-7])

sec = np.array([-13,-11,-10])

innerRes = np.inner(fst , sec)

print("The inner product of two 1D arrays " , innerRes)


import numpy as np

fst = np.array([[3,1],[2,5]])

sec = np.array([[-5,10],[7,4]])

innerRes = np.inner(fst , sec)

print("The inner product of two 1D arrays " , innerRes)


########## matmul() function
"""
In case of matmul() function returns the matrix product of two arrays.

*) If both arguments are 2-D they are multiplied like conventional matrices.

*) If either argument is N-D, N > 2, it is treated as a stack of matrices 
residing in the last two indexes and broadcast accordingly.

*) If the first argument is 1-D, it is promoted to a matrix by prepending 
a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.

*)If the second argument is 1-D, it is promoted to a matrix by appending 
a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

syntax ---->
        matmul(a, b, out=None)
"""


import numpy as np 

fst = [1,0] 
sec = [[4,1,4],[2,2,3]] 
matmulRes = np.matmul(fst,sec)
print("The matmul restult of 1X2 and 2X3 array is ", matmulRes)

import numpy as np 

fst = [[1,0],[2,1]] 
sec = [[4,1],[2,3]] 
matmulRes = np.matmul(fst,sec)
print("The matmul restult of 2X2 and 2X2 array is ", matmulRes)

import numpy as np 

fst = [[1,0],[2,1]] 
sec = [[4,1,4],[2,2,3]] 
matmulRes = np.matmul(fst,sec)
print("The matmul restult of 2X2 and 2X3 array is ", matmulRes)

import numpy as np 

fst = [[1,0],[2,1]] 
sec = [4,1]         # -----------> here we can not use 1X3 array 
matmulRes = np.matmul(fst,sec)
print("The matmul restult of 2X2 and 1X2 array is ", matmulRes)


######### det() function
"""
This det() function is use to find the determinant of a matrix.

For a 2x2 matrix, it is simply the subtraction of the product of the 
top left and bottom right element from the product of other two.

if matrix is mXn then 
    det(A) = sum of  1<j1 <.... jm<=n ( ( (-1)**m+n ) det[[a_1j1.... a_1jm],
                                                          [a_mj1.... a_mjm] ] )  
    
Syntax ---->
        np.linalg.det(arr)
"""

import numpy as np
arr = np.array([[1,2], [3,4]]) 
detRes = np.linalg.det(arr)

print("The determinant of 2X2 matrix " , detRes)



import numpy as np 

arr = np.array([[6,1,1], [4, -2, 5], [2,8,7]]) 
print("The array is ",arr) 
detRes = np.linalg.det(arr) 
print("The determinant of 3X3 matrix " , detRes)

print("The above result is same as 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2)   "
      ,6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2))


#######  solve() function
"""
The solve() function is use to solve the liner equaltion and return a matrix.

Considering the following linear equations −

x + y + z = 6

2y + 5z = -4

2x + 5y - z = 27

They can be represented in the matrix form as − A matrix * variables = B matrix

          [ [ 1, 1,  1],       [ [x]            [ [6 ]
            [ 0, 2,  5],    *    [y]     =        [-4]
            [ 2, 5, -1]  ]       [z] ]            [27] ]
            
To find the value of the varibales we can do = inverse of A * B

Syntax -----> 
            linalg.solve(a, b)      

"""

import numpy as np

matrixDat = np.array([ [1,1,1],
                       [0,2,5],
                       [2,5,-1]
                        ])
equVal = np.array([[6],
                   [-4],
                   [27]
                   ])
valXYZ = np.linalg.solve(matrixDat,equVal)

print("Value of x , y , z by solving liner equations ", valXYZ)


##### linalg.inv() function
"""
The linalg.inv() function is use to find the invrese of the matrix.

inverse A = adj A / det A
            ### adj A = find minor of A then find the cofactor of the result
                    and then find the transpose of the result.

syntax ----------->
            linalg.inv(a)
"""

import numpy as np


matrixDat = np.array([ [1,1,1],
                       [0,2,5],
                       [2,5,-1]
                        ])
          
invMatrix  = np.linalg.inv(matrixDat)

print("The inverse of the matrix is ",invMatrix )


##################    I/O  with numpy

#####      savetxt() function
"""
The savetxt() function,  save an array to a text file.

Syntax ---------->
        savetxt(fname, arr, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

Where the arguments are as below

    fname : filename or file handle , If the filename ends in .gz then the
            file is automatically saved in compressed gzip format.        
    arr : arr is the array which need to save in the file.
    fmt : A sequence of formats or a multi-format string.
    delimiter: The String or character which is used for separating columns.
    newline  : if newline is use as the separator for the line
    header   : if you need any header at the begining of the file.
    footer   : if you need any String that will be written at the end 
                of the file.
    comments : String that will be prepended to the header and footer 
                strings, to mark them as comments. Default: ' # ' as comment
    
"""

import numpy as np

arr = np.array([1,2,3,4,5,6])

np.savetxt('saveArray.txt' , arr)

####   loadtxt()  function

"""
The loadtxt() function is use to Load data from a text file. 
The each row in the text file must have the same number of values.

Syntax --------->
        loadtxt(fname, dtype=<type 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
        
Where the arguments are as below
    
    fname : file name from which we need to load the data. If the file 
            extension is .gz or .bz2 then first that file need to be decompress.
    
    dtype : Data-type of the resulting array; default: float.
    
    comments : The characters or list of characters used to indicate comment.
                the default: ‘#’.
    
    delimiter : The string used to separate values. 
                By default, this is any whitespace.
    
    converters : A dictionary mapping column number to a function that 
                will convert that column to a float.
                
    skiprows : Skip the first n number of rows ; default: 0.
    
    usecols :  Which columns to read. For example, usecols = (1,4,5) 
                will extract the 2nd, 5th and 6th columns. The default is 
                None, results in all columns being read. When a single 
                column has to be read it is possible to use an integer 
                instead of a tuple. E.g usecols = 3 reads the fourth column
   
    unpack :   If True, the returned array is transposed, so that arguments 
                may be unpacked using x, y, z = loadtxt(...). When used 
                with a structured data-type, arrays are returned for each 
                field. Default is False
                
    ndmin :     The returned array will have at least ndmin dimensions. 
                Otherwise mono-dimensional axes will be squeezed. 
                Legal values: 0 (default), 1 or 2
                
    
"""


import numpy as np

loadArr = np.loadtxt('saveArray.txt')

dataTypeIs = type(loadArr)
print("the datatype of loadArr " , dataTypeIs)

print("The data loaded from the file and form an array " , loadArr)
 

# Load CSV from URL using NumPy loadtxt() in form of an array 

import numpy as np

from urllib.request import urlopen

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

urlData =  urlopen(url)

dataInArr = np.loadtxt(urlData, delimiter = ',')

print("Url data loaded to an array is  " , dataInArr)


# we can read the CSV file data and convert it into Array in two ways.
##### First reading data using  csv  module
       
import csv
import numpy as np

fileName = 'indians-diabetes.data.csv'
                            
fopen = open(fileName , 'rt')
csvData =  csv.reader(fopen,delimiter=',',quoting=csv.QUOTE_NONE)
convList = list(csvData)        ### here in this case we read all the data
print(convList)
dataInArr = np.array(convList) 

print("The data loaded into array by reading CSV file  ", dataInArr)

##### second is by using loadtxt() 

import numpy as np

fileName = 'indians-diabetes.data.csv'

fopen = open(fileName , 'rt')

dataToArr = np.loadtxt(fopen , delimiter =  ',')
              ## here commment lines are ignored as per the loadtxt() syntax
print("The data loaded into array by reading CSV file  ", dataToArr)











