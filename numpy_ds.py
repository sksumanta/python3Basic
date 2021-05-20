# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 08:54:11 2018

@author: Sumanta
"""



import numpy as np 
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
   
print( 'Our array is:' )
print(x )
print( '\n' )

rows = np.array([[0,0],[3,3]])
print(rows)

print( '\n' )

cols = np.array([[0,2],[0,2]]) 
#print(cols)
y = x[rows] 
print(y)

y = x[cols]   
print( 'The corner elements of this array are:' )
print( y)


y = x[rows,cols]   
print( 'The corner elements of this array are:' )
print( y)

