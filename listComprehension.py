# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:31:48 2018

@author: Sumanta
"""

# The range function in the loop

for i in range(5):
    print(i)

##################### List Comprenhension ############################
   
lis=[]
for x in range(7):
    lis.append(2**x)
print(lis)      # --------  [1, 2, 4, 8, 16, 32, 64]   

lis=[2**x for x in range(7)]
print(lis) # --------  [1, 2, 4, 8, 16, 32, 64] same output as above in list Comprenhension

 

lis=[]
for x in range(7):
    if x>3:
        lis.append(2**x)
print(lis)      # --------  [16, 32, 64]

lis=[2**x for x in range(7) if x>3]  # find the power list if x > 3
print(lis) # -----  [16, 32, 64]  same output as above in list Comprenhension



lis1=[x for x in range(7) if x%2==0 ]
print(lis1)

lis1=[x for x in range(7) if x%2==1 ]
print(lis1)



######  [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]


