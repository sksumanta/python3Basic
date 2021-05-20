# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:56:18 2018

@author: Sumanta
"""

# what is while loop  and what is the difference between for and while loop

# why while loop

"""
when the range of the iteration is not fixed then we can use while loop

when the range of the iteration is fixed then we can use for loop

"""

######  Syntax of while
"""
while expression:
    body
"""

x = res =0
while x<10:
    res=res+x
    x=x+1
    
print(res)

##### Break and continue
## to break the infinite loop and come out from the loop.

name = "Sumanta"
for n in name:
    if n =='t':
        print("as " , n , " found loop exit")
        break
    print(n)


# countinue will skeep the current execution and go to the begining of the loop
    
name = "Sumanta"
for n in name:
    if n =='t':
        print("as " , n , " found it is skeeped")
        continue
    print(n)


# while loop can be execute with else
    """
    while exp:
        body
    else:
        else_body
    """
    
# pass -- it is not ignore by the complier, so we can use any block having pass
# for our future use.

x = "apple"
for i in x:
    if i == 'a':
        pass
    elif i=='l':
        print(i)







