# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 00:48:13 2018

@author: Sumanta
"""

# unless user enter the correct the numeric till that time it will ask user input
# and find the input is present in the tuple or not. 

theTuple = (20,21,23,24,26,27)


infiVal = True
while infiVal:
    inVal = input("Enter the item & check item present in tuple or not : ")
    try:
        if type(int(inVal)) == int:
            print("The input value is numeric that is ", inVal)
            if int(inVal) in theTuple:
                print("The value is present in the Tuple ")
                break
    except:
        infiVal = True
        
# unless user enter the correct the numeric till that time it will ask user input 
# for the devision of the items in the tuple
 
theTuple = (20,21,23,24,26,27)
       
infiVal = True
while infiVal:        
    divNo = input("Enter the devision factor : ")
    try:
        if type(int(divNo)) == int:
            print("The input value is numeric that is ", divNo)
            try:
                for item in theTuple:
                    div = item/int(divNo)
                    print(" The division of ", item ," / " , divNo , " is " , div)
                break
            except ZeroDivisionError as zde:
                print("please divide by number greater than zero")
            
    except:
        infiVal = True



# Combine both the program to find the item present in the list and devide the item
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
            
        
    



