# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 11:43:40 2018

@author: Sumanta
"""

# dictonary is key value pair 

# syntax  -----   dicts = {'key' : 'value'}

# key must be immuatble data object so key can be  " tuple , string , int "

# value can be any type

dict1= {1:'apple',
        2:'mango',
        3:'banana'
        }
print(type(dict1))

print(dict1)

# create dictionary from a dictionary function "dict"


dict2 = dict({1:'apple',
        2:'mango',
        3:'banana'
        })

print(type(dict2))

print(dict2)


# dictionary can be created for a  sequence ex ---  list

dict3 = dict(
          [
            (1,'apple'),
            (2,'mango'),
            (3,'banana')
          ]
        )

dict4 = {}
print(type(dict4))   # why it is  dictionary 

#empty set has one element  faay  so to satisfy mathmeticaly  the 
# representation is dictionary

# so to create the empty set
set1 = set()


dict1= {1:'apple',
        2:'mango',
        3:'banana'
        }
print(type(dict1))

print(dict1)

print(dict1[1])
print(dict1[3])

for key , value in dict1.items():    # to get all items of the dictionary  
    print(key)
    print(value)
    
for key in dict1.keys():  # to get the keys of the dictionary
    print(key)
    
for value in dict1.values():   # to get all values of the dictionary
    print(value)
 



dict1[4] = "Grapes"   # adding new key and value with respect to the key
for key , value in dict1.items():
    print(key)
    print(value)
    
    
dict1[4] = "Straberry"  # update the value of the existing key
for key , value in dict1.items():
    print(key)
    print(value)

del dict1[4]      # deleted the item using the del  
for key , value in dict1.items():
    print(key)
    print(value)
"""
del dict1  # delete whole dictionary
print(dict1)
"""


dict1[5] = ('rice','gram')   # added the tuple as the value in the dictionary
for key , value in dict1.items():
    print(key)
    print(value)
    


"""
create the dictionary using dictionary comprehnesion

dict6 = {2:4,
3:9,
4:16,
5:25
}
"""

powerDict={x:x**2 for x in range(6) if x > 1 }
print(powerDict)

"""
lis1 = [1,2,3,4,5,6]
lis2 = [20,22,23,25,27,28]

create dictionary from the above lists 
dict7 = {
        1:20,
        2:22,
        3:23,
        :
        :
        6:28}
"""

lis1 = [1,2,3,4,5,6]
lis2 = [20,22,23,25,27,28]


newDict = { lis1[length] : lis2[length] for length in range(len(lis1))}
print(newDict)

print(dict(zip(lis1,lis2)))   # predefined method zip to 

"""
Read dictionary for other file by passing the user key input

below dictionay in one py file
dict1= {1:'apple',
        2:'mango',
        3:'banana',
        4:'grape',
        5:'strabery'
        }

"""







