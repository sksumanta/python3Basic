# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:14:44 2018

@author: Sumanta

"""

print("====== find the single elememt of the list taking index as user input =======")
theList = [20,30,40,10,9,8,50]
def getItem(usrIndex , givenList):
    theIndex = usrIndex
    newList = givenList
    elemFound = newList[theIndex]
    return elemFound

#theLis=input("Enter the list  ")
theInd=int(input("Enter the index to find the element  "))
searchItem = getItem(theInd , theList)
print(searchItem)
 

print(theList[:len(theList)-1])

   
print("====== Find the range of elements of list by taking index as user input =====")
"""
def getRangeOfItems(startIndex = None , endIndex = None , givenList):
    length = len(givenList)
    if startInedx == None:
        startIndex = 0
    if endIndex == None:
        endIndex = length
    
  """  
    


#print("======= occurence of the elements in the list =========\n")

ll =  ['a','b','c','d','a','e','c']

def chCount(l):
    sl = set(ll)
    a={}
    for i in sl:
        a[i]=l.count(i)
    o=[(k,a[k])for k in sorted(a.keys())]
    return o

#print(chCount(ll))


#print("\n======= read file line by line ==========\n")
cou=0
with open(r'E:/datascience/pythontut/basics/basics.py','r') as fd:
    for lin in fd:
        cou = cou+1
        #print(cou,"    ", lin)
        
