# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:18:45 2019

@author: Sumanta
"""


charCount =[]
def countchar(theStr):  #unique char Count in a string in sorted order
    uniqChar = sorted(list(set(theStr)))
    for ch in uniqChar:
        occur = theStr.count(ch)
        charCount.append( ( ch , occur))
    charCount.sort(key = lambda x: x[1] , reverse=True) 
    #print(charCount)      
    return charCount

chars=[]
def topThree(charlist):
    for i in range(3):
        charNcount = charCount[i]  
        print(charNcount[0],charNcount[1])
    #return chars

if __name__ == '__main__':
    n = str(input())
    charstr = countchar(n)
    topThree(charstr)