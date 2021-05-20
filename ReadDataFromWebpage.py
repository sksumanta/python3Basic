# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:27:39 2019

@author: Sumanta
"""

            # Read data from web page using html parser

from bs4 import BeautifulSoup # for parsing HTML & XML documents and extract data
import requests 

def allNews(): 
    # the target we want to open     
    url='http://www.hindustantimes.com/top-news'
      
    #open with GET method 
    resp=requests.get(url) 
      
    #http_respone 200 means OK status 
    if resp.status_code==200: 
        print("Successfully opened the web page") 
        print("The news are as follow :-\n") 
      
        # parse using HTML parser  
        soup=BeautifulSoup(resp.text,'html.parser')     
  
        # containts   is the list which contains all the news text   
        containts = soup.findAll("div",{"class":"para-txt"})
        #print( len(containts)) 
      
        #now we want to print only the text part of the news        
        for i in containts: 
            print(i.text) 

    else: 
        print("Error") 


allNews()

