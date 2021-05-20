# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:23:34 2019

@author: Sumanta
"""


import random

class A(object):
	def __init__(self,name):
		self.name = name

class B(A): # B inherite A class
    def __init__(self, name):
        super(B , self).__init__(name)		# called parent class constructor with super
        self.mychoice = random.choice(['kfc' , 'fivestar' , 'magD'])
        
    def place(self, thing):
        print(self.name,"  like  " ,self.mychoice, thing)
        
b = B('sumanta')
b.name
b.place('chicken')


'''
A decorator in Python is any callable Python object that is used to modify 
a function or a class without modifying  the function or class permanently.

'''


import dask
from dask import delayed, compute

@delayed
def square(num):
    print("square fn:", num)
    print()
    return num * num

@delayed
def sum_list(args):
    print("sum_list fn:", args)
    return sum(args)

items = [1, 2, 3]

computation_graph = sum_list([square(i) for i in items])

computation_graph.visualize()  # graphviz library need to install

print(computation_graph.compute())

