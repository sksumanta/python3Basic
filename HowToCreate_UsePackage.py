# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:19:27 2019

@author: Sumanta
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:29:24 2019

@author: Sumanta
"""



            #   How to Create and Access a Python Package

'''
Packages are a way of structuring many sub packages and modules in 
well-organized hierarchical order. 

Generaly, In python we create a directory which act a package. 
In python a directory act as a package when dirctory contains "__init__.py"
file.
Inside a package directory we can create other modules and sub-packages.

            Step to create  package in python
            
*) First, we create a directory and give it a package name, preferably 
        related to its operation.    # EX --> HowToCreatePackage is a directory

*) Then we put the classes and the required functions in it.
            Ex --> cars directory , Bmw.py , Audi.py , Nissan.py

*) Finally we create an __init__.py file inside the directory, (HowToCreatePackage ) 
        to let Python know that the directory is a package.

'''

            # How to use the package that we created.
'''

Simply create a module or python pogram file where you want to use the package

Then import the class from the package which you need to use 

'''

import HowToCreatePackage
#from HowToCreatePackage  cars.Bmw 

from HowToCreatePackage.cars import Bmw
from HowToCreatePackage.cars import Audi
from HowToCreatePackage.cars import Nissan

# Create an object of Bmw class & call its method 
ModBMW = Bmw() 
ModBMW.outModels() 
   
# Create an object of Audi class & call its method 
ModAudi = Audi() 
ModAudi.outModels() 
  
# Create an object of Nissan class & call its method 
ModNissan = Nissan() 
ModNissan.outModels() 

