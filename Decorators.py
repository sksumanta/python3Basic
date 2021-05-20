# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:36:28 2019

@author: Sumanta
"""

# Before moving into decorators in python, Lets take a look on funcitons.

# We know everything in Python is an object:  

def  hi():
    return "this is a function"

theFunc =  hi   # we are just putting 'hi' into the 'theFunc' variable

print(hi())

print(theFunc())

del hi

print(hi())   # Here we will get "NameError: name 'hi' is not defined" error

print(theFunc()) # But this function theFunc() will give output 


# Let’s take a look 'functions within functions' nested function

def  funWithinFun():
    print( "this is a funWithinFun() function" )
    
    def otherOne():
        return  "in first nested function"

    def otherTwo():
        return "in sec nested function"
    
    print(otherOne())
    print(otherTwo())
    print("back to funWithinFun" )

funWithinFun()

# otherOne()  # it will throw error "NameError: name 'otherOne' is not defined"

# Returning functions from a function

def  returnFun(n=5):
    print( "Returning functions from a function" )
    
    def returnFstOne():
        return  "Returning first nested function"

    def returnSecOne():
        return "Returning sec nested function"
    
    if n==5:
        return returnFstOne
    else:
        return returnSecOne

fst = returnFun()

print(fst())

sec = returnFun(6)

print(sec())


# Pass function as an argument to another function

def passAfunc(func):  
    print(func())
    
passAfunc(returnFun())  # here  one function is passed an argument to other function

passAfunc(returnFun(8))



        ###############   Decorator ##################
'''
Decorators allow us to wrap another function in order to extend the behavior 
of wrapped function, without permanently modifying it.
'''

def firstDecoratorFunc(funct):
    def wrappedFunction():
        print(" program is inside the wrapper function ")
        
        funct()
        
        print( "wrapped ", funct , " executed")
    return wrappedFunction

def aFunctionReqDecorator():
    print("the function which needs some decoration ")
    
# now call the function  'aFunctionReqDecorator' we will get below result
    
aFunctionReqDecorator()
    
# now wrapped the function 'aFunctionReqDecorator' by 'wrappedFunction'

aFunctionReqDecorator = firstDecoratorFunc(aFunctionReqDecorator)

# now call again 

aFunctionReqDecorator() 


'''
Python gives the sortest way to create a Decorator by using  @ 

'''
# so we modify the same program using @ 

@firstDecoratorFunc
def aFunctionReqDecorator():
    print("the function which needs some decoration ")


# call the function
    
aFunctionReqDecorator()

'''
 So  '@firstDecoratorFunc'  is the sort form of 
 'aFunctionReqDecorator = firstDecoratorFunc(aFunctionReqDecorator)'
 
'''

''' 
 But we have a problem in the above decorator code, if we execute 
print( aFunctionReqDecorator.__name__ )  give the function name as an output 

In this case we will get 'wrappedFunction' as function name. 
which should be  'aFunctionReqDecorator'

'''

print( aFunctionReqDecorator.__name__ ) 


'''
To fix this error python gives one decorator function '@wraps' which is in
functools package of python.
'''

# now the code will be as below

from functools import wraps
def firstDecoratorFunc(funct):
    @wraps(funct)
    def wrappedFunction():
        print(" program is inside the wrapper function ")
        
        funct()
        
        print( "wrapped ", funct , " executed")
    return wrappedFunction

@firstDecoratorFunc
def aFunctionReqDecorator():
    print("the function which needs some decoration ")


# call the function
    
aFunctionReqDecorator()

print( aFunctionReqDecorator.__name__ )   # output 'aFunctionReqDecorator'



            ######## Real time use case of Decorators  ############

# Example-1  someone is authorized to use application or not 

from functools import wraps

def currentUserId():
    # this function returns the current logged in user id 
    id = 4    # lets say user id is 4  for the executing below code
    return id

def getPermissions(iUserId):
    # returns a list of permission strings ( use select statement to get it)
    return 'admin'

lPermissions=['admin','premiumMember']
def premiumMember(fn):
    @wraps(fn)
    def returnFn(*args,**kwargs):
        permissions = getPermissions(currentUserId())
        if permissions in lPermissions:
            return fn(*args,**kwargs)
        else:
            raise Exception("Not allowed")
    return returnFn


@premiumMember
def deleteUser(iUserId):  # need to write code to validate the userid, here no validation in deleteUser()
    return 'deleted'

deleteUser(5)


# Example-2 Write the output to a log file 

from functools import wraps

def write2LogFile(logfil = 'logfile.txt'):
    def logDecorator(fn):
        @wraps(fn)
        def writeLog(*args , **kargs):
            logString = fn.__name__ + " is called"
            print(logString)
            with open(logfil , 'a') as lf:
                lf.write(logString + '\n')
        return writeLog
    return logDecorator

@write2LogFile()   # it will create the default file  "logfile.txt"
def myfunc1():
    pass

myfunc1()

@write2LogFile(logfil='func2.log')   # it will create the file  "func2.log"
def myfunc2():
    pass        

myfunc2()



'''
Instead of a function we can use class to build decorators

'''

from functools import wraps

class writeLog(object):
    
    _logFile = 'logfile.txt'  # global variable _logFile
   
    def __init__(self, fn):    # init will create the instance of a class
        self.fn = fn    
    
    def __call__(self, *args):  # instance of class called as fucntion the call() will execute automatically
        
        logStr = self.fn.__name__ + " was called"
        print(logStr)
        
        with open(self._logFile , 'a') as lf:
            lf.write(logStr +  '\n')
            
        self.notify()  # send a notification
        
        return self.fn(*args)
    
    def notify(self):
        pass
    
    
@writeLog     
def myFunc1():
    pass

myFunc1() 


writeLog._logFile='func2.log'

@writeLog     
def myFunc2():
    pass

myFunc2() 



# Example-2  inherite the writeing log 


class writeLog(object):
    
    _logFile = 'logfile.txt'  # global variable _logFile
   
    def __init__(self, fn):    # init will create the instance of a class
        self.fn = fn    
        
    def __call__(self, *args):
        with open(self._logFile , 'r') as lf:
            line = lf.readlines()
        return line
        
    def notify(self):
        pass
    
    
    
class email_logit(writeLog):
    '''
    A logit implementation for sending emails to admins
    when the function is called.
    '''
    def __init__(self,email='admin@myproject.com', *args, **kwargs):
        self.email = email
        super(email_logit, self).__init__(self, *args, **kwargs)

    def notify(self):
        # Send an email to self.email
        # Will not be implemented here
        pass

@email_logit
def myFunc2():
    pass


myFunc2()



