# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:18:34 2019

@author: Sumanta
"""

'''
To run multiple operations concurrently in the same process space we use 
multithreading.
To perform multithreading In python we need to import  'threading' module.

init method of threading module is as below.
    def __init__(self, group=None, target=None, name=None,
             args=(), kwargs=None, *, daemon=None):

'''

import threading

def f():
    print('thread function')
    return

if __name__ == '__main__':
    for i in range(3):
        t = threading.Thread(target=f)
        # The target is the callable object to be invoked by the run() method.
        # The start() starts the thread's activity and arranges the object's for run() method
        t.start()
        


# The 'args' takes  argument tuple and use those argument to invoke target

import threading

def f(id):
    print('thread function %s' %(id))
    return

if __name__ == '__main__':
    for i in range(3):
        t = threading.Thread(target=f, args=(i,)) 
        t.start()


'''
In multithreading, when many threads trying to access the same piece of data
in that case we need to mange threads in an organized manner.
So we can do this by using synchronization primitives like Locks, RLocks, 
Semaphores, Events, Conditions and Barriers. 

'''


# Identifying threads by naming the threads
# Create log for the threads 
# Create Daemon thread 
# Use of join() and  isAlive() method 

'''
For naming a thread we need to use 'name' argumant in Thread() method

To create log for the thread we should import 'logging' package and 
should use ' logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)  ' method

For creating a daemon thread we should use 'setDaemon()' method. Usually  
main program implicitly waits until all other threads complete their work. 
But sometime we need to release the thread as daemon thread without 
blocking the main method.  If thread die in the middle of its work then 
daemon thread will execute without losing or corrupting data. 

The 'join()' method blocks the calling thread indefinitely, until the threads
are terminated - either normally or through an unhandled exception.
In 'join()' we can pass a timeout argument (float point number for seconds),
So that join() will block the thread for that many seconds and then thread
become inactive.

when we pass a timeout argument in 'join()' it returns 'None'. So by using
'isAlive()' method we can detect the thread is alive or not.

'''



import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

def nondaemonFun():
    logging.debug('Starting')
    logging.debug('Exiting')

def daemonFunc():
    logging.debug('Starting')
    time.sleep(5)
    logging.debug('Exiting')

if __name__ == '__main__':

    n = threading.Thread(name='non-daemon', target=nondaemonFun)
    d = threading.Thread(name='daemon', target=daemonFunc)
    d.setDaemon(True)

    d.start()
    n.start()

    d.join(3.0)     # timeout in 3 second  
    print('daemonFunc d.isAlive()', d.isAlive()) # check thread is alive or not
    n.join()
    
    
# threading.enumerate()  and threading.current_thread() 
'''
The daemon thread is completed or not can be handle explicitly, But python 
provide 'threading.enumerate()' method to do that job instade of doing that
explicitly.

    threading.enumerate() returns a list of all Thread objects whcih are
currently alive. It excludes terminated threads and threads that have not started.
   
    threading.current_thread() return a list of all daemon threads and a dummy
thread object.
    
    So by using these two methods we can track the daemon threads are completed
or not.
'''   


import threading
import time
import logging
import random

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

def f():
    #ct = threading.currentThread()
    r = random.randint(1,10)
    logging.debug('sleeping %s', r)
    time.sleep(r)
    logging.debug('ending')
    return

if __name__ == '__main__':
    for i in range(3):
        t = threading.Thread(target=f)
        t.setDaemon(True)
        t.start()

    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
    logging.debug('joining %s', t.getName())
    t.join()
        
        

'''
Once a thread object is created, its activity must be started by calling 
the thread's start() method.
The start() method invokes run() method in a separate thread of control.

Once thread's activity is started then thread is considered as 'alive'.
Thread stops being alive when its run() method terminates either normally, 
or treminated by raising an unhandled exception.

The is_alive() method tests whether the thread is alive or not.

'''

'''
The Thread class represents an activity that is run in a separate 
thread of control. 
There are two ways to specify the activity: 
    by passing a callable object to the constructor __init__, 
        or 
    by overriding the run() method in a subclass.

 by passing a callable object to the constructor __init__
================================================================    
If the subclass overrides the constructor, it must make sure to invoke 
the base class constructor (threading.Thread.__init__(......)  or
super().__init__(......) ) before doing anything else to the thread.

In other words, override the __init__() and run() methods of parent class
to specify activity.
'''

import threading
import time

class myThreadDemo(threading.Thread):
    # Override the __init__() of parent class
    def __init__(self,i):
        threading.Thread.__init__(self)
        self.k =  i
    
    # Override run method
    def run(self):
        print("the value send to k " , self.k )
        
        
if __name__ == '__main__':
    th1 = myThreadDemo(1)
    th1.start()
    th1.join(2.0)  # timeout for 2 sec
    
    print("good by ",th1.getName())
    th1.join()
    
    th2 = myThreadDemo(2)
    th2.start()
    th2.join(2.0)  # timeout for 2 sec
    
    print("good by ",th2.getName())
    th2.join()


# Passing args to an userdefined thread

'''
when we create an instance for user-defined thread we can use 
"*args and **kwargs" arguments in thread constructors.
So value passed to these arguments can be access through arg() and kwargs()
'''



import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

class MyThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None):
        	super().__init__(group=group, target=target, name=name)
        	self.args = args
        	self.kwargs = kwargs
			#return

    def run(self):
        	logging.debug('running with %s and %s', self.args, self.kwargs)
        	return

if __name__ == '__main__':
    for i in range(3):
        	t = MyThread(args=(i,), kwargs={'a':1, 'b':2})
        	t.start()
			#t.join(2.0)
            
			

 
    
        
            
            
            
            
            
            
            
            