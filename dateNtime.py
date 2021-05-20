# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 10:53:40 2018

@author: Sumanta
"""

# 30-06-2018 00:00:00

#from date "30-06-2018" we will get  --->  day (day name), month (weekend, weekday) , year 

#from time "00:00:00" we will get --->  hh (11), mm(59) , ss(59) , micro second (99999)

"""
How unix calculate the datetime and store

# base date ( it is fix ) 01-01-1970 00:00:00 is fix by system.

lets current date is  01-07-2018 11:34:33
base date is          01-01-1970 00:00:00

if we substrct these two time stamps then we will get 1235791020 (Epoch & Unix Timestamp Conversion)
this is nothing but 01-07-2018 11:34:33

### epochconverter.com 

"""
"""
The datetime module supplies classes for manipulating dates and times in both 
simple and complex ways. 

There are two kinds of date and time objects: “naive” and “aware”.

A naive object does not contain enough information of date/time objects. 

The naive object use to represent Coordinated Universal Time (UTC), local time, 
or time in some other timezone. It is upto a programer how he/she is writing a 
program to get what type of navie object.

An aware object use for algorithmic and political time adjustments, such as 
time zone and daylight saving time information.
To get the required aware objects python provides datetime and time module. 

The datetime module contains date , time  , datetime , tzinfo and timezone classes.

The structure of datetime module
        object
            timedelta
            tzinfo
                timezone
            time
            date
                datetime

In datetime module there is a timezone class which is a subclass of tzinfo class.
The timezone class can represent simple timezones that is EST and EDT timezones.

"""

# when we import datetime module we will get the following constants:

# datetime.MINYEAR ----> The smallest year number allowed in a date or datetime object. 

import datetime as dt

print(dt.MINYEAR)   # output is 1  the minimum year is 1.

# datetime.MAXYEAR ----> The largest year number allowed in a date or datetime object. 

print(dt.MAXYEAR)  # output is 9999  the maximum extend of the year is 9999.



##### date class in datetime module

"""
The date class takes year, month, and day as the ttributes
"""

reqDate = dt.date(2018,6,28)
print(reqDate)

#To get today's date simply use today() in date class

print(dt.date.today()) 

# To know what is the day for today's date

toDay = dt.date.today()

print(toDay.day)


# To know what is the month for today's date
print(toDay.month)

# To know what is the year for today's date
print(toDay.year)

# To know the day of the week  we can use weekday() , it will give any 
# number between 0 to 6 where 0 is monday and 6 is sunday

print(toDay.weekday())


# to get the minimum date in yyyy-mm-dd format 
print(toDay.min) # output is  0001-01-01

# to get the maximum date in yyyy-mm-dd format 
print(toDay.max) # output is 9999-12-31

###### time class in datetime module

"""
The time class has hour, minute, second, microsecond, and tzinfo attributes

"""
print(dt.time(23,35,45,509))  # time(hh,mm,ss,microsec,tzinfo)

theTime=dt.time(23,35,45,509)

# To know the hour of the given time we can use hour
print(theTime.hour)

# To know the minute of the given time we can use minute
print(theTime.minute)

# To know the second of the given time we can use second
print(theTime.second)

# To know the microsecond of the given time we can use microsecond
print(theTime.microsecond)

# to get the minimum time in hr:mm:ss format 
print(theTime.min)   # the out put is  00:00:00

# to get the maximum time in hr:mm:ss:microsecond format 
print(theTime.max)   # the out put is  23:59:59.999999

# The timetuple() method of datetime.date instances returns an object 
# of type time.struct_time. The struct_time is a named tuple object 
# (A named tuple object has attributes that can be accessed by an index or by name).

import datetime
import time
todaysDate = datetime.date.today()
timeTuple = todaysDate.timetuple()
print("time tuple is ",timeTuple)

# The mktime() method is the inverse function of localtime().
# The mktime() method takes timetuple ( all 9 element of timetuple) as an argument and
# returns a floating point number, for compatibility with time().

makeTime = time.mktime(timeTuple)
print("make time is " , makeTime)



####### datetime class in datetime module

"""
The datetime class is a combination of a date and a time class.
The class having  year, month, day, hour, minute, second, microsecond, and tzinfo attributes.

"""

from datetime import datetime

# to know today's date and current time we can use now() method of datetime class
# the output of now() is   yyyy-mm-dd hh:mm:ss:microsec 

print(datetime.now())    # the output is 2018-07-03 15:45:56.818018 

# To get the current time without the date

print(datetime.time(datetime.now())) # the output is 15:52:41.118199


# To get the current date without the time

print(datetime.date(datetime.now())) # the output is 2018-07-03

# to know which date and time of today
toDay = datetime.today()
print(toDay)

# to know which day is today

theDays=['monday','tuesday' ,'wednesday','thersday','friday','suterday' ,'sunday']

toDay = datetime.today()
weekDay = toDay.weekday()
print("Today's date is " , theDays[weekDay])

# to know the current utc ( Universal Time ) date and time we can use utcnow()  
print("current utc time " , datetime.utcnow())

# to get current utc time only

print(datetime.time(datetime.utcnow()))

# to get current utc date only

print(datetime.date(datetime.utcnow()))

# To get the local date and time corresponding to the platform ( where you are accessing)
# which is returned by using time.time() method,  we can use fromtimestamp()

# the fromtimestamp() method takes two arguments timestamp (as an integer) and 
# tz. By default the tz value is None (tz=None)

import datetime
import time
theTime = time.time()  
print(type(theTime))
print( 'theTime:', theTime)  ##   --->  It will give the unix time stamp
localDate =  datetime.date.fromtimestamp(theTime)
print( 'local Date ', localDate )
timeTuple =  localDate.timetuple()
print("time tuple is ", timeTuple)



# we can create new date instances uses the replace() method from an existing date.
# By specifying year, day and month we can create new date.

import datetime

d1 = datetime.date(2017, 3, 15)
print( 'd1:', d1)

d2 = d1.replace(year=2018)
print( 'd2:', d2 )


# The replace() method is not the only way to calculate future/past dates.
# We can use timedeltas() to produce another date by adding and substracting dates.


import datetime

print( "microseconds:", datetime.timedelta(microseconds=1) )
print( "milliseconds:", datetime.timedelta(milliseconds=1) )
print( "seconds     :", datetime.timedelta(seconds=1)      )
print( "minutes     :", datetime.timedelta(minutes=1)      )
print( "hours       :", datetime.timedelta(hours=1)        )
print( "days        :", datetime.timedelta(days=1)         )
print( "weeks       :", datetime.timedelta(weeks=1)        )

# construct a basic timedelta and print it
print (datetime.timedelta(days=365, hours=8, minutes=15))

import datetime

toDay = datetime.date.today()
print( 'Today    :', toDay)

one_day = datetime.timedelta(days=1)  # normal casting is not posible from str to date
#it will give TypeError so we need to use timedelta() to meet our requirement.
print( 'One day  :', one_day)

yesterday = toDay - one_day
print( 'Yesterday:', yesterday)

tomorrow = toDay + one_day
print( 'Tomorrow :', tomorrow)

print( 'tomorrow - yesterday:', tomorrow - yesterday)
print( 'yesterday - tomorrow:', yesterday - tomorrow)

# to reduce or add 7 days from a given date
week_day=datetime.timedelta(weeks=1) 
lastWeekDay = toDay - week_day
print( 'lastWeekDay:', lastWeekDay)

# example of time delat

import datetime
timeDeltaVal = datetime.timedelta(days=365, hours=8 , minutes = 15 )
toDay = datetime.datetime.today()
print("today date and time is " , toDay)
oneYearAdd = toDay + timeDeltaVal
print("one Year Added " , oneYearAdd  )

# Find the day of the year from a given date

from datetime import datetime
dayOfYear = datetime.now()
dayOfYear = datetime.now().timetuple().tm_yday
print(dayOfYear)


# if you already have a date instance and time instance then you can 
# create a datetime instance by using combine().

import datetime

theTime = datetime.time(5)
print( 'the Time  :', theTime )

theTime = datetime.time(1, 3, 40)
print( 'the Time  :', theTime )

theDate = datetime.date.today()
print( 'the Date :', theDate)

newDateTime = datetime.datetime.combine(theDate, theTime)
print( 'newDateTime after combine :', newDateTime )



# unix to pyton date time

import datetime

unix2pydatetime =datetime.datetime.fromtimestamp(1521198578)

print("unix to pyton date time " ,unix2pydatetime )


# We can create the unix time form a given time using maketim()


import datetime , time

theDate = datetime.date(2018, 7 , 15)
print( 'The date is :', theDate)

theTime = datetime.time(10, 25, 40)
print( 'the Time  :', theTime )

newDateTime = datetime.datetime.combine(theDate, theTime)
print( 'newDateTime after combine :', newDateTime )

unix_secs = time.mktime(newDateTime.timetuple())
print("unix time from a given  time is  " , unix_secs)

# Formatting and Parsing datetime 
#The default string representation of datetime object is "YYYY-MM-DDTHH:MM:SS.mmmmmm"
# We can alternate the format by using datetime_object.strftime() method.
#
# The method strptime() parses a time which is in a string representation to a requred format. 

import datetime

format = "%a %b %d %H:%M:%S %Y"

toDay = datetime.datetime.today()
print( 'ISO     :', toDay)

strTime = toDay.strftime(format)
print(type(strTime))
print( 'strftime:', strTime)

dInst = datetime.datetime.strptime(strTime, format)
print(type(dInst))
print( 'strptime:', dInst.strftime(format) )


# %Y  is for four digit notiation of year in strftime()
# %y  is for two digit notiation of year in strftime()

from datetime import datetime
Now=datetime.now()
theYear = Now.strftime("%Y")
print("the year in 4 digit format" , theYear)


from datetime import datetime
Now=datetime.now()
theYear = Now.strftime("%y")
print("the year in 4 digit format" , theYear)

# Similarly we can use  
# %a --- for day (mon , tue, wed )
# %A ---- for day ( monday , sunday ....) 
# %b or %h --- month (jan , feb , ...)
# %B --- month (january , february , ...)
# %m --- month (00---12)
# %e --- day of month (1 to 31)
# %d --- date /day of month (1 to 31)
# %Y ---- year ( 2015,2016,2017  .....)
# %y ---- year (15 , 16 , 17 .......)
# %D – the same as %m/%d/%y
# %H --- (hour --- 00---23  ( 24 hr format ))
# %I --- (hour ---- 00 --- 12 ( 12 hr format))
# %j – day of year (1 to 366)
# %M --- (minits --- 00 --- 59)
# %S --- (sec --- 00 --- 59)
# %x- indicates the local date (07/05/2018 ---- mm/dd/yyyy)
# %X- indicates the local time (15:29:22  ----- hh:mm:ss)
# %T – current time; equal to %H:%M:%S
# %c- indicates the local date and time ( Thu Jul  5 15:34:28 2018 )
# %p or %r -indicates PM / AM
# %u – day of week as a number (1 to 7); Monday=1
# %C – century number (the year divided by 100; range 00 to 99)
# %w – day of week as a decimal; Sunday=0
# %U – week of year, beginning with the first Sunday as the first day of the first week
# %W – week of year; beginning with the first Monday as the first day of the first week
# %Z or %z – time zone/name/abbreviation





