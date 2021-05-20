# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:16:53 2018

@author: Sumanta
"""

"""

Data capturing & storage ( relational databases or input from app or web site)
        |
        |
data exploration & cleaning ( pandas or numpy , matplotlib or seaborn) (80%)
        |
        |
data modelling & predicting (scikit-learn , statsmodels , scipy ) (20%)
        |
        |
Final product (webapp , formal , report ) (tools---saphana , tablu )
   
"""     

"""
Dataset --------------- keep in dataframe  ( it look like table )
        ex -----  day           siteid   accid  energy
                    01-04-18        20      1       20.52
                    02-04-18        20      1       20.55
                    03-04-18        |       |       19.2
                    01-05-18        |       |       21.5
                    |
                    |
                    02-07-18
                    |
                    |
                    31-07-18
----> predict energy for the fist twodays of augest 2018

my data frame is in the above example

siteid , acid is the configurational data

day , energy  where i need to compute

so data points is  day  , siteid , acid , energy ----- of each row

prediction data is the unseen data


Trainset  ----  the data to analyse from historical data to build a 
                model
                
                let consider the data from 01-04-18 --- 02-07-18 
                
                so we can compare the program output  with test set data

Testset  ----  testset is the sample which we can use to verify the  
                model result.

cross validate ----> by suffing of 80% trainset data and 20% testset
                    we can do the cross validataion.

"""

#=================  Create  dataframe reading a csv file =====================

import pandas as pd

fstDF = pd.read_csv("E:/datascienceNml/DataScienceInPy/BasicPythonForDS/Salestransactions.csv",sep='\t').head(10) 
            # head(10) will give the first 10 rows default is 5 rows
print("the data frame is  \n" , fstDF)
 
newDF = fstDF[['Name','Product','Price','Payment_Type']]

print(newDF)


#=================  Create  dataframe =====================
# Pandas DataFrame is two-dimensional heterogeneous data structure with 
# rows and columns label

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }
df= pd.DataFrame(data)
print("the data frame is  \n\n" , df)

#============ assign index to the dataframe ==================

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }
df = pd.DataFrame(data,index=['r1','r2','r3','r4']) #assign index 
print("data frame with new index is \n\n " , df)

df.index=['rr1','rr2','rr3','rr4']      #assign index
print("data frame with new index is \n\n " , df)

#=========== create a new column by adding existing column  =====

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }

df = pd.DataFrame(data)

df['total'] = df['test1'] +  df ['test2'] # adding existing column

print("The new dataframe \n \n ", df)

#=========== Transpose of the dataframe ========================
# transpose --> row to column and column to row conversion 
import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }

df = pd.DataFrame(data)

print("The new dataframe \n \n ", df)

df = df.T # to generated the transpose of the dataframe.
print("The transpose of the dataframe \n \n ", df)


#========== Delete column from the dataframe using  del  or  pop()============

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }
df = pd.DataFrame(data)
df2 = pd.DataFrame(data)

df['total'] = df['test1'] + df['test2']
print("The new dataframe \n \n ", df)

del(df['total'])        #Delete column from the dataframe using del()
print("The dataframe after deleting total column \n \n ", df)

df2.pop('Age')      #Delete column from the dataframe using  pop()
print("\n thedataframe after deleting Age column \n\n" , df2)

#========= change the column name ===========================


import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }

df = pd.DataFrame(data)

print("The new dataframe \n \n ", df)

# to change the column name , the length of the list must be equal to
# the no of columns in the dataframe.
df.columns=['c1','c2','c3','c4']  #change the column name

print("\n The data frame is  \n", df)

#========= slicing the data frame  =========================
"""
There is three different way of slicing the dataframe to get the
required data.
        1) slicing 
        3) using  loc() and iloc()  functions
        4) using boolean operation
"""
#========= slicing ====================================
"""
in case of index slicing the syntax is 
    df[row_list][column_list]
    or
    df[row_list]
    or
    df[column_list]
    
For slicing we can use index for row slicing, as well as the row lable
if row will be having any specific name.

But for the column we need to use column label must be string ( the column 
lable may be an integer ( if column name has not given by the user 
then column label will be integer and we can not use it in slicing) or string).

"""

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 
         'Age':[28,34,29,42],
       'test1':[56,79,75,63],
       'test2':[58,73,53,78]
           }

df = pd.DataFrame(data)

df['total'] = df['test1'] + df['test2']
print("The new dataframe \n \n ", df)

data = df[1:3]   # to retrive the specific rows with all columns 

print("the data recived after slicing \n ", data)

data = df[1:3][['test1','test2']] # to retrive the specific rows then retrive the specific  columns

print("the data recived after slicing \n ", data)


import pandas as pd
data = [
        ['Tom', 'Jack', 'Steve', 'Ricky'], 
        [28,34,29,42],
        [56,79,75,63],
        [58,73,53,78]
            ]

df = pd.DataFrame(data)
print(df)

data = df[1:3][1:]   # [1:] not consider as column lable and will not give req result.
print("the data recived after slicing \n ",data)


import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data , index=['r1','r2','r3','r4'])
print("The data frame is  \n \n "  , df)

data = df[['test1','test2']]  

print("the data recived after slicing \n ",data)

data = df['r2':'r4'][['test1','test2']]  # If we will use row lables 
             # then it will include the last lable but incase of index
             # slicing the last index will be exclude.
print("the data recived after slicing \n ",data)


###########  using loc()
"""
df.loc[[row_list],[column_lable_list]]

to access  all row and multiple column label
 
df.loc[: ,[column_lable_list]]

to access  multiple row and all column

df.loc[row_index_1:row_index_N]

"""
##  to access  single row and single column label

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[[2],['test1']]

print("the data recived after slicing \n ",data)


##  to access  all row and single column label / access single column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[:,'test2']

print("the data recived after slicing \n ",data)

##  to access  single row and multiple column label

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[[1],['Name','test2']]

print("the data recived after slicing \n ",data)

##  to access  multiple row and multiple column label

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[[1,2],['Name','test2']]

print("the data recived after slicing \n ",data)


##  to access  all row and multiple column label

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[:,['Name','test2']]

print("the data recived after slicing \n ",data)

##  to access  single row

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[[1]]   # we will get one row and all column data

print("the data recived after slicing \n ",data)

##  to access  multiple row

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[1:3]  # we will get multiple row and all column data

print("the data recived after slicing \n ",data)

##  to access  all row

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.loc[:]  # we will get all row and all column data

print("the data recived after slicing \n ",data)


#######  using  .iloc() 
"""
df.iloc[[row_index], [column_index]]  ----if column index is not mention 
then the given index in  iloc() is a row index.

to access  multiple row and all column
df.iloc[row_index_1:row_index_N]

"""

##  to access  single row and all column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[[2]]  

print("the data recived after slicing \n ",data)

##  to access  multiple row and all column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[2:4]  

print("the data recived after slicing \n ",data)

##  to access  single row and single column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[[2],[3]]  

print("the data recived after slicing \n ",data)

##  to access  multiple row and multiple column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[[2,3],[0,3]]  

print("the data recived after slicing \n ",data)

##  to access  single row and multiple column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[[2],[0,3,2]]  

print("the data recived after slicing \n ",data)

##  to access  all row and single column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[:,[0]]  

print("the data recived after slicing \n ",data)

##  to access  all row and multiple column

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data = df.iloc[:,[0,3]]  

print("the data recived after slicing \n ",data)


#========== using boolean operation ==============================
"""
Booleaan values are True and False.

df[[list_of_boolean_value_of_Rows]] ----> if we specify True then
                        we will get that row in the result. 
"""

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

data= df[[True,True,False,True]] # 0th , 1st and 3rd row will be the result

print("the data recived after boolean filter \n ",data)


"""
This is helpful when we want to filter the data with some condition
"""

# ex ---> if the total mark is greater than 150 we want those rows

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)

print("The data frame is  \n \n "  , df)

res = df['test1'] + df['test2'] > 150 # it is a series of f,f,T,f

print("the boolean value of the result \n ", res ) 

data = df[res]  # we will get the third record as it is True.
print("student having more than 150 mark is  \n", data) 

#ex2 ---> if we want to find the students whose agv is more than 40% 
# with continuous growth

# sol step1 --> check continuous growth

import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28,34,29,42],
        'test1':[56,79,75,63],'test2':[58,73,53,78]}

df = pd.DataFrame(data)
print("The data frame is  \n \n "  , df)

res = df['test2'] > df['test1']  #  t,f,f,t
resData = df[res]

#step2 ---> get the mean of the students greater than 50

markDF = resData.loc[ : , ['test1','test2'] ] 
resDf = list(markDF.mean() > 50)  # t , t
print(resDf)     # here I have casted because in my dataframe there
                 # 4 columns  but True value i got for two rows 
#resStu =  resData[resDf]
resStu =  resData.loc[resDf]
print("Progressing students are \n" ,resStu)


############# groupby() , merge() and transform() #############

# use of aggregate function groupby  and transform.

import os
import pandas as pd

thePath= 'E:/datascienceNml/DataScienceInPy/DataExploration_HouseSales/data/'
theFile='trainSellingPrice.csv'
def readFileToDF(Path,File):
    file=os.path.join(Path, File)
    fileDF = pd.read_csv(file)
    return fileDF
'''
HouseSalesDF= readFileToDF(thePath,theFile)
HouseSalesDF.iloc[:,-3:].head(15)
'''
HouseSalesDF= readFileToDF(thePath,theFile).iloc[:,-3:].head(15)
HouseSalesDF.groupby('SaleCondition')["SalePrice"].sum() 

'''
            Output will be as below
SaleCondition
Abnorml     14799189
Normal     209892259
Partial     34036469            

''' 

totalOrders=HouseSalesDF.groupby('SaleCondition')["SalePrice"].sum().rename("totalOrders").reset_index()

# merge() ------ merge one dataframe  with other dataframe
HouseSalesDF=HouseSalesDF.merge(totalOrders)

HouseSalesDF.groupby(['SaleCondition','totalOrders']).SaleCondition.unique()

# transform() --- transform returns a different size dataset from our normal groupby functions. 

print(HouseSalesDF.groupby('SaleCondition')["SalePrice"].transform('sum'))

# instade writing a merge(),  we can use below code

HouseSalesDF["totalOrders"] = HouseSalesDF.groupby('SaleCondition')["SalePrice"].transform('sum')

#============= resample() =================================



#============ DatetimeIndex() =============================


#=============== Lambda Operator, filter, reduce, map , Apply and Applymap ============
'''
The lambda operator or lambda function use to create anonymous functions
That is functions without a name. 
Syntax-->
    lambda argument_list: expression
Expression is a mathmatical operation. 
Multiple argument list separated by comma. 

'''
addTen = lambda x: 10+x
addTen(7)

f = lambda x, y : x + y
f(4,6)

'''
filter iterates over each element of a list/sequence and apply the function
on each element. The filter(),  filter out all the elements of a list, 
for which the 'function argument' returns True.

Syntax -->
    filter(function, list)
    

'''

val = (36, 37, 12 ,39)

def theMod(T):
    return T

r = list ( filter(theMod , val))  # filter iterates over each element of sequence and apply the function on each element
print(r)

res = filter(lambda T : T % 2 == 0 , val)
print(list(res))  # filter out all the elements which returns True 

'''
The reduce() apply a function to the sequence and returns a single value.

Syntax -->
    reduce(func, seq)
    
'''

import functools

res = functools.reduce(lambda x,y: x+y, [47,11,42,13])

print(res)



'''
Map iterates over each element of a series/sequence and apply the function
on each element.

syntax -->
    map(func, seq)
        or
    series.map(func)
'''
temp = (36.5, 37, 37.5,39)

def fahrenheit(T):
    return ((float(9)/5)*T + 32)

fahrenheitTmp = map(fahrenheit,temp)

print(list(fahrenheitTmp))

#Map: It iterates over each element of a series/sequence.
df['column1'].map(lambda x: 10+x)		# this will add 10 to each element of column1.
df['column2'].map(lambda x: 'AV'+x)		# this will concatenate "AV" at the beginning of each element of column2 (column format is string).

#Apply: As the name suggests, applies a function along any axis of the DataFrame.
df[['column1','column2']].apply(sum)	# it will returns the sum of all the values of column1 and column2.

#ApplyMap: This helps to apply a function to each element of dataframe.
func = lambda x: x+2
df.applymap(func)	# it will add 2 to each element of dataframe (all columns of dataframe must be numeric type)



########################### Constructing a DataFrame from a dictionary ##################

# the keys will become the column names

import pandas as pd

def bagWords(*args):
    spam_corpus = list(map(lambda x :  x.split(' '), *args))
    #print(spam_corpus)
    
    unique_words = set([ word for doc in spam_corpus for word in doc ])
    #print(unique_words)
    
    word_counts = [ (word, list(map(lambda doc: doc.count(word), spam_corpus)))
    for word in unique_words ]
    #print(dict(word_counts))
    
    bag_of_words = pd.DataFrame(dict(word_counts))
    #print(bag_of_words)
    
    return bag_of_words


lis1 = ["buy viagra", "buy antibody"]

bagDf1 = bagWords(lis1)

print(bagDf1)

lis2 = ["buy time", "hello"]

bagDf2 = bagWords(lis2)

print(bagDf2)


###################### Concat() ##############################
'''
we can concatenate a sequence or mapping of Series, DataFrame, or Panel objects.
Also we can concatenate Dictionary, But when we use Dictionary we need to specify the keys.

Syntax
    pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)
    

'''


df1 = pd.DataFrame ({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
 
 

df2 = pd.DataFrame({'A': ['A6','A9','A1',  'A0'],
                     'B': ['B6','B9','B1',  'B0'],
                     'C': ['C6','C9','C1',  'C0'],
                     'D': ['D6','D9','D1',  'D0']},
                    index=[6, 4, 2, 0])


pd.concat([df1,df2])   # df1 and df2 are in the sequence/list


frames = [df1, df2 ]

pd.concat(frames)

pd.concat([df1, df2], axis=1) # default join is outer and default axis is zero

pd.concat([df1, df2], axis=1, join='inner') # inner join is for intersection 

pd.concat([df1, df2], axis=1, join='inner', join_axes = [df1.index])

            # join_axes is nothing but, wrt which dataframe index we need to consider for joining


df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])


 