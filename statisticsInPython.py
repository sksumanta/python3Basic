# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:05:51 2019

@author: Sumanta
"""

'''
Python provide 'statistics' module to perform statistical operation.

mean() :- This function returns the mean or average of the data passed 
in its arguments. If passed argument is empty, StatisticsError is raised.

mode() :- This function returns the the number with maximum number 
of occurrences. If passed argument is empty, StatisticsError is raised.

'''

import statistics

li = [1, 2, 3, 3, 2, 2, 2, 1,4,4]

# mean() to calculate average of list elements 
print ("The average of list values is : ",end="") 
print (statistics.mean(li)) 

# mode() to print maximum occurring of list elements 
print ("The maximum occurring element is  : ",end="") 
print (statistics.mode(li)) 


'''
median() :- This function is used to calculate the median, i.e middle element 
of data after sorting the data in ascending order. If passed argument is 
empty, StatisticsError is raised.

median_low() :- This function returns the median of data in case of odd number 
of elements, but in case of even number of elements, returns the lower of two 
middle elements. If passed argument is empty, StatisticsError is raised.

median_high() :- This function returns the median of data in case of odd number 
of elements, but in case of even number of elements, returns the higher of two 
middle elements. If passed argument is empty, StatisticsError is raised.

median_grouped() :- This function is used to compute group median, i.e 50th 
percentile of the data after sorting the data in ascending order. If passed 
argument is empty, StatisticsError is raised.
'''

import statistics

li = [1, 2, 3, 3, 2, 2, 2, 3,4,4]

print("printing sorted list", sorted(li))

# median() to print median of list elements 
print ("The median of list element is : ",end="") 
print (statistics.median(li)) 
  
# median_low() to print low median of list elements 
print ("The lower median of list element is : ",end="") 
print (statistics.median_low(li)) 
  
# median_high() to print high median of list elements 
print ("The higher median of list element is : ",end="") 
print (statistics.median_high(li)) 

# median_grouped() to calculate 50th percentile 
print ("The 50th percentile of data is : ",end="") 
print (statistics.median_grouped(li))

'''
variance() :- This function calculates the sample variance, assuming data is 
a part of population. If passed argument is empty, StatisticsError is raised.

    sum of Xi - (mean of X) square  devided by N-1 

pvariance() :- This function computes the variance of the entire population.
If passed argument is empty, StatisticsError is raised.

    sum of Xi - (mean of X) square  devided by N
    
'''

import math as m
import statistics 
  
# initializing list 
li = [1.5, 2.5, 2.5, 3.5, 3.5, 3.5] 
  
# using variance to calculate variance of data 

'''
means = statistics.mean(li)
Xi_meansSquare  = [(ele-means)**2 for ele in li ]
res=0
for ele in Xi_meansSquare:
    res=res+ele 
sumOfXi_meansSquare=res
sampleVariance = sumOfXi_meansSquare / (len(li)-1)
print ("The sample variance of data is : ",sampleVariance,end="")  
'''

print ("The sample variance of data is : ",end="") 
print (statistics.variance(li)) 
  
# using pvariance to calculate population variance of data 
print ("The population variance of data is : ",end="") 
print (statistics.pvariance(li)) 

'''
stdev() :- This function returns the standard deviation ( square root 
of sample variance ) of the data. If passed argument is empty, 
StatisticsError is raised.

pstdev() :- This function returns the population standard deviation 
 ( square root of population variance ) of the data. If passed argument 
 is empty, StatisticsError is raised.
 
'''

import statistics 
  
# initializing list 
li = [1.5, 2.5, 2.5, 3.5, 3.5, 3.5] 
  
# using stdev to calculate sample standard deviation of data 
print ("The sample standard deviation of data is : ",end="") 
print (statistics.stdev(li)) 
  
# using pstdev to calculate population standard deviation of data 
print ("The population standard deviation of data is : ",end="") 
print (statistics.pstdev(li)) 


'''
P-value is less than or equal to 'Alpha' we will reject null hypothesis.
'''

'''
T-test
========
T-test tells us whether a sample differ significantly from the population.
It also talks about two samples- whether they’re different.

In other words, it gives us the probability of difference between populations.

There are three main types of t-test:
1. An Independent Samples t-test compares the means for two groups.
2. A Paired sample t-test compares means from the same group at different times (say, one year apart).
3. A One sample t-test tests the mean of a single group against a known mean.

'''


# How to perform a 2 sample t-test?  ttest_ind()
'''
Lets us say we have to test whether the height of men in the population is 
different from height of women in general.

'''
#step 1
# Determine a null and alternate hypothesis.

# Null Hypothesis --> Hight of man and hight of women are same
# Alternate Hypothesis --> Hight of man and hight of women are different

#step 2
# Collect sample data

# No of sample for men (X1 column) is Nx1
# No of sample for women (X2 column) is Nx2

# so Degree of  freedom is    (Nx1 + Nx2) - 2
# the significance level is 'Alpha'  0.05

# Step 3
# State deceission rule

# It is a two tail test, as age of women may greater than or smaller than 
# the age of men

# step 4
# Calculate the t-statistic for two sample t-test

# t = ( Mx1 - Mx2 ) / sqrt( ( (Sx1)^2 /Nx1 ) + ( (Sx2)^2 /Nx2 ) ) 

# Mx1 , Mx2 are the mean  and Sx1, Sx2 are the sample standard deviation

# (Sx1)^2 and (Sx2)^2 are the variance 
 
# step 5
# Calculate the critical t-value from the T-table

# form the T-table we will get T-critical but in python we will use sciPy package


#step 6
# calculte the t-statistic value 


## Import the packages
import numpy as np
from scipy import stats


## Define 2 random distributions for t-test ttest_ind()
#Sample Size
N = 10

# get 10 random samples using Gaussian distributed data with mean = 2 and var = 1
sample1 = np.random.randn(N) + 2
print(sample1)

# get 10 random samples using Gaussian distributed data with with mean = 0 and var = 1
sample2 = np.random.randn(N)
print(sample2)

#Calculate variance “Delta Degrees of Freedom --> ddof ” to get standard deviation

variance1 = np.var(sample1,ddof=1) 
print("variance1", variance1)
variance2 = np.var(sample2,ddof=1) 
print("variance2" , variance2)

'''

#for two sample t-test std deviation is sqrt( ( (Sx1)^2 /Nx1 ) + ( (Sx2)^2 /Nx2 ) ) 

stdDeviation = np.sqrt((variance1 + variance2)/2)
print("for two sample t-test std deviation", stdDeviation )


## Calculate the t-statistics
t_statistics  = (sample1.mean() - sample2.mean())/(stdDeviation*np.sqrt(2/N))

'''

## Calculate the t-statistics
t_statistics = (sample1.mean() - sample2.mean())/np.sqrt((variance1+variance2)/N)
print("t_statistics",t_statistics)

#Degrees of freedom
dfd = 2*N - 2

#p-value after comparison with the t_statistics 
pVal = 1 - stats.t.cdf(t_statistics,df=dfd)
print("p-Value = " + str(2*pVal))

############################################################################

## Cross Checking with the internal scipy function
t_statistics, pValue = stats.ttest_ind(sample1,sample2)
print("t-statistics = " + str(t_statistics))
print("p-Value = " + str(pValue))


# One-sample T-test with Python ttest_1samp()

# T-statistics = Xmean - PopulationMean / sqrt( (Sx)^2 / (N-1) )

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
np.random.seed(6)

# Generate Poisson discrete random variable rvs(mu, loc=0, size=1, random_state=None)
'''
# The argument 'loc' is the N-dimensional reference point of the distribution, 
# that centroid being chosen appropriately to the function. The default value 0, 
# and is only changed if your application starts at something other than 0.

mu is the mean of the function.

size is the sample size.

'''

population_ages1=stats.poisson.rvs(loc=18,mu=35,size=1500)
#print(population_ages1)
population_ages2=stats.poisson.rvs(loc=18,mu=10,size=1000)
#print(population_ages2)
population_ages=np.concatenate((population_ages1,population_ages2))
#print(population_ages)
# population_ages mean is 
population_ages.mean()

gujarat_age_sample1=stats.poisson.rvs(loc=18,mu=30,size=30)
gujarat_age_sample2=stats.poisson.rvs(loc=18,mu=10,size=20)
gujarat_age_sample =np.concatenate((gujarat_age_sample1,gujarat_age_sample2))
#gujarat_age_sample mean is 
gujarat_age_sample.mean()

t_statistics , p_value = stats.ttest_1samp(a=gujarat_age_sample ,popmean=population_ages.mean())
print("t_statistics"  ,t_statistics )
print("p_value" , p_value)


# Paired T-test With Python ttest_rel()

# When you want to check different samples from the same group 


np.random.seed(11)

before=stats.norm.rvs(scale=30,loc=250,size=100)

after=before+stats.norm.rvs(scale=5,loc=-1.25,size=100)

weight_df=pd.DataFrame({"weight_before":before,
                     "weight_after":after,
                     "weight_change":after-before})
weight_df.describe()

stats.ttest_rel(a=before,b=after)


                        ##### correlation and covariance in python
                        
# The relationship between two variables can be measured, using correlation and covariance.
# correlation is preferred over covariance, because it remains unaffected by 
# the change in location and scale, and can also be used to make a comparison 
# between two pairs of variables.

# when the correlation coefficient is zero, covariance is also zero then two 
    #variables have linear relationship.

'''
Benefits of correlation 
============================
How strongly two random variables are related known as correlation.
The value of correlation takes place between -1 and +1.
correlation is not influenced by the change in scale.


Predicting one quantity from another
Discovering the existence of a causal relationship
Foundation for other modeling techniques

Covariance
===============
Covariance indicate which two random variables change occur at the same time.
The value of covariance lies between -∞ and +∞.
Covariance is affected by the change in scale, i.e. if all the value of one 
    variable is multiplied by a constant and all the value of another variable 
    are multiplied, by a similar or different constant, then covariance is changed.
when the data is standardized and covariance is calculated then we will get same 
    result as correlation.
'''

# E:/datascienceNml/DataScienceInPy/CorrelationProg/



import pandas as pd
import numpy as np

TravelTimeDF = pd.read_csv(
    filepath_or_buffer = 'E:/datascienceNml/DataScienceInPy/CorrelationProg/AnalyiseTravelTime.csv')


#Remove all the nan from the data

TravelTimeDF.dropna(how = "all", inplace = True)
print(TravelTimeDF.tail())

TravelTimeDF.describe()

# correlation using corr()

TravelTimeDF.corr()  # from output MilesTraveled, NoOfDeliveries,TravelTime are highly correlated

# covariance using cov()

TravelTimeDF.cov()  # from output MilesTraveled, NoOfDeliveries,TravelTime are highly correlated


# Chi-Squared Test (Pearson's Chi-Squared Test)
#========================================================

# Tests whether two categorical variables are related or independent.

# Interpretation

    # H0: the two samples are independent.
    # H1: there is a dependency between the samples.

'''
Consider an example, Gender and interest wrt to the gender, 
The interest may have the labels 'science', 'math', or 'art'.
So the observation has  two categorical variables.
    Sex	  Interest
    ====  ==========
    Male    	Art
    Female  	Math
    Male 	   Science
    Male    	Math
    ...
    ...
As the number of observations for a category (such as male and female) may 
or may not the same. So we calculated the frequency of observations for each group.

Counts might look as follows

        Science,	    Math,	 Art
Male         20,      30,    15
Female       20,      15,    30

Then determining whether the division of the groups, called the observed 
frequencies, matches the expected frequencies.

X^2 give a measure of distance between observed and expected frequencies.

When observed frequency is far from the expected frequency, the corresponding 
term X^2 in the sum is large.
When observed frequency is close to the expected frequency, this X^2 term is small. 

If Statistic >= Critical Value: significant result, reject null 
                                hypothesis (H0), means samples are dependent.
                                
If Statistic < Critical Value: not significant result, fail to reject null 
                                hypothesis (H0), means samples are independent.

In Python 'chi2_contingency()' SciPy function use for Pearson’s chi-squared test.

This function returns, the calculated statistic and p-value for interpretation 
and also calculated degrees of freedom and table of expected frequencies.

===============================================================================

        Science,	  Math,	 Art	Row_Total
Male         25,      30,    15		70
Female       20,      20,    30		70

Col_Total	 45,	  50	 45		All_total = 140

    		Observed 	Expected 	   	Observed 	Expected 		Observed 	Expected
Male         25,   (70x45)/140=22.5		30,    	(70x50)/140=25	  15		(70x45)/140=22.5
Female       20,   (70x45)/140=22.5		20,    	(70x50)/140=25	  30		(70x45)/140=22.5

Formula for Calculating χ2
- - - - - - - - - - - - - - -
χ2 =  sum of ( (Observed - Expected)^2 / Expected )
		
Male         (25-22.5)^2/22.5=0.277		(30-25)^2/25=1	  (15-22.5)^2/22.5=2.5
Female       (20-22.5)^2/22.5=0.277		(20-25)^2/25=1	  (30-22.5)^2/22.5=2.5
 
χ2 = 0.277 +  0.277 + 1 + 1 + 2.5 + 2.5 = 7.554

In our example, R = 2 and C = 3, so degree of freedom is df = (2 – 1)(3 – 1) = 2
For  alpha level of 0.05 and df 2  the  critical χ2 value of 5.99

Therefore, we rejected the null hypothesis because our χ2 value > critical χ2 value 
 
===============================================================================
'''

# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# contingency table
table = [	[25,      30,    15],
			   [20,      20,    30]  ]

print("Observed data is \n",table)

calculatedChiVal, p, dof, expected = chi2_contingency(table)

print('degree of freedom = %d' % dof)
print("expected value for the table \n" , expected)


# interpret test-statistic / conclusion of the test
prob = 0.95
chiCritical = chi2.ppf(prob, dof)
print(' probability=%.3f, \n Chicritical=%.3f, \n observ/calculatedChiVal=%.3f' % (prob, chiCritical, calculatedChiVal))

if abs(calculatedChiVal) >= chiCritical:
	print('Dependent (reject null hypothesis H0)')
else:
	print('Independent (fail to reject null hypothesis H0)')
    
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p_value=%.3f' % (alpha, p))

if p <= alpha:
	print('Dependent (reject null hypothesis H0)')
else:
	print('Independent (fail to reject null hypothesis H0)')



# Anova ( Analysis of Variance Test )
#=============================================
# In anova we tests whether the means of two or more independent samples 
# are significantly different.

'''
In the ANOVA framework, the independent variables are called as factors 
and each category/group within the independent variables are called a level.

Assumptions For Anova
---------------------------
Each sample are identically distributed having almost equal means.
Each sample have the same variance.

Interpretation
--------------------
H0: the means of the samples are equal.
H1: one or more of the means of the samples are not equal.

Types of Anova
-----------------------
one-way ANOVA -->  if you want to compare the means of two independent 
groups/variables on a single/dependent variable, you can use independent 
samples t-test or  one-way ANOVA. The t-test will produce t-value where
from one-way Anova will produce f-ratio.

The t test and the one-way ANOVA produce identical results when there are 
only two groups (one independent and one dependent) being compared.

To conduct a one-way ANOVA, you need to have one categorical (or nominal) 
independent variable having at least two independent groups in that 
categorical independent variable. Also one continuois dependent variable.

    Team        Score_rating
------------    -------------
    India           5
    Pak             3
    America         3           # here Team is the independent variable
    India           4           # having three differnt categories
    Pak             2           # 
    India           4           # Score rating is the dependent variable
    America         4           # having continuous data
    ......

two-way Anova -->  In case of two-way ANOVA there will be more than two  
categorical independent variables having at least two independent groups 
in that categorical variable and one dependent variable.

Fert  Water   Yield
A     High    27.4
A		High	   33.6
A		High	   29.8
B		High	   30.2
B		High	   30.8
B		High	   26.4
A		Low		32
A		Low		32.2
B		Low		26.8
B		Low		23.2
B		Low		29.4
....
....

In a one-way ANOVA or in a two-way ANOVA tests if there is a difference 
between the means, but it does not tell which groups differ. So to get that
information, we need to use post-hoc testing.

For Anova  there is three types of 'sum of squares' that should be considered.
SST =  Sums of Squares Total, SSC =  column Sums of Squares   and
SSE = Sums of Squares Error (Residual)  

F-ratio = mean square between / mean square error  = MSC/MSE

'''

# Example of one-way ANOVA
# ===============================

import pandas as pd
import numpy as np
from scipy import stats 
from scipy.stats import f_oneway

aDf = pd.read_csv('E:/datascienceNml/DataScienceInPy/BasicPythonForDS/difficile_oneWayAnova.csv')

aDf['medi_dose'] = aDf['medicine_dose'].replace({1: 'low', 2: 'medium', 3: 'high'})

#aDf.groupby(['medicine_dose']).reaction_rate.sum()

# mean per group

aDf.groupby(['medi_dose']).reaction_rate.mean()

# standard deviation 

aDf['reaction_rate'].std()

aDf.groupby(['medi_dose']).reaction_rate.std()

# standard error

aDf['reaction_rate'].sem() # sem() / tsem() method gives standard error or mean standard error 

aDf.groupby(['medi_dose']).reaction_rate.sem()

# stats.f_oneway(data_group1, data_group2, data_group3, data_groupN)  

stat , pVal = f_oneway(aDf['reaction_rate'][aDf['medi_dose'] == 'high'], 
             aDf['reaction_rate'][aDf['medi_dose'] == 'low'],
             aDf['reaction_rate'][aDf['medi_dose'] == 'medium'] )

print("f-test statistics  ", stat)
print(" pVal ", pVal)

# Tests whether the distributions of two paired samples are equal or not.
# so we will use   wilcoxon  test
 
from scipy.stats import wilcoxon

stat, p = wilcoxon(aDf['medicine_dose'] , aDf['reaction_rate'])

print("pvalue " , p )





# Example of two-way / n-way ANOVA
# ===============================

# https://www.analyticsvidhya.com/blog/2018/01/anova-analysis-of-variance/

# python does not support two/N-way anova so we need to use manual logic

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf  # to calculate p-value 
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison as mc

aDf = pd.read_csv('E:/datascienceNml/DataScienceInPy/BasicPythonForDS/crop_yield_anova.csv')

aDf.shape
aDf['Yield'].describe()  # Yield  meaning is agricultural productivity

aDf['Yield'].sem() # sem() / tsem() method gives standard error or mean standard error 

aDf.groupby(['Fertilizer']).Yield.mean()

aDf.groupby(['Fertilizer','Water']).Yield.mean()

aDf.groupby(['Fertilizer','Water']).Yield.sem()

# To calculate p-value and f-statistics we will use  ols()

model =  smf.ols(formula='Yield ~ C(Fertilizer)*C(Water)', data=aDf ) 

# “~” separates the left-hand side of the model from the right-hand side
# C() is use for catagorical variable
# “:”  or “*” will include the individual columns, were multiplied together
# and gives result with interaction of the factors. 

''' 
# The “-” sign can be used to remove the intercept from a model 
model =  smf.ols(formula='Yield ~ C(Fertilizer)*C(Water) -1 ', data=aDf ) 

'''
model_1_Result =  model.fit()
interactionResult =  model_1_Result.summary()

# The R-Square will give use the variance, R-square will affect 
# by number of independent variable and number of sample/observation in the 
# independent variable. 

print(interactionResult)

# the output
'''
          OLS Regression Results                            
==============================================================================
Dep. Variable:                  Yield   R-squared:                   0.435
Model:                            OLS   Adj. R-squared:              0.330
Method:                 Least Squares   F-statistic:                 4.112
Date:                Sat, 06 Apr 2019   Prob (F-statistic):          0.0243
Time:                        10:32:32   Log-Likelihood:             -50.996
No. Observations:                  20   AIC:                         110.0
Df Residuals:                      16   BIC:                         114.0
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
        										   coef   std err   t      P>|t|   [0.025   0.975]
------------------------------------------------------------------------------
Intercept                           31.8000  1.549  20.527   0.000   28.516  35.084
C(Fertilizer)[T.B] 					     -1.9600  2.191  -0.895   0.384   -6.604   2.684
C(Water)[T.Low]					        -1.8000  2.191  -0.822   0.423   -6.444   2.844
C(Fertilizer)[T.B]:C(Water)[T.Low]  -3.5200  3.098  -1.136   0.273  -10.088   3.048
==============================================================================
Omnibus:                        3.427   Durbin-Watson:                2.963
Prob(Omnibus):                  0.180   Jarque-Bera (JB):             1.319
Skew:                          -0.082   Prob(JB):                     0.517
Kurtosis:                       1.752   Cond. No.                     6.85
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# Summary
=================
Intercept means -------->(expected mean value of Y when all X=0)
from the coef we come to know
           Y = ß0 + ß1*x1 +ß2*x2  ==>  31.8000 + (-1.9600)x1 + (-1.8000)x2
           
For every 1 unit increase in Fertilizer, Yield decreases by 1.9600 (holding Water constant)
For every 1 unit increase in Water, Yield decreases by 1.8000 (holding Fertilizer constant)
At 0 weight and 0 cylinders, we expect Yield to be 31.8000


C(Fertilizer)[T.B]:C(Water)[T.Low] ===> the interaction term
From the interaction we will come to know which factor is influencing 
more impact to other factor " here    Fertilizer-->B  influence by Water--->Low

The Durban-Watson  ==> detect the correlation
Jarque-Bera  ==> detect the normality
Omnibus   ==> detect the  variance
Condition Number (Cond. No.)    ==> detect the   multicollinearity

'''

# calculat the f-value and p-value of the model

identityMatrix = np.identity(len(model_1_Result.params))
identityMatrix = identityMatrix[1:,:] 
print(model_1_Result.f_test(identityMatrix))

# From ftest we come to know the model is significant 

# check the interaction is significant or not

res = sm.stats.anova_lm(model_1_Result, typ= 2)
print(res)

'''
                        sum_sq    df         F    PR(>F)
C(Fertilizer)            69.192   1.0  5.766000  0.028847
C(Water)                 63.368   1.0  5.280667  0.035386
C(Fertilizer):C(Water)   15.488   1.0  1.290667  0.272656
Residual                192.000  16.0       NaN       NaN

From   C(Fertilizer)[T.B]:C(Water)[T.Low]  and  C(Fertilizer):C(Water)
we come to know,
                 the interaction term is not significant. This indicates 
there is no interaction between the type of fertilizer and the amount of water 
on the mean crop yield. 
Since interaction is not significant, let's remove interaction from the model.

'''

# to get result without interaction of the factors, C(Fertilizer)*C(Water) 
# is not include , means we are not going to use below line

# model2 =  smf.ols(formula='Yield ~ C(Fertilizer)+C(Water)+C(Fertilizer)*C(Water)', data=aDf )

# “+” adds new columns to the design matrix 

model2 =  smf.ols(formula='Yield ~ C(Fertilizer)+C(Water)', data=aDf )
model_2_result =  model2.fit()
withoutInteraction =  model_2_result.summary()

# The R-Square will give use the variance, R-square will affect 
# by number of independent variable and number of sample/observation in the 
# independent variable. 

print(withoutInteraction)

# The ouput 

'''                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Yield   R-squared:                    0.390
Model:                            OLS   Adj. R-squared:               0.318
Method:                 Least Squares   F-statistic:                  5.430
Date:                Sat, 06 Apr 2019   Prob (F-statistic):           0.0150
Time:                        13:03:36   Log-Likelihood:             -51.772
No. Observations:                  20   AIC:                        109.5
Df Residuals:                      17   BIC:                        112.5
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===================================================================================
                 coef    std err          t      P>|t|      [0.025    0.975]
-----------------------------------------------------------------------------------
Intercept           32.6800      1.353     24.153      0.000    29.825    35.535
C(Fertilizer)[T.B]  -3.7200      1.562     -2.381      0.029    -7.016    -0.424
C(Water)[T.Low]     -3.5600      1.562     -2.279      0.036    -6.856    -0.264
==============================================================================
Omnibus:                     1.169     Durbin-Watson:                 2.736
Prob(Omnibus):               0.557     Jarque-Bera (JB):              0.820
Skew:                       -0.081     Prob(JB):                      0.664
Kurtosis:                    2.022     Cond. No.                      3.19
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is 
correctly specified.

''' 

# calculat the f-value and p-value of the model

identityMatrix = np.identity(len(model_2_result.params))
identityMatrix = identityMatrix[1:,:] 
print(model_2_result.f_test(identityMatrix))

# check the interaction is significant or not

res2 = sm.stats.anova_lm(model_2_result, typ= 2)
print(res2)

'''
(res2)indicate--->Each factor has an independent/different significant effect 
on the mean of 'yield'.

So to find out effect of each factor, fertilizer and water on the mean of crop yield.
we need to calculate "eta_sq" and "omega_sq".

"eta_sq" is a better measure than "omega_sq" as it considers degrees of freedom 
and produce unbiased  result.
R2 and "eta_sq" are the same thing in the ANOVA framework.
'''

# Anova eta-squared and omega-squared 

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    
    cols = ['sum_sq', 'mean_sq', 'df', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

etaNomega =  anova_table(res2) 
print(etaNomega)  # if we sum the eta_sq value we will get same R-square value

'''
Output is
=============
    			sum_sq	mean_sq		   df		  F	 	    PR(>F)		eta_sq	omega_sq
C(Fertilizer)69.192	69.192000	1.0	 5.669070	0.029228	   0.203477	   0.161778
C(Water)	   63.368	63.368000	1.0	 5.191895	0.035887	   0.186350   0.145244
Residual	   207.488	12.205176	17.0	     NaN		NaN			NaN				NaN

From the eta-sq we come to know each factor, fertilizer and water, has a 
small effect on the mean crop yield. 

So to know which factor is more important we need to do the post-hoc test.

'''

# Tukey’s Post-hoc Testing using tukeyhsd()

multicompare = mc(aDf['Yield'], aDf['Fertilizer'])
mc_results = multicompare.tukeyhsd()
print(mc_results)

'''
Multiple Comparison of Means - Tukey HSD,FWER=0.05
=============================================
group1 group2 meandiff  lower   upper  reject
---------------------------------------------
  A      B     -3.72   -7.3647 -0.0753  True 
---------------------------------------------

From the result we come to know " considering Type1-error 0.05 we are 
rejecting null hypothessis as 'Fertilizer A'  and 'Fertilizer B' both 
have different mean."
And "Fertilizer A" signifiantly higher than the "Fertilizer B". 
'''
multicompare = mc(aDf['Yield'], aDf['Water'])
mc_results = multicompare.tukeyhsd()
print(mc_results)

'''
Multiple Comparison of Means - Tukey HSD,FWER=0.05
============================================
group1 group2 meandiff  lower  upper  reject
--------------------------------------------
 High   Low    -3.56   -7.2436 0.1236 False 
--------------------------------------------

From the result we come to know " considering Type1-error 0.05 we fail 
to reject null hypothessis"

There is not a statistically significant difference in the mean crop 
yield between the amount of water used.

'''

'''
# we can apply vectorized functions "like  log , log_plus_1 " to the variables
# in a model
# res = smf.ols(formula='Lottery ~ np.log(Literacy)', data=df).fit()
'''



