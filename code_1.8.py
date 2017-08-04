# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:44:31 2017

@author: charles satterlee

"""

import numpy as np 
import matplotlib.pyplot as plt  
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress 
import os
 
#input array 
x = np.genfromtxt('time_data_71417.py',delimiter=',')
y = np.genfromtxt('novo_c4.py',delimiter=',')

#Smooths line using loess 

def smoother(x,y):
    lowess = sm.nonparametric.lowess(y,x, frac=.5)
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    plt.plot(lowess_x,lowess_y)
    return lowess_x,lowess_y

#takes the diff within the array
 
def differentiate(y1):
    differ = np.diff(y1,n=1,axis=1)
    return differ

#groups the array by some (* group size) amount. Should be changed to a percent of line in future. 

def group(x1,y1):
    #xgroup = list(zip(*[iter(x1)] * 5)) #[len(x1) * .10]))
    ygroup = list(zip(*[iter(y1)] * 5)) #[len(y1) * .10]))
    diff_group = differentiate(ygroup)
    avg_group = np.average(diff_group,axis=1)
    df = pd.DataFrame({'avg_g' : avg_group})
    return df

#Creates a column with incremental values to weight values based on x location in the array 

def increment(df):
    incr = np.arange(0.5,100,.5) #creates an index of incremental values from .5 to 100 with .5 increment 
    g = incr[0:len(df.index)] #selects elements(inc) from 0-length of dataframe index
    df = df.assign(inc=g)
    df['inc_value'] = df['avg_g'] * df['inc']
    df0 = df[['growth','inc_value']]
    df0.columns = range(df0.shape[1]) #removes the avg_g header so it doesn't pivot into the row of the next matrix
    return df0

#bins values based on amount of growth. Sums the total weighted values within the bins 

def binner(df):
    bins = [0.0001,0.005,0.01,0.05,0.1] 
    group_names = ['low','medium','high','very-high']
    df['growth'] = pd.cut(df['avg_g'], bins, labels=group_names)#cuts, bins, and labels avg_g and places labels in a 'growth' column next to avg_g
    df0 = increment(df)
    table = pd.pivot_table(df0, columns=[0],aggfunc=np.sum)
    df0 = pd.DataFrame.fillna(table,value=0, method=None, axis=None, inplace=False, limit=None, downcast=None) #places 0.0 in place of NaN  
    return df0

# writes values to file 

def writer(df): 
    
    if not os.path.exists('/write.py'): #looks for output file, if not, creates file  
        os.mkdir('/write.py')
        
    mind = open('write.py','a')
    writer = df.to_csv('write.py', mode='a',index=False, header=False)
    mind.close()
    return writer 

# finds slope of line 

def findslope(x,y): # find the slope of the curve
    slope = linregress(x,y)   
    return slope[0]

#finds final optical density of array 

def finalod(lowess_y): #final optical density 
    final_od = lowess_y[-1]
    return final_od

# raw input for important data 

def compoundinfo():
    name = input("What is the full name of the compound used for this curve?")
    mic = input("What is the MIC of the experiment?")
    return mic,name

#runs all functions and formats data into desired dataframe

def framer(a,b,c,d):
    #a = pd.DataFrame(a, index=['very-high', 'high', 'medium', 'low'], columns="avg_g")
    df = pd.DataFrame({'very-high': a['very-high'],
                       'high' : a['high'],
                       'medium' : a['medium'],
                       'low' : a['low'],
                       'name': [b[1]],
                       'mic' : [b[0]],
                       'final_od' : [c],
                       'slope': [d]})
       
    df = df[['name','final_od','mic','slope','very-high','high','medium','low']] #reorder the headers     
    return df

#Runs required functions and outputs a pandas dataframe for LDA attributes 

def main(x,y):
    raw_info = compoundinfo()
    lines = smoother(x,y)
    slope = findslope(lines[0],lines[1])
    final_od = finalod(lines[1])
    avg_g = group(x,y)
    growth_bins = binner(avg_g)
    df = framer(growth_bins,raw_info,final_od,slope)
    df = df.round(4)
    writer(df)
    return df
#index slope 
print(main(x,y))