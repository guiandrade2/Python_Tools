
import numpy as np 
import pandas as pd 
import seaborn as sns #graphs
import matplotlib.pyplot as plt
plt.rc("font", size=14)



#---------------------------------------------------------------------------------------------------------------------------

def count_categorical(data,output=True):
    #cicle through data.columns checking for their types and saving those that have 'object' type in 'categorical'
    categorical = [var for var in data.columns if data[var].dtype=='object']
    if(output == True):
        #the length of categorical is the number of categorical variables
        print('There are {} categorical variables\n'.format(len(categorical)))
    
        #check No labels per feature
        for var in categorical:
            no_unique_values = len(data[var].unique())
            n_missing_values = data[var].isnull().sum()
        
            print(var + " has " + str(no_unique_values) +
                  " distinct labels, with " + 
                  str(n_missing_values) + " missing values representing " + str( round(data[var].isnull().mean() * 100,2) ) + "%" ) 
        
    return categorical
    
#---------------------------------------------------------------------------------------------------------------------------
    
def count_numerical(data,output=True):
    #cicle through data.columns checking for their types and saving those that have 'object' type in 'categorical'
    numerical = [var for var in data.columns if data[var].dtype!='object']
    if (output == True):
           #the length of categorical is the number of categorical variables
        print('There are {} numerical variables\n'.format(len(numerical)))

          #check No labels per feature
        for var in numerical:
            print(var + " has " +  str(data[var].isnull().sum()) + " missing values representing " + str( round(data[var].isnull().mean() * 100,2) ) + "%"  ) 
    return numerical
    
    
#---------------------------------------------------------------------------------------------------------------------------
    
    
def boxplot_hist_numerical_data(data,numerical):
    for j in range(0,len(numerical)):
        plt.subplot(1,2,1)
        data.boxplot(column = numerical[j])
        
        plt.subplot(1,2,2)
        data[numerical[j]].hist().set_xlabel(numerical[j])
        
        plt.tight_layout()
        plt.show()
        
#---------------------------------------------------------------------------------------------------------------------------


def heatmap(data,title):
    correlation = data.corr()
    plt.figure(figsize=(16,12))
    plt.title(title)
    ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------

def correlation_graphs(data,columns): #takes a bit
    sns.pairplot(data[columns], kind='scatter', diag_kind='hist', palette='Rainbow')
    plt.show()
    
    
#---------------------------------------------------------------------------------------------------------------------------

def absolute_frequency(data,categorical):
    for var in categorical:
        print("-------------------------------" + var + "-------------------------------")
        print(data[var].value_counts())
    
#---------------------------------------------------------------------------------------------------------------------------

def relative_frequency(data,categorical):
    for var in categorical:
        print("-------------------------------" + var + "-------------------------------")
        print(data[var].value_counts()/np.float(data.shape[0]))
