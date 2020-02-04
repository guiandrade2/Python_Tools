from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class DistributionScaler(BaseEstimator, TransformerMixin):   #transformer into diferent scales or normalization/standartization
    def __init__(self, strategy = None): # no *args or **kargs      
        self.strategy = strategy
        self.estimator = None
    def fit(self, X):
        if(self.strategy=="StandardScaler"):
            self.estimator=StandardScaler().fit(X)
        elif(self.strategy=="MinMaxScaler"):
            self.estimator=MinMaxScaler().fit(X)
        elif(self.strategy=="MaxAbsScaler"):
            self.estimator=MaxAbsScaler().fit(X)
        elif(self.strategy=="RobustScaler"):
            self.estimator=RobustScaler(quantile_range=(25, 75)).fit(X)
        elif(self.strategy=="PowerTransformer_Yeo_Johnson"):
            self.estimator=PowerTransformer(method='yeo-johnson').fit(X)
        elif(self.strategy=="PowerTransformer_Box_Cox"):
            self.estimator=PowerTransformer(method='box-cox').fit(X)
        elif(self.strategy=="QuantileTransformer_Normal"):
            self.estimator=QuantileTransformer(output_distribution='normal').fit(X)
        elif(self.strategy=="QuantileTransformer_Uniform"):
            self.estimator=QuantileTransformer(output_distribution='uniform').fit(X)
        elif(self.strategy=="Normalizer"):
            self.estimator=Normalizer().fit(X)
        return self  # nothing else to do  
    def transform(self, X):      
        if (self.estimator != None):
            return self.estimator.transform(X)
        else:
            if (isinstance(X,(pd.core.frame.DataFrame))):
                return X.to_numpy()
            else:
                return X
#-----------------------------------------------------------------------------------------------------------   


class OutlierApproacher(BaseEstimator, TransformerMixin): 
    def __init__(self, strategy = None):#,threshold=3): # add treatment later for removal or imputation, add default/custom threshold = 1.5 for IQR and threshold = 3 for ZScore
        self.strategy = strategy
        #self.threshold = threshold
        self.iqr = None
        self.q3 = None
        self.q1 = None
        self.zscore = None
    def fit(self, X):
        if (self.strategy=="ZScore"):
            self.zscore = np.abs(stats.zscore(X))
        if (self.strategy=="IQR"):
            Q1 = wines.quantile(0.25)
            Q3 = wines.quantile(0.75)
            self.iqr = Q3 - Q1
        return self
        
    def transform(self, X):
        if (self.strategy=="ZScore"):
            return X[(self.zscore < 3).all(axis=1)].to_numpy()
        if (self.strategy=="IQR"):
            return X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)].to_numpy()
        else:
            return X.to_numpy()
