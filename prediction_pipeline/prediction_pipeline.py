import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as time
from typing import List


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.ensemble import GradientBoostingRegressor

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy.stats import ks_2samp, kendalltau
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import TimeSeriesSplit

from autofeat import FeatureSelector

import random

#-----feature engineering
#target lagged values
def addStatisticalAggregations(df: pd.DataFrame, columnsToAggregate: List[str], suffix: str):
    cols = df[columnsToAggregate]
    df[f'{suffix}mean'] = cols.mean(axis=1)
    df[f'{suffix}std'] = cols.std(axis=1)
    df[f'{suffix}min'] = cols.min(axis=1)
    df[f'{suffix}max'] = cols.max(axis=1)
    return df

def lagTarget(data: pd.DataFrame, targetColumnName: str,  shift=24): 
    colName = targetColumnName
    d = data[[colName]].rename(columns={colName:'y'})
    dd = d.copy() 
    for i in range(1,25): dd[f'y-{i}'] = dd['y'].shift(periods=i,freq='H') #1-23hour shifts
    for i in range(1,8): dd[f'y-{i}d'] = dd['y'].shift(periods=i,freq='D') #1-7 calendar day shift
    for i in range(1,5): dd[f'y-{i}w'] = dd['y'].shift(periods=7*i,freq='D') #1-4week shifts
    for i in range(1,25): dd[f'y-{shift}-{i}'] = dd['y'].shift(periods=i+shift,freq='H') #1-23hour shifts after main shift
    
    dd = addStatisticalAggregations(dd,columnsToAggregate=[f'y-{i}' for i in range(1,25)], suffix='y_24hP')
    dd = addStatisticalAggregations(dd,columnsToAggregate=[f'y-{shift}-{i}' for i in range(1,25)], suffix=f'y-{shift}_24hP')
    dd = addStatisticalAggregations(dd,columnsToAggregate=[f'y-{i}w' for i in range(1,5)], suffix='y_4wP')
    dd = addStatisticalAggregations(dd,columnsToAggregate=[f'y-{i}d' for i in range(1,8)], suffix='y_7dP')

    dd['dayOfWeek'] = dd.reset_index()['index'].dt.weekday.values #0,1..5,6
    #lets calculate average in previous 5 working days or average in previous 2 weekends depending on current day (working or weekend)
    #this implementation works 15x faster compared to apply(func) solution used previously
    workingDays = dd[dd['dayOfWeek']<5][['y']] #dataset with only working days
    dd_wd = workingDays.copy()
    #1-5 business days (bD) shifts
    dd_wd_shifted = [workingDays.shift(freq=pd.tseries.offsets.BusinessDay(i)).rename(columns={'y': f'y-{i}bD'}) for i in range(1,6)] 
    dd_wd = pd.concat([dd_wd, *dd_wd_shifted], axis=1)
    dd_wd = addStatisticalAggregations(dd_wd,columnsToAggregate=[f'y-{i}bD' for i in range(1,6)], suffix='y_5wd/2weP')
    dd_wd = dd_wd[['y_5wd/2wePmean','y_5wd/2wePstd','y_5wd/2wePmin','y_5wd/2wePmax']]

    weekendsDays = dd[dd['dayOfWeek']>4][['y']] #dataset with only Sat and Sun
    dd_we = weekendsDays.copy()
    #1-2 weekend days (weD) shifts
    dd_we_shifted = [weekendsDays.shift(freq=pd.tseries.offsets.CustomBusinessDay(i,weekmask='Sat Sun')).rename(columns={'y': f'y-{i}weD'}) for i in range(1,3)]
    dd_we = pd.concat([dd_we, *dd_we_shifted], axis=1)
    dd_we = addStatisticalAggregations(dd_we,columnsToAggregate=[f'y-{i}weD' for i in range(1,3)], suffix='y_5wd/2weP')
    dd_we = dd_we[['y_5wd/2wePmean','y_5wd/2wePstd','y_5wd/2wePmin','y_5wd/2wePmax']]
    dd_both = pd.concat([dd_wd,dd_we],axis=0)
    dd = pd.concat([dd,dd_both],axis=1)

    dd['hour'] = dd.reset_index()['index'].dt.hour.values
    #lets calculate average in previous 12 dayhours or average in previous 12 nighthours depending on current hour
    dayHours = dd[dd['hour'].isin(np.arange(8,20))][['y']] #dataset with only day hours
    dd_dh = dayHours.copy()
    #1-12 day hours (dH) shifts
    dd_dh_shifted = [dayHours.shift(i).rename(columns={'y': f'y-{i}dH'}) for i in range(1,13)]
    dd_dh = pd.concat([dd_dh, *dd_dh_shifted], axis=1)
    dd_dh = addStatisticalAggregations(dd_dh,columnsToAggregate=[f'y-{i}dH' for i in range(1,13)], suffix='y_12nh/12dhP')
    dd_dh = dd_dh[['y_12nh/12dhPmean','y_12nh/12dhPstd','y_12nh/12dhPmin','y_12nh/12dhPmax']]

    nightHours = dd[~dd['hour'].isin(np.arange(8,20))][['y']] #dataset with only night hours
    dd_nh = nightHours.copy()
    #1-12 hight hours (nH) shifts
    dd_nh_shifted = [nightHours.shift(i).rename(columns={'y': f'y-{i}nH'}) for i in range(1,13)]
    dd_nh = pd.concat([dd_nh, *dd_nh_shifted], axis=1)
    dd_nh = addStatisticalAggregations(dd_nh,columnsToAggregate=[f'y-{i}nH' for i in range(1,13)], suffix='y_12nh/12dhP')
    dd_nh = dd_nh[['y_12nh/12dhPmean','y_12nh/12dhPstd','y_12nh/12dhPmin','y_12nh/12dhPmax']]
    dd_both = pd.concat([dd_dh,dd_nh],axis=0)
    dd = pd.concat([dd,dd_both],axis=1)

    dd = dd.dropna(axis=0)
    return dd.drop(columns=['dayOfWeek','hour']+
    ['y-'+str(i) for i in range(1,25)]+ #we need to drop the following features since they are inside shift interval so that not accessible for test data
    ['y_24hPmean','y_24hPstd','y_24hPmin','y_24hPmax']+
    ['y_12nh/12dhPmean','y_12nh/12dhPstd','y_12nh/12dhPmin','y_12nh/12dhPmax']
    )

#DateTime features
def dtFeatures(target: pd.DataFrame):
    Xy = target.copy()
    Xy['time'] = Xy.index.values
    Xy['month'] = Xy['time'].dt.month.values  #1..12 categorical
    Xy['month_sin'] = np.sin((Xy['month'] - 1)*2*np.pi/11) # this sin/cos transformation puts month on a circle, so that position on a circle is given by x=sin, y=cos
    Xy['month_cos'] = np.cos((Xy['month'] - 1)*2*np.pi/11) 
    Xy['day'] = Xy['time'].dt.day.values   #1..30(1)
    Xy['day_sin'] = np.sin((Xy['day'] - 1)*2*np.pi/30)     
    Xy['day_cos'] = np.cos((Xy['day'] - 1)*2*np.pi/30)     
    Xy['hour'] = Xy['time'].dt.hour.values #1..24
    Xy['hour_sin'] = np.sin((Xy['hour'] - 0)*2*np.pi/23)
    Xy['hour_cos'] = np.cos((Xy['hour'] - 0)*2*np.pi/23)
    Xy['dayofyear'] = Xy['time'].dt.dayofyear.values #1..365
    Xy['dayofyear_sin'] = np.sin((Xy['dayofyear'] - 1)*2*np.pi/364)     
    Xy['dayofyear_cos'] = np.cos((Xy['dayofyear'] - 1)*2*np.pi/364)     
    Xy['weekofyear'] = Xy['time'].dt.weekofyear.values #1..52
    Xy['weekofyear_sin'] = np.sin((Xy['weekofyear'] - 1)*2*np.pi/51)
    Xy['weekofyear_cos'] = np.cos((Xy['weekofyear'] - 1)*2*np.pi/51)
    Xy['weekday'] = Xy['time'].dt.weekday.values #0..6
    Xy['weekday_sin'] = np.sin((Xy['weekday'] - 0)*2*np.pi/6)
    Xy['weekday_cos'] = np.cos((Xy['weekday'] - 0)*2*np.pi/6)
    Xy['quarter'] = Xy['time'].dt.quarter.values  #1..4
    Xy['quarter_sin'] = np.sin((Xy['quarter'] - 1)*2*np.pi/3) 
    Xy['quarter_cos'] = np.cos((Xy['quarter'] - 1)*2*np.pi/3) 
    #one-hot encoding of categorical features
    Xy['month1'] = Xy['month']
    Xy = pd.get_dummies(Xy, columns=['month1'], drop_first=False, prefix='month')
    Xy['weekday1'] = Xy['weekday']
    Xy = pd.get_dummies(Xy, columns=['weekday1'], drop_first=False, prefix='wday')
    Xy['quarter1'] = Xy['quarter']
    Xy = pd.get_dummies(Xy, columns=['quarter1'], drop_first=False, prefix='qrtr')
    #single valued coulumns will be removed when we do feature selection
    Xy = Xy.drop(columns=['time'])

    #add missing one-hot encoded features
    cols=['month_'+str(i) for i in range(1,13)]+['wday_'+str(i) for i in range(7)]+['qrtr_'+str(i) for i in range(1,5)]
    newCols = Xy.columns.union(cols,sort=False) #missing cols
    Xy = Xy.reindex(newCols, axis=1, fill_value=0)
    return Xy

#covariate based features
def lagCovariate(data: pd.DataFrame, covariateColumnName: str):
    c = covariateColumnName[0]
    d = data.rename(columns={covariateColumnName:c})  #temp to t; price to p
    dd = d.copy() 

    #1-24hour shifts into past
    dd_shifted = [d.shift(periods=i,freq='H').rename(columns={c:f'{c}-{i}'}) for i in range(1,25)]
    dd = pd.concat([dd, *dd_shifted], axis=1)

    #1-12hour shifts into future
    dd_shifted = [d.shift(periods=-i,freq='H').rename(columns={c:f'{c}+{i}'}) for i in range(1,13)]
    dd = pd.concat([dd, *dd_shifted], axis=1)

    dd = addStatisticalAggregations(dd,columnsToAggregate=[f'{c}-{i}' for i in range(1,25)], suffix=f'{c}_24hP')    
    dd[c+'_24hPstandard'] = (dd[c] - dd[c+'_24hPmean'])/dd[c+'_24hPstd']
    dd.loc[dd[c+'_24hPstd']<0.0001,[c+'_24hPstandard']] = 0.0  #if std = 0 then all values are the same; giving 0 means that dd[c]=dd[c_24hPmean]
    dd[c+'_24hPminmax'] = (dd[c] - dd[c+'_24hPmin'])/(dd[c+'_24hPmax']-dd[c+'_24hPmin'])
    dd.loc[np.abs(dd[c+'_24hPmax']-dd[c+'_24hPmin'])<0.0001,[c+'_24hPminmax']] = 0.5 
    
    dd = addStatisticalAggregations(dd,columnsToAggregate=[f'{c}-{i}' for i in range(1,13)], suffix=f'{c}_12hF')
    dd[c+'_12hFstandard'] = (dd[c] - dd[c+'_12hFmean'])/dd[c+'_12hFstd']
    dd.loc[dd[c+'_12hFstd']<0.0001,[c+'_12hFstandard']] = 0.0 
    dd[c+'_12hFminmax'] = (dd[c] - dd[c+'_12hFmin'])/(dd[c+'_12hFmax']-dd[c+'_12hFmin'])
    dd.loc[np.abs(dd[c+'_12hFmax']-dd[c+'_12hFmin'])<0.0001,[c+'_12hFminmax']] = 0.5
    
    dd['hour'] = dd.reset_index()['index'].dt.hour.values

    #lets calculate average in previous 12 dayhours or average in previous 12 nighthours depending on current hour
    dayHours = dd[dd['hour'].isin(np.arange(8,20))][[c]] #dataset with only day hours 8,9...18,19
    dd_dh = dayHours.copy()
    #1-12 day hours (dH) shifts
    dd_dh_shifted = [dayHours.shift(i).rename(columns={c:f'{c}-{i}PdH'}) for i in range(1,13)]
    dd_dh = pd.concat([dd_dh, *dd_dh_shifted], axis=1)
    dd_dh = addStatisticalAggregations(dd_dh,columnsToAggregate=[f'{c}-{i}PdH' for i in range(1,13)], suffix=f'{c}_12nh/12dhP')
    dd_dh[c+'_12nh/12dhPstandard'] = (dd_dh[c] - dd_dh[c+'_12nh/12dhPmean'])/dd_dh[c+'_12nh/12dhPstd']
    dd_dh.loc[dd_dh[c+'_12nh/12dhPstd']<0.0001,[c+'_12nh/12dhPstandard']] = 0.0 
    dd_dh[c+'_12nh/12dhPminmax'] = (dd_dh[c] - dd_dh[c+'_12nh/12dhPmin'])/(dd_dh[c+'_12nh/12dhPmax']-dd_dh[c+'_12nh/12dhPmin'])
    dd_dh.loc[np.abs(dd_dh[c+'_12nh/12dhPmax']-dd_dh[c+'_12nh/12dhPmin'])<0.0001,[c+'_12nh/12dhPminmax']] = 0.5

    #lets calculate average in next 12 dayhours or average in next 12 nighthours depending on current hour
    #1-12 day hours (dH) shifts
    dd_dh_shifted = [dayHours.shift(-i).rename(columns={c:f'{c}+{i}FdH'}) for i in range(1,13)]
    dd_dh = pd.concat([dd_dh, *dd_dh_shifted], axis=1)
    dd_dh = addStatisticalAggregations(dd_dh,columnsToAggregate=[f'{c}+{i}FdH' for i in range(1,13)], suffix=f'{c}_12nh/12dhF')
    dd_dh[c+'_12nh/12dhFstandard'] = (dd_dh[c] - dd_dh[c+'_12nh/12dhFmean'])/dd_dh[c+'_12nh/12dhFstd']
    dd_dh.loc[dd_dh[c+'_12nh/12dhFstd']<0.0001,[c+'_12nh/12dhFstandard']] = 0.0 
    dd_dh[c+'_12nh/12dhFminmax'] = (dd_dh[c] - dd_dh[c+'_12nh/12dhFmin'])/(dd_dh[c+'_12nh/12dhFmax']-dd_dh[c+'_12nh/12dhFmin'])
    dd_dh.loc[np.abs(dd_dh[c+'_12nh/12dhFmax']-dd_dh[c+'_12nh/12dhFmin'])<0.0001,[c+'_12nh/12dhFminmax']] = 0.5
    dd_dh = dd_dh[[c+'_12nh/12dhPmean',c+'_12nh/12dhPstd',c+'_12nh/12dhPstandard',
                   c+'_12nh/12dhPmin',c+'_12nh/12dhPmax',c+'_12nh/12dhPminmax',
                   c+'_12nh/12dhFmean',c+'_12nh/12dhFstd',c+'_12nh/12dhFstandard',
                   c+'_12nh/12dhFmin',c+'_12nh/12dhFmax',c+'_12nh/12dhFminmax']]

    nightHours = dd[~dd['hour'].isin(np.arange(8,20))][[c]] #dataset with only night hours
    dd_nh = nightHours.copy()
    #1-12 hight hours (nH) shifts
    dd_nh_shifted = [nightHours.shift(i).rename(columns={c:f'{c}-{i}PnH'}) for i in range(1,13)]
    dd_nh = pd.concat([dd_nh, *dd_nh_shifted], axis=1)
    dd_nh = addStatisticalAggregations(dd_nh,columnsToAggregate=[f'{c}-{i}PnH' for i in range(1,13)], suffix=f'{c}_12nh/12dhP')
    dd_nh[c+'_12nh/12dhPstandard'] = (dd_nh[c] - dd_nh[c+'_12nh/12dhPmean'])/dd_nh[c+'_12nh/12dhPstd']
    dd_nh.loc[dd_nh[c+'_12nh/12dhPstd']<0.0001,[c+'_12nh/12dhPstandard']] = 0.0 
    dd_nh[c+'_12nh/12dhPminmax'] = (dd_nh[c] - dd_nh[c+'_12nh/12dhPmin'])/(dd_nh[c+'_12nh/12dhPmax']-dd_nh[c+'_12nh/12dhPmin'])
    dd_nh.loc[np.abs(dd_nh[c+'_12nh/12dhPmax']-dd_nh[c+'_12nh/12dhPmin'])<0.0001,[c+'_12nh/12dhPminmax']] = 0.5
    #lets calculate average in next 12 dayhours or average in next 12 nighthours depending on current hour
    #1-12 nighthours (nH) shifts
    dd_nh_shifted = [nightHours.shift(-i).rename(columns={c:f'{c}+{i}FnH'}) for i in range(1,13)]
    dd_nh = pd.concat([dd_nh, *dd_nh_shifted], axis=1)
    dd_nh = addStatisticalAggregations(dd_nh,columnsToAggregate=[f'{c}+{i}FnH' for i in range(1,13)], suffix=f'{c}_12nh/12dhF')
    dd_nh[c+'_12nh/12dhFstandard'] = (dd_nh[c] - dd_nh[c+'_12nh/12dhFmean'])/dd_nh[c+'_12nh/12dhFstd']
    dd_nh.loc[dd_nh[c+'_12nh/12dhFstd']<0.0001,[c+'_12nh/12dhFstandard']] = 0.0 
    dd_nh[c+'_12nh/12dhFminmax'] = (dd_nh[c] - dd_nh[c+'_12nh/12dhFmin'])/(dd_nh[c+'_12nh/12dhFmax']-dd_nh[c+'_12nh/12dhFmin'])
    dd_nh.loc[np.abs(dd_nh[c+'_12nh/12dhFmax']-dd_nh[c+'_12nh/12dhFmin'])<0.0001,[c+'_12nh/12dhFminmax']] = 0.5
    dd_nh = dd_nh[[c+'_12nh/12dhPmean',c+'_12nh/12dhPstd',c+'_12nh/12dhPstandard',
                   c+'_12nh/12dhPmin',c+'_12nh/12dhPmax',c+'_12nh/12dhPminmax',
                   c+'_12nh/12dhFmean',c+'_12nh/12dhFstd',c+'_12nh/12dhFstandard',
                   c+'_12nh/12dhFmin',c+'_12nh/12dhFmax',c+'_12nh/12dhFminmax']]
    dd_both = pd.concat([dd_dh,dd_nh],axis=0)
    dd = pd.concat([dd,dd_both],axis=1)

    #lets rescale based on the same day min/max values
    dd['dayOfYear'] = dd.reset_index()['index'].dt.dayofyear.values
    dd = dd[-(24*364):] #only last year considered
    grouped = dd[[c,'dayOfYear']].groupby(by='dayOfYear').max().rename(columns={c:c+'_dayMax'})
    grouped[c+'_dayMin'] = dd[[c,'dayOfYear']].groupby(by='dayOfYear').min()[c].values
    grouped[c+'_dayStd'] = dd[[c,'dayOfYear']].groupby(by='dayOfYear').std()[c].values
    grouped[c+'_dayMean'] = dd[[c,'dayOfYear']].groupby(by='dayOfYear').mean()[c].values
    grouped = grouped.rename_axis(None)
    dd = pd.concat([dd.reset_index().set_index('dayOfYear').rename_axis(None),grouped],axis=1).set_index('index').rename_axis(None)
    dd[c+'_dayMinmax'] = (dd[c]-dd[c+'_dayMin'])/(dd[c+'_dayMax']-dd[c+'_dayMin'])
    dd.loc[np.abs(dd[c+'_dayMax']-dd[c+'_dayMin'])<0.0001,[c+'_dayMinmax']] = 0.5
    dd[c+'_dayStandard'] = (dd[c] - dd[c+'_dayMean'])/dd[c+'_dayStd']
    dd.loc[dd[c+'_dayStd']<0.0001,[c+'_dayStandard']] = 0.0 

#    dd = dd.dropna(axis=0)
    return dd.drop(columns=['hour'])

def makeDatasets(load: pd.DataFrame, temp: pd.DataFrame, price: pd.DataFrame, horizon, setList = {'TSFresh':0,'Lagged':1,'DateTime':1,'Temp':1,'Price':1}, verbose=False): 
    #load/temp/price are 1 column dataframes with datetime index
    # setList says what features to engineer
    keys = np.array([i for i in setList.keys()])
    vals = np.array([i for i in setList.values()])==1
    datasets = {'Lagged':{'trainVal':None, 'test':None},
            'DateTime':{'trainVal':None, 'test':None},
            'Temp':{'trainVal':None, 'test':None},
            'Price':{'trainVal':None, 'test':None},
           }
    
    dd = load.rename(columns={load.columns[0]:'y'})
    testSet = pd.DataFrame({'y':-1},index = pd.date_range(start=load.index[-1],periods=(horizon+1),freq='H')[1:])
    eicdataTest = pd.concat([dd,testSet],axis=0).sort_index() #trainValTest

    for featureSet in keys[vals]:
        if featureSet == 'Lagged':
            start = time.time()
            targetColumnName = eicdataTest.columns[0]
            XY_L = lagTarget(eicdataTest, targetColumnName)
            datasets['Lagged']['trainVal'] = {
                'trainValFeatures':XY_L[:(-horizon)].drop(columns=['y']),
                'trainValTarget':XY_L[:(-horizon)][['y']]
                }
            datasets['Lagged']['test'] = {
                'testFeatures':XY_L[(-horizon):].drop(columns=['y']),
                'testTarget':None
                }
            if verbose == True: print(f'{featureSet} features prepatation time {(time.time()-start):.3f}s')

        if featureSet == 'DateTime':
            start = time.time()
            XY_D = dtFeatures(eicdataTest)
            datasets['DateTime']['trainVal'] = {
                'trainValFeatures':XY_D[:(-horizon)].drop(columns=['y']),
                'trainValTarget':XY_D[:(-horizon)][['y']]
                }
            datasets['DateTime']['test'] = {
                'testFeatures':XY_D[(-horizon):].drop(columns=['y']),
                'testTarget':None
                }
            if verbose == True: print(f'{featureSet} features prepatation time {(time.time()-start):.3f}s')

        if featureSet == 'Temp':
            start = time.time()
            covariateColumnName = temp.columns[0]
            X_Temp = lagCovariate(temp,covariateColumnName)
            datasets['Temp']['trainVal'] = {
                'trainValFeatures':X_Temp[X_Temp.index<=dd.index[-1]],
                'trainValTarget':None
                }
            datasets['Temp']['test'] = {
                'testFeatures':X_Temp[X_Temp.index>dd.index[-1]],
                'testTarget':None
                }
            if verbose == True: print(f'{featureSet} features prepatation time {(time.time()-start):.3f}s')
        if featureSet == 'Price':
            start = time.time()
            covariateColumnName = price.columns[0]
            X_Price = lagCovariate(price,covariateColumnName)
            datasets['Price']['trainVal'] = {
                'trainValFeatures':X_Price[X_Price.index<=dd.index[-1]],
                'trainValTarget':None
                }
            datasets['Price']['test'] = {
                'testFeatures':X_Price[X_Price.index>dd.index[-1]],
                'testTarget':None
                }
            if verbose == True: print(f'{featureSet} features prepatation time {(time.time()-start):.3f}s')
    return datasets

def joinFeatures(datasets,pastLags,horizon):
    trainValFeatures = pd.DataFrame()
    trainValTarget = pd.DataFrame()
    testFeatures = pd.DataFrame()
    for ds in [i for i in datasets.keys()]:
        trainValFeatures = pd.concat([trainValFeatures,datasets[ds]['trainVal']['trainValFeatures'][-pastLags:]],axis=1)
        testFeatures = pd.concat([testFeatures, datasets[ds]['test']['testFeatures'][:horizon]],axis=1)
    trainValTarget = datasets['DateTime']['trainVal']['trainValTarget'][-pastLags:]
    return {'trainValFeatures':trainValFeatures,'trainValTarget':trainValTarget,'testFeatures':testFeatures}
    
#-----feature selection
def tsfresh_select_features(dataX, dataY):
#function to replicate tsfresh select_feature functionality
#we assume that target is always real in our case
#we assume that features can be real or binary
#binary feature vs real target: Kolmogorov-Smirnov test - similarity of two distributions
#https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_selection.html?highlight=select_features#tsfresh.feature_selection.significance_tests.target_real_feature_real_test
#https://stats.stackexchange.com/questions/575740/usefulness-of-ks-tests-and-other-similar-distribution-comparing-tests
    binary_features = dataX.nunique()[dataX.nunique()==2].index.values
    binary_features_pvalues = []
    for c in binary_features:
        val1 = dataX[c].min()
        val2 = dataX[c].max()
        val1data = dataX[dataX[c]==val1][[c]]
        val2data = dataX[dataX[c]==val2][[c]]
        dist1 = dataY.loc[val1data.index]
        dist2 = dataY.loc[val2data.index]
        pvalue = ks_2samp(dist1.values.flatten(),dist2.values.flatten())[1] 
        #we create distributions for the one value of binary feature and for another, and conpare these distributions
        #H0: distributions are identical, so that value of the feature is not important
        binary_features_pvalues += [pvalue]
# Benjamini Hochberg procedure for multiple testing  
#see tsfresh select_feature code https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_selection/selection.py
    rejectH0KS = multipletests(binary_features_pvalues, alpha=0.05, method = 'fdr_by')[0]
    meaningful_binary_features = binary_features[rejectH0KS]

#real feature vs real target: Kendall-tau test - correspondence between two rankings. More robust than Spearman correlation
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
#https://www.kaggle.com/code/liamfitzpatrick/kendall-tau-for-feature-selection/notebook
#https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_selection/significance_tests.py
    real_features = dataX.nunique()[dataX.nunique()>2].index.values
    real_features_pvalues = []
    for c in real_features:
        #we find ranks for two arrays
        #H0: distributions are identical, so that value of the feature is not important
        d1 = dataX[[c]].rank()
        d2 = dataY.rank()
        pvalue = kendalltau(d1,d2,method="asymptotic")[1]
        real_features_pvalues += [pvalue]
# Benjamini Hochberg procedure for multiple testing  
#https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_selection/relevance.py
#https://stats.stackexchange.com/questions/63441/what-are-the-practical-differences-between-the-benjamini-hochberg-1995-and-t    
    rejectH0Kendall = multipletests(real_features_pvalues, alpha=0.05,method = 'fdr_by')[0]
    meaningful_real_features = real_features[rejectH0Kendall]   
    return dataX[list(meaningful_binary_features)+list(meaningful_real_features)]

def dropStatistically(trainX,trainY,verbose=True):
    #X_train = select_features(trainX,trainY) #tsfresh implementation
    X_train = tsfresh_select_features(trainX, trainY) #my implementation of tsfresh functionality
    if verbose==True: print(f'features left {X_train.shape[1]}')
    return X_train

def shortenFeatures(features,target,method='TSFresh'):  #according to staristical tests TSFresh is best choice for all households
    #drop uninformative features with single value
    uninformativeFeatures = features.columns[(features.nunique()==1).values] 
    features = features.drop(columns=uninformativeFeatures)

    if method == 'Sel':
        fsel = FeatureSelector(verbose=0,n_jobs=10)
        featuresOut = fsel.fit_transform(features, target.values)  #index is dropped here
        featuresOut.index = features.index
    if method == 'Cor':
        targetName = target.columns[0]
        aa = pd.concat([features, target],axis=1)
        cor = aa.corr()
        cor_target = abs(cor[targetName].drop(index=[targetName]))
        #corrFeatures = cor_target[cor_target > 0.15].index  #0.1 is fitting param
        corrFeatures = cor_target.sort_values()[-30:].index.values
        featuresOut = features[corrFeatures]
    if method == 'Freg':
        best = SelectKBest(score_func=f_regression, k=20)   #k=20 us fitting param
        best = best.fit(features, target)
        fregFeatures = features.columns[best.get_support(indices=True)].values
        featuresOut = features[fregFeatures]
    if method == 'TSFresh':
        featuresOut = dropStatistically(features,target,verbose=False)
    if method =='Uku':
        #manual selection: only two features - hour and temperature
        featuresOut = pd.DataFrame({'t':[1],'hour':[2]})
    if method =='Kristjan':
        # manual selection:
        # for consumption - lagged values; some aggregations (last weeks same hour average or working day same hour average)
        # for covariates - past and future lagged values; comparison with same day min/max value
        y = {i:[1] for i in features.columns if 'y-24-' in i}
        mean_y = {'y-24_24hPmean':[1],'y_4wPmean':[1],'y_7dPmean':[1],'y_5wd/2wePmean':[1]}
        y.update(mean_y)
        t = {f't-{i}':[1] for i in range(1,13)}
        t.update({f't+{i}':[1] for i in range(1,13)})
        t.update({'t':[1],'t_dayMinmax':[1]})
        p = {f'p-{i}':[1] for i in range(1,13)}
        p.update({f'p+{i}':[1] for i in range(1,13)})
        p.update({'p':[1],'p_dayMinmax':[1]})
        
        y.update(t)
        y.update(p)
        featuresOut = pd.DataFrame(y)

    return featuresOut.columns

#-----fit regressor
def trainScale(train):
    scaler = StandardScaler()
    trainData = scaler.fit_transform(train)
    trainData = pd.DataFrame(trainData, columns=train.columns, index=train.index)
    return trainData, scaler

def plotImportance(importance,features,n_features,verbose=False):
    indices = np.argsort(np.abs(np.array(importance)))[-n_features:] #for importance abs is not needed, but use this function for LR
    if verbose == True:
        fig = plt.figure()
        plt.barh(features[indices],np.abs(np.array(importance[indices])))
        plt.title('feature importance')
        #plt.xlim(0,1)
        plt.show()
        #plt.savefig('featureImporance_'+eic+'.png',bbox_inches = "tight")
        plt.close(fig)
    return list(features[indices])

def wmape(actual,pred):
    if np.sum(actual.values) <0.0001: out = 1 #to avoid devision by 0
    else: out = np.sum(np.abs(actual.values-pred.values))/np.sum(actual.values)
    return out

def fixRegressor(): #regressor and its hyperparameters
    regressor = GradientBoostingRegressor
    SEARCH_PARAMS = {
        'n_estimators': np.arange(50,250,50),#default 100
        'max_depth': np.arange(4,9), #default 3
        'learning_rate': np.array([0.1,0.2]), #default 0.1
        'subsample': np.arange(7,11)*0.1, #default 1
        'random_state': [2022],
        'loss':['absolute_error']
    }
    return {'regressor':regressor, 'hyperparams': SEARCH_PARAMS}

def fitRegressor(regressor,trainValFeatures,trainValTarget,verbose=False,validate='expanding',bestVal='top3'):
    start = time.time()

    #regressor
    model = regressor['regressor']
    SEARCH_PARAMS = regressor['hyperparams']

    #data
    joinedX = trainValFeatures.copy(deep=True)
    joinedX, scaler = trainScale(joinedX) #we scale train/val together, since scaling separately will be problematic in case of 5fold cv
    joinedY = trainValTarget.copy(deep=True)

    #validation rule
    # validate = rolling/expanding/lastFold/5folds
    if validate == 'lastFold':  #validate on last actual data
        valLength = joinedX.shape[0]//5
        folds = -1*(joinedX.index < joinedX.index[-valLength])
        #-1 value if point in train ; 0 value if point in val fold 0, etc
        splitRule = PredefinedSplit(folds)
    if validate == '5folds': #usual 5fold cross validation with random folds
        folds = np.arange(joinedX.shape[0])%5  #5folds ie 1 for test, 4 for train
        random.seed(2022)
        random.shuffle(folds)
        splitRule = PredefinedSplit(folds)
    if validate == 'expanding': #expanding window cv
    #https://github.com/scikit-learn/scikit-learn/issues/22523
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    #https://towardsdatascience.com/model-validation-techniques-for-time-series-3518269bd5b3
        valLength = joinedX.shape[0]//5
        tscv = TimeSeriesSplit(
            n_splits=4, #if val is 20%, then we get [0..80]/[81..100]; [0..60]/[61..80]; [0..40]/[41..60]; [0..20]/[21..40]
            test_size=valLength
            )
        splitRule = tscv.split(np.arange(joinedX.shape[0]))
    if validate == 'rolling': # rolling window cv
        valLength = joinedX.shape[0]//5
        tscv = TimeSeriesSplit(
            n_splits=4, #if val is 20%, then we get [61..80]/[81..100]; [41..60]/[61..80]; [21..40]/[41..60]; [1..20]/[21..40]
            test_size=valLength,
            max_train_size=valLength
            )
        splitRule = tscv.split(np.arange(joinedX.shape[0]))

    #random search cross validation
    randomSearchCV= RandomizedSearchCV(estimator=model(),
                                  param_distributions=SEARCH_PARAMS,
                                  scoring='neg_mean_absolute_error', #negative mae
                                  cv=splitRule, #splitRule,
                                  n_iter=20, #how many configuration to check
                                  n_jobs=10,
                                  random_state=2022,
                                  return_train_score=True,
                                  refit=False #do not refit regressor with best params
                                 ) 
    randomSearchCV.fit(joinedX, joinedY.squeeze())
    cvResults = pd.DataFrame(randomSearchCV.cv_results_)

    #best params according to lowest val mae
    if bestVal == 'val':
        #bestEstimator = dt_random.best_estimator_ #trained estimator
        bestResult = cvResults[cvResults['rank_test_score']==1].iloc[0]
        bestParams = bestResult['params'] #same as dt_random.best_params_
    #params of the best model which overfits as less as possible and has low test error
    if 'top' in bestVal:
        topN = int(bestVal.split('top')[1])
        cvResults['absTrainValErrorDiff'] = (cvResults['mean_test_score']-cvResults['mean_train_score']).abs()
        lowestDiffTopN = cvResults.sort_values(by='absTrainValErrorDiff')[:topN]
        bestResult = lowestDiffTopN.sort_values(by='mean_test_score',ascending=False).iloc[0]  #False since scoring is neg mae
        bestParams = bestResult['params'] 

    bestEstimator = model(**bestParams)
    if validate == 'rolling':  #if rolling window cv is preferable (ie data shifts exist), then retraining the final model requires only most recent historical data
        valLength = joinedX.shape[0]//5
        refitX, refitY = joinedX[-valLength:], joinedY[-valLength:]  #refit on most recent data fold used in rolling window cv
    else:    #if expanding window cv is preferable, then retraining final model requires all historical data
        refitX, refitY = joinedX, joinedY  #refit on whole dataset; same as refit param does
        
    bestEstimator = bestEstimator.fit(refitX, refitY) 
    featureImportance = bestEstimator.feature_importances_

    pred = pd.DataFrame(bestEstimator.predict(refitX),index=refitY.index,columns=['prediction'])
    trainValMae = mean_absolute_error(pred[pred.index.isin(refitY.index)],refitY)
    trainValR2 = r2_score(refitY, pred[pred.index.isin(refitY.index)])
    trainValWmape = wmape(refitY, pred[pred.index.isin(refitY.index)]) 

    if verbose==True: print(f'training time {(time.time()-start):.3f}s')
    
    return {'bestEstimator':bestEstimator,
            'bestScaler': scaler,
            'top5Features': plotImportance(featureImportance,joinedX.columns,5), #5 most important features
            'scoreVal': - bestResult['mean_test_score'],
            'scoreTrain': - bestResult['mean_train_score'],
            'cvResults': cvResults,
            'bestParams': bestParams,
            'scoreTrainVal': [trainValMae, trainValR2, trainValWmape],
            'predictionTrainVal': pred
            }

#-----prediction and diagnostics

def showPrediction(features,modelData,target=pd.DataFrame(),mode='trainVal',verbose_img=False,n=24*30*6):
    #validation metrics are in fitRegressor; here only test metrics
    Mae, R2, Wmape = None, None, None
    if mode == 'trainVal':  #model diagnostic on trainVal
        pred = modelData['predictionTrainVal']
        truth = target[target.index.isin(modelData['predictionTrainVal'].index)]
        Mae = mean_absolute_error(truth,pred)
        R2 = r2_score(truth, pred)
        Wmape = wmape(truth, pred) 
    if mode == 'test': #case for test data
        featuresScaled = pd.DataFrame(modelData['bestScaler'].transform(features),index=features.index,columns=features.columns)
        pred = pd.DataFrame({'prediction':modelData['bestEstimator'].predict(featuresScaled)},index=features.index)
        if target.shape[0] != 0:
            truth = target
            Mae = mean_absolute_error(truth,pred)
            R2 = r2_score(truth, pred)
            Wmape = wmape(truth, pred) 

    if verbose_img==True:
            ax = truth[-n:].plot(color='r',figsize=(10,6))
            pred[-n:].plot(ax=ax,color='b')
            ax.set_title(f'Mae/R2/Wmape {np.round([Mae,R2,Wmape],3)}')
            plt.show()
    return {
        'prediction': pred,
        'score': [Mae,R2,Wmape]
        }

def showACF(target, modelData):
    targetName = target.columns[0]
    prediction = modelData['predictionTrainVal']
    predName = prediction.columns[0]
    dd = pd.concat([target[target.index.isin(prediction.index)],prediction],axis=1)
    dd['residual'] = dd[predName]-dd[targetName]

    fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(10,4))
    dd['residual'].plot(ax=axs[0],title='residuals')
    plot_acf(dd['residual'], ax=axs[1], lags = 60)
    axs[1].set_ylim(-0.2,0.2) 
    axs[1].axvline(24,c='r')
    axs[1].axvline(48,c='r')
    plot_pacf(dd['residual'],ax=axs[2], lags = 60)
    axs[2].set_ylim(-0.2,0.2)
    axs[2].axvline(24,c='r')
    axs[2].axvline(48,c='r')
    plt.tight_layout()
    plt.show()


