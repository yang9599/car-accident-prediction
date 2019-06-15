# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:20:06 2019

@author: Administrator
"""

import pandas as pd
import numpy as np 
from datetime import datetime, date
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt, log
from crash_time import add_crash_timestamp
from time import mktime
import gc

def data_preprocessing():
    flag = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    user = list(flag['vin'])
    number = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14']
    result = pd.DataFrame()
    for num in tqdm(number):
        filename = 'G:/shanghaimotor/dataset/loc/part-000'
        filename += num
        dataset = pd.read_csv(filename, names=['vin', 'latitude', 'longitude', 'speed','heading_val',
                                               'ignition_on_flag', 'record_time', 'i_flag', 'ignition_on_flag_new'])
        dataset = dataset.loc[:,('vin', 'latitude', 'longitude', 'speed', 'record_time')]
        dataset = dataset.astype(np.str)
        #数据截取
        dataset['vin'] = dataset['vin'].apply(lambda x:((str(x)).split("=")))
        dataset['vin'] = dataset['vin'].apply(lambda x:x[1])
        dataset['vin'] = dataset['vin'].astype('int32')
        dataset = dataset[dataset['vin'].isin(user)]
        #dataset = dataset[dataset['vin'].isin(user)]
        dataset['nan'] = dataset['latitude'].apply(lambda x:str(x)[-4:])
        dataset['none'] = dataset['nan'].apply(lambda x:1 if(x=='None') else 0)
        dataset = dataset[dataset['none'].isin([0])]
        dataset.drop(['nan', 'none'], axis=1, inplace=True)
        dataset['latitude'] = dataset['latitude'].apply(lambda x:(str(x)).split("'"))
        dataset['latitude'] = dataset['latitude'].apply(lambda x:x[1])
        dataset['longitude'] = dataset['longitude'].apply(lambda x:(str(x)).split("'"))
        dataset['longitude'] = dataset['longitude'].apply(lambda x:x[1])
        dataset['speed1'] = dataset['speed'].apply(lambda x:(str(x)).replace("None", "u'None'"))
        dataset['speed1'] = dataset['speed1'].apply(lambda x:(str(x)).split("'"))
        dataset['speed'] = dataset['speed1'].apply(lambda x:x[1])
        dataset['record_time'] = dataset['record_time'].apply(lambda x:(str(x)).split("'"))
        dataset['record_time'] = dataset['record_time'].apply(lambda x:x[1])
        dataset.drop('speed1', axis=1, inplace=True)
        dataset = dataset[dataset['speed'] != 'None']
        #数据类型转换

        dataset['latitude'] = dataset['latitude'].astype('float32')
        dataset['longitude'] = dataset['longitude'].astype('float32')
        dataset['speed'] = dataset['speed'].astype('float16')
        dataset = dataset[dataset['latitude'] < 500]
        dataset = dataset[dataset['latitude'] > 10]
        dataset = dataset[dataset['longitude'] < 500]
        dataset['record_time'] = dataset['record_time'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M"))
        result = result.append(dataset, ignore_index=True)
    return(result)
    
def to_csv(dataset):
    l = len(dataset) // 10000000
    k1 = 0 
    k2 = 10000000
    for i in range(l):
        result = dataset.loc[k1:k2]
        result.to_csv("G:/shanghaimotor/code/track_21/part"+str(i)+".csv", index=False)
        add_timestamp(result)
        k1 += 10000000
        k2 += 10000000
        gc.collect()
    result = dataset.loc[k1:]
    add_timestamp(result)
    result.to_csv("G:/shanghaimotor/code/track_21/part"+str(l)+".csv", index=False)

    
def read_dataset():
    result = pd.DataFrame()
    for i in tqdm(range(9)):
        filename = "G:/shanghaimotor/code/track_0/part" + str(i) + ".csv"
        dataset = pd.read_csv(filename)
        #add_timestamp(dataset)
        result = result.append(dataset, ignore_index=True)
    result = result[result['latitude'] < 54]
    result = result[result['latitude'] > 10]
    result = result[result['longitude'] < 136]
    result = result[result['longitude'] > 73]
    #result.sort_values(['vin', 'record_time'], inplace=True)
    return(result)    

#2018年发生碰撞的车辆只计算发生碰撞前的特征
def flag2_2018(dataset):
    result = dataset.copy()
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    flag1 = flag1[flag1['year'] == 2018]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]    
    result = result[result['year'] == 2018]
    result = pd.merge(result, flag1, on='vin')
    result = result[result['month'] < result['crash_month']]
    result.drop(['crash_year', 'crash_month'], axis=1, inplace=True)
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return(result)
    
def flag2_2019(dataset):
    result = dataset.copy()
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    flag1 = flag1[flag1['year'] == 2019]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]    
    result = result[result['year'] == 2018]
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return(result)

def flag1_2018(dataset):
    result = dataset.copy()
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag1.csv")
    flag1 = flag1[flag1['year'] == 2018]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]    
    result = result[result['year'] == 2018]
    result = pd.merge(result, flag1, on='vin')
    result = result[result['month'] < result['crash_month']]
    result.drop(['crash_year', 'crash_month'], axis=1, inplace=True)
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return(result)
    
#增加年份、月份、天、小时等信息
def add_timestamp(dataset):
    dataset['record_time'] = dataset['record_time'].astype(str)
    dataset['record_time'] = dataset['record_time'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    dataset['year'] = dataset['record_time'].apply(lambda x:x.year)
    dataset['month'] = dataset['record_time'].apply(lambda x:x.month)
    dataset['day'] = dataset['record_time'].apply(lambda x:x.day)
    dataset['hour'] = dataset['record_time'].apply(lambda x:x.hour)
    
#夜间驾驶的总时间占比
def night_driving_pro(dataset):
    loc = dataset.copy()
    loc['night'] = loc['hour'].apply(lambda x:1 if(((x>=0)&(x<6))|(x>=18)&(x<24)) else 0)
    loc = loc[['vin', 'night']]
    loc = loc.groupby('vin').mean()
    loc.reset_index(inplace=True)
    return(loc)

#平均每次夜间驾驶的时长
def night_driving_time(dataset):
    loc = dataset.copy()
    loc['night'] = loc['hour'].apply(lambda x:1 if(((x>=0)&(x<6))|(x>=18)&(x<24)) else 0)
    loc = loc[loc['night'] == 1]
    night_driving_time = pd.DataFrame(columns=['vin', 'night_driving_time'])
    k = 0
    for vin in tqdm(loc['vin'].unique()):
        dt_user = loc[loc['vin'] == vin]
        dt_user.sort_values('record_time', inplace=True)
        dt_user.reset_index(drop=True, inplace=True)            
        dt_user['record_time'] = dt_user['record_time'].astype(str)
        dt_user['record_time'] = dt_user['record_time'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
        time = []
        start_time = dt_user.loc[0, 'record_time']
        if len(dt_user) < 1:
            continue
        for i in range(len(dt_user)-1):
            if (dt_user.loc[i+1, 'record_time'] - dt_user.loc[i, 'record_time']).seconds <= 600:
                continue
            else:
                end_time = dt_user.loc[i, 'record_time']
                t = int((end_time-start_time).seconds / 60)
                if t != 0:
                    time.append(t)
                start_time = dt_user.loc[i+1, 'record_time']
        night_driving_time.loc[k, 'vin'] = vin
        night_driving_time.loc[k, 'night_driving_time'] = float(np.mean(time))
        k += 1
    return(night_driving_time)

#夜间驾驶天数占比
def night_driving_day(dataset):
    loc = dataset.copy()
    loc['night'] = loc['hour'].apply(lambda x:1 if(((x>=0)&(x<6))|(x>=18)&(x<24)) else 0)     
    night_day = loc[loc['night'] == 1]
    night_day = night_day.drop_duplicates(['vin', 'year', 'month', 'day'])
    night_day = night_day.groupby('vin').count()
    night_day.reset_index(inplace=True)
    night_day = night_day.iloc[:,:2]
    night_day.columns=['vin', 'night_day']
    day = loc.drop_duplicates(['vin', 'year', 'month', 'day'])
    day = day.groupby('vin').count()
    day.reset_index(inplace=True)
    day = day.iloc[:,:2]
    day.columns=['vin', 'day']
    night_day = pd.merge(night_day, day, on='vin', how='outer')
    night_day = night_day.fillna(0)
    #night_day = night_day[night_day['day']>=5]
    night_day['night_day_pro'] = night_day['night_day'] / night_day['day']
    night_day = night_day[['vin', 'night_day_pro']]
    return(night_day)         

if __name__ == "__main__":
    
    result = read_dataset()
    result = result[result['year']==2018]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    