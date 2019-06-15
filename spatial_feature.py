# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:41:58 2019

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
    for i in tqdm(range(25)):
        filename = "G:/shanghaimotor/code/track_21/part" + str(i) + ".csv"
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

#计算日均驾驶时长
def driving_time(dataset):
    num_of_point = dataset.copy()
    add_timestamp(num_of_point)
    num_of_point = num_of_point.loc[:,('vin', 'latitude', 'longitude', 'year', 'month', 'day')]
    num_of_point = num_of_point.groupby(['vin', 'year', 'month', 'day']).count()
    num_of_point.reset_index(inplace=True)
    num_of_point = num_of_point.loc[:,('vin', 'latitude')]
    num_of_point = num_of_point.groupby('vin').mean()
    num_of_point.reset_index(inplace=True)
    num_of_point.columns = ['vin', 'driving_time']
    return(num_of_point)

#计算日均覆盖区域
def covering(dataset):
    covering = dataset.copy()
    add_timestamp(covering)
    covering['lat'] = covering['latitude'].apply(lambda x:round(x,2))
    covering['lgt'] = covering['longitude'].apply(lambda x:round(x,2))
    covering = covering.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
    covering = covering.loc[:,('vin', 'lat', 'lgt', 'year', 'month', 'day')]
    covering = covering.groupby(['vin', 'year', 'month', 'day']).count()
    covering.reset_index(inplace=True)
    covering = covering.loc[:,('vin', 'lat')]
    covering = covering.groupby('vin').mean()
    covering.reset_index(inplace=True)
    covering.columns = ['vin', 'covering']
    return(covering)

#计算最常访问地点的访问频次差异
def visiting_frequency(dataset):
    place = dataset.copy()
    add_timestamp(place)
    place['lat3'] = place['latitude'].apply(lambda x:round(x,3))
    place['lgt3'] = place['longitude'].apply(lambda x:round(x,3))
    place.drop(['latitude', 'longitude', 'speed'], axis=1, inplace=True)
    place = place.drop_duplicates(['vin', 'lat3', 'lgt3', 'year', 'month', 'day', 'hour'])
    place.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    place = place.groupby(['vin', 'lat3', 'lgt3']).count()
    place.columns = ['visiting_count']
    place.reset_index(inplace=True)
    user_place = pd.DataFrame(columns = ['vin', 'place_dif'])
    i = 0
    for vin in tqdm(place['vin'].unique()):
        dt_user = place[place['vin'] == vin]
        dt_user.sort_values('visiting_count', inplace=True, ascending=False)
        dt_user.reset_index(drop=True, inplace=True)
        dt_user['visiting_pro'] = dt_user['visiting_count'] / dt_user['visiting_count'].sum()
        user_place.loc[i, 'vin'] = vin
        if len(dt_user) == 1:
            user_place.loc[i, 'place_dif'] = dt_user.loc[0, 'visiting_pro']
        else:
            user_place.loc[i, 'place_dif'] = dt_user.loc[0, 'visiting_pro'] - dt_user.loc[1, 'visiting_pro']
        i += 1
    user_place['place_dif'] = user_place['place_dif'].apply(lambda x:float('%.4f' %x))
    return(user_place)

#计算球面距离
def haversine(lon1, lon2, lat1, lat2):
    #将十进制转化为弧度
    lon1, lon2, lat1, lat2 = map(radians, [lon1, lon2, lat1, lat2])
    #haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return(c * r)

#计算回旋半径        
def turning_radius(dataset):
    turning = dataset.copy()
    add_timestamp(turning)
    turning['lat3'] = turning['latitude'].apply(lambda x:round(x,3))
    turning['lgt3'] = turning['longitude'].apply(lambda x:round(x,3))
    turning.drop(['latitude', 'longitude'], axis=1, inplace=True)
    turning = turning.drop_duplicates(['vin', 'lat3', 'lgt3', 'year', 'month', 'day', 'hour'])
    turning.drop(['year', 'month', 'day', 'hour', 'speed'], axis=1, inplace=True)
    turning = turning.groupby(['vin', 'lat3', 'lgt3']).count()
    turning.columns = ['visiting_count']
    turning.reset_index(inplace=True)            
    home = pd.DataFrame(columns = ['vin', 'lat3_home', 'lgt3_home'])
    i = 0
    for vin in tqdm(turning['vin'].unique()):
        dt_user = turning[turning['vin'] == vin]
        dt_user.sort_values('visiting_count', ascending=False, inplace=True)
        dt_user.reset_index(drop=True, inplace=True)
        home.loc[i, 'vin'] = vin
        home.loc[i, 'lat3_home'] = dt_user.loc[0, 'lat3']
        home.loc[i, 'lgt3_home'] = dt_user.loc[0, 'lgt3']
        i += 1
    i =0 
    turning_radius = pd.DataFrame(columns=['vin', 'turning_radius'])
    for vin in tqdm(turning['vin'].unique()):
        dt_user = turning[turning['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        dt_user = pd.merge(dt_user, home, on='vin')
        l = len(dt_user)
        d_s = 0
        for j in range(l):
            d = haversine(dt_user.loc[j, 'lgt3_home'], dt_user.loc[j, 'lgt3'], dt_user.loc[j, 'lat3_home'], dt_user.loc[j, 'lat3'])
            d_s += d
        d_s /= l
        turning_radius.loc[i, 'vin'] = vin
        turning_radius.loc[i, 'turning_radius'] = d_s
        i += 1
    return(turning_radius)

#周日和周末平均出行时间的差异
def weekdays_weekends(dataset):
    ww = dataset.copy()
    add_timestamp(ww)
    ww['weekday'] = ww['record_time'].apply(lambda x:date.isoweekday(x))
    ww['isweekend'] = ww['weekday'].apply(lambda x:1 if((x==6)|(x==7)) else 0)
    ww.drop('weekday', axis=1, inplace=True)
    weekdays = ww[ww['isweekend'] == 0]
    weekends = ww[ww['isweekend'] == 1]
    weekdays = weekdays.groupby(['vin', 'year', 'month', 'day']).count()
    weekdays.reset_index(inplace=True)
    weekdays = weekdays.loc[:,('vin', 'record_time')]
    weekends = weekends.groupby(['vin', 'year', 'month', 'day']).count()
    weekends.reset_index(inplace=True)
    weekends = weekends.loc[:,('vin', 'record_time')]    
    weekdays = weekdays.groupby('vin').mean()
    weekends = weekends.groupby('vin').mean()
    weekdays.reset_index(inplace=True)
    weekends.reset_index(inplace=True)
    weekdays.columns = ['vin', 'weekdays_count']
    weekends.columns = ['vin', 'weekends_count']
    ww = pd.merge(weekdays, weekends, on='vin', how='outer')
    ww = ww.fillna(0)
    ww['weekdays_weekends'] = ww['weekdays_count'] - ww['weekends_count']
    ww = ww.loc[:,('vin', 'weekdays_weekends')]
    return(ww)

#生活熵
def life_entropy(dataset):
    life_en = dataset.copy()
    add_timestamp(life_en)
    life_en['lat'] = life_en['latitude'].apply(lambda x:float('%.2f' %x))
    life_en['lgt'] = life_en['longitude'].apply(lambda x:float('%.2f' %x))
    life_en = life_en.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
    life_en = life_en.loc[:,('vin', 'lat', 'lgt', 'year', 'month', 'day')]
    life_en = life_en.groupby(['vin', 'year', 'month', 'day']).count()
    life_en.reset_index(inplace=True)
    life_en = life_en.loc[:,('vin', 'month', 'lat')]    
    entropy = pd.DataFrame(columns = ['vin', 'life_entropy'])
    i = 0
    for vin in tqdm(life_en['vin'].unique()):
        dt_user = life_en[life_en['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        month_count = dt_user.groupby(['vin', 'month']).sum()
        month_count.reset_index(inplace=True)
        month_count.columns = ['vin', 'month', 'month_count']
        dt_user = pd.merge(dt_user, month_count, on=['vin', 'month'])
        dt_user['pro'] = dt_user['lat'] / dt_user['month_count']
        dt_user['entropy'] = dt_user['pro'].apply(lambda x:(-x*log(x)))
        dt_user = dt_user.groupby(['vin', 'month']).sum()
        dt_user.reset_index(inplace=True)
        dt_user = dt_user.groupby('vin').mean()
        dt_user.reset_index(inplace=True)
        dt_user.reset_index(drop=True, inplace=True)
        entropy.loc[i, 'vin'] = vin
        entropy.loc[i, 'life_entropy'] = dt_user.loc[0, 'entropy']
        i += 1
    return(entropy)

#计算日均行驶距离
def driving_distance(dataset):
    dis = dataset.copy()
    add_timestamp(dis)
    distance = pd.DataFrame(columns=['vin', 'distance', 'day'])
    i = 0
    for vin in tqdm(dis['vin'].unique()):
        dt_user = dis[dis['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        d = 0.0
        if len(dt_user) > 1:
            for j in range(len(dt_user)-1):
                time = (dt_user.loc[j+1, 'record_time'] - dt_user.loc[j, 'record_time']).seconds
                if time <= 600:
                    lon1 = dt_user.loc[j, 'longitude']
                    lon2 = dt_user.loc[j+1, 'longitude']
                    lat1 = dt_user.loc[j, 'latitude']
                    lat2 = dt_user.loc[j+1, 'latitude']
                    d += haversine(lon1, lon2, lat1, lat2)
                else:
                    continue
            distance.loc[i, 'vin'] = vin
            distance.loc[i, 'distance'] = d
            day_count = dt_user.drop_duplicates(['year', 'month', 'day'])
            distance.loc[i, 'day'] = len(day_count)
        else:
            distance.loc[i, 'vin'] = vin
            distance.loc[i, 'distance'] = 0
            distance.loc[i, 'day'] = 1 
        i += 1
    distance['dis_per_day'] = distance['distance'] / distance['day']
    return(distance)

#旅行次数
def driving_trip(dataset):
    t = dataset.copy()
    t['record_time'] = t['record_time'].apply(lambda x:str(x))
    t['record_time'] = t['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    trip = pd.DataFrame(columns = ['vin', 'trip'])
    i = 0
    for vin in tqdm(t['vin'].unique()):
        dt_user = t[t['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        count = 1
        for j in range(len(dt_user)-1):
            time = (dt_user.loc[j+1, 'record_time'] - dt_user.loc[j, 'record_time']).seconds
            if time > 1800:
                count += 1
            else:
                continue
        trip.loc[i, 'vin'] = vin
        trip.loc[i, 'trip'] = count
        i += 1
    return(trip)
    
#工作日和节假日出行天数的比例
def dayofweek_pro(dataset):
    loc = dataset.copy()
    loc = loc[loc['year'] == 2018]
    #flag = pd.read_csv(r"G:\shanghaimotor\code\flag2019.csv")
    #loc = loc[loc['vin'].isin(list(flag['vin']))]
    loc = loc.drop_duplicates(['vin', 'year', 'month', 'day'])
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    loc['dayofweek'] = loc['record_time'].apply(lambda x:date.isoweekday(x))
    dayofweek_day = pd.DataFrame(columns=['vin', 'weekdays', 'weekends', 'weekdays_pro', 'weekends_pro'])
    j = 0
    for vin in tqdm(loc['vin'].unique()):
        wd = 0
        wk = 0
        dt_user = loc[loc['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)):
            day = dt_user.loc[i, 'dayofweek']
            if (day == 6) | (day == 7):
                wk += 1
            else:
                wd += 1
        dayofweek_day.loc[j, 'vin'] = vin
        dayofweek_day.loc[j, 'weekdays'] = wd
        dayofweek_day.loc[j, 'weekends'] = wk
        dayofweek_day.loc[j, 'weekdays_pro'] = (wd / (wd + wk))
        dayofweek_day.loc[j, 'weekends_pro'] = (wk / (wd + wk))  
        j += 1
    return(dayofweek_day)
           

#每个月最常访问top10的地点访问频次
def most_visiting(loc):
    visiting = loc.copy()
    add_timestamp(visiting)
    month = pd.DataFrame(columns=['vin', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    i = 0
    for vin in tqdm(visiting['vin'].unique()):
        dt_user = visiting[visiting['vin']==vin]
        dt_user['lat'] = dt_user['latitude'].apply(lambda x:round(x, 2))
        dt_user['lgt'] = dt_user['longitude'].apply(lambda x:round(x, 2))
        dt_user.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'], inplace=True)
        dt_user.reset_index(drop=True, inplace=True)
        dt_user.drop(['latitude', 'longitude', 'speed', 'record_time', 'year', 'day', 'hour', 'speed'], axis=1, inplace=True)
        dt_user = dt_user.groupby(['month', 'lat', 'lgt']).count()
        dt_user.reset_index(inplace=True)
        for m in dt_user['month'].unique():
            dt_user_month = dt_user[dt_user['month']==m]
            dt_user_month.sort_values(['vin'], ascending=False, inplace=True)
            dt_user_month.reset_index(drop=True, inplace=True)
            if (len(dt_user_month)) <= 10:
                month.loc[i, str(m)] = 1
            else:
                top = dt_user_month.loc[:9]
                p = top['vin'].sum() / dt_user_month['vin'].sum()
                month.loc[i, str(m)] = p
        month.loc[i, 'vin'] = vin
        i += 1
    month = month.astype(float)
    for i in range(len(month)):
        t1 = month.iloc[i,1:12]
        t2 = t1.mean()
        month.loc[i, 'mean'] = t2
    return(month)

#周末是否经常去不经常去的地方
def weekends_travelling(dataset):
    loc = dataset.copy()
    loc = loc[loc['year'] == 2018]
    add_timestamp(loc)
    flag = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    loc = loc[loc['vin'].isin(list(flag['vin']))]    
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    loc['dayofweek'] = loc['record_time'].apply(lambda x:date.isoweekday(x))
    loc['lat'] = loc['latitude'].apply(lambda x:round(x, 2))
    loc['lgt'] = loc['longitude'].apply(lambda x:round(x, 2))
    weekends_loc = loc[loc['dayofweek'].isin([6, 7])]
    weekends_visiting = pd.DataFrame(columns = ['vin', 'weekends_visiting'])
    i = 0
    for vin in tqdm(list(flag['vin'])):
        dt_user_weekends = weekends_loc[weekends_loc['vin'] == vin]
        dt_user = weekends_loc[weekends_loc['vin'] == vin]
        if len(dt_user_weekends) == 0:
            weekends_visiting.loc[i, 'weekends_visiting'] = 1
        else:
            dt_user = dt_user.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
            month_weekends = []
            for m in dt_user['month'].unique():
                dt_user_weekends_month = dt_user_weekends[dt_user_weekends['month'] == m]
                dt_user_weekends_month = dt_user_weekends_month[['vin', 'year', 'month', 'day', 'lat', 'lgt']]
                dt_user_weekends_month = dt_user_weekends_month.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
                dt_user_weekends_month = dt_user_weekends_month[['lat', 'lgt']]
                dt_user_month = dt_user[dt_user['month'] == m]
                dt_user_month = dt_user_month.groupby(['lat', 'lgt']).count()
                month_visiting = dt_user_month.reset_index()
                month_visiting.sort_values('vin', ascending=False, inplace=True)
                if len(month_visiting) <= 10:
                    month_visiting = month_visiting.iloc[:,:3]
                else:
                    month_visiting = month_visiting.iloc[:10,:3]
                
                month_visiting1 = pd.merge(month_visiting, dt_user_weekends_month, on=['lat', 'lgt'], how='outer')
                month_visiting1 = month_visiting1.fillna(0)
                month_visiting1['tag'] = month_visiting1['vin'].apply(lambda x:1 if(x==0) else 0)
                month_weekends.append(np.mean(month_visiting1['tag']))
            weekends_visiting.loc[i, 'vin'] = vin
            weekends_visiting.loc[i, 'weekends_visiting'] = np.mean(month_weekends)
        i += 1
    return(weekends_visiting)
'''    
#统计疲劳驾驶的次数
def fatigue_driving_count(dataset):
    h4_driving = dataset.copy()
    fatigue_driving = pd.DataFrame(columns=['vin', 'fat_driving_count'])
    k = 0
    for vin in tqdm(h4_driving['vin'].unique()):
        s = 0
        dt_user = h4_driving[h4_driving['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        j = 1
        for i in range(0,len(dt_user),j):
            start_time = dt_user.loc[i, 'record_time']
            for j in range(i, len(dt_user)):
                if (dt_user.loc[j, 'record_time'] - dt_user.loc[j, 'record_time']).seconds >= 1800:
                    break
                else:
                    end_time = dt_user.loc[j, 'record_time']
                if (end_time - start_time).seconds >= 9600:
                    s += 1
                    break
            if j == 0:
                j = 1
        fatigue_driving.loc[k, 'vin'] = vin
        fatigue_driving.loc[k, 'fat_driving_count'] = s
        k += 1  
    return(fatigue_driving) 
'''
#统计疲劳驾驶的次数
def fatigue_driving_count(dataset):
    loc = dataset.copy()
    loc = loc[loc['year'] == 2018]
    #flag = pd.read_csv(r"G:\shanghaimotor\code\flag2019.csv")
    #loc = loc[loc['vin'].isin(list(flag['vin']))]
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))    
    k = 0
    fatigue_driving = pd.DataFrame(columns=['vin', 'fat_driving_count'])
    for vin in tqdm(loc['vin'].unique()):
        dt_user = loc[loc['vin'] ==  vin]
        dt_user.reset_index(drop=True, inplace=True)
        dt_user['seconds'] = dt_user['record_time'].apply(lambda x:mktime(x.timetuple()))
        i = 0
        count = 0
        while(i < len(dt_user)-1):
            start_time = dt_user.loc[i, 'seconds']
            for j in range(i, len(dt_user)-1):
                if (dt_user.loc[j+1, 'seconds'] - dt_user.loc[j, 'seconds']) < 1800.0:
                    continue
                else:
                    end_time = dt_user.loc[j, 'seconds']
                    if (end_time - start_time) >= 14400.0:
                        count += 1
                    break
            i = j+1
        fatigue_driving.loc[k, 'vin'] = vin
        fatigue_driving.loc[k, 'fat_driving_count'] = count
        k += 1
    return(fatigue_driving)
                        

#每辆车发生事故的概率
def accident_rate(dataset):
    loc = dataset.copy()
    #add_timestamp(loc)
    crash = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    #crash = crash[crash['flag']==1]
    add_crash_timestamp(crash)
    crash_place = pd.merge(loc, crash, on=['vin', 'year', 'month', 'day', 'hour'])
    crash_place['lat'] = crash_place['latitude'].apply(lambda x:round(x, 2))
    crash_place['lgt'] = crash_place['longitude'].apply(lambda x:round(x, 2))
    crash_place['time'] = (crash_place['crash_time'] - crash_place['record_time'])
    crash_place['time'] = crash_place['time'].apply(lambda x:int(x.seconds))
    crash_place = crash_place[crash_place['time'] < 600]
    crash_place.drop_duplicates('vin', keep='last', inplace=True)
    crash_place = crash_place[['vin', 'lat', 'lgt']]
    accident_account = crash_place.groupby(['lat', 'lgt']).count()
    accident_account.reset_index(inplace=True)
    loc['lat'] = loc['latitude'].apply(lambda x:round(x, 2))
    loc['lgt'] = loc['longitude'].apply(lambda x:round(x, 2))
    loc_account = loc.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
    loc_account = loc_account[['vin', 'lat', 'lgt']]
    loc_account = loc_account.groupby(['lat', 'lgt']).count()
    loc_account.reset_index(inplace=True)
    accident_account.columns = ['lat', 'lgt', 'accident_account']
    loc_account.columns = ['lat', 'lgt', 'loc_account']
    accident_rate = pd.merge(accident_account, loc_account, on=['lat', 'lgt'])
    accident_rate['accident_rate'] = accident_rate['accident_account'] / accident_rate['loc_account']
    accident_rate = accident_rate[accident_rate['lat'] > 30.40]
    accident_rate = accident_rate[accident_rate['lat'] < 31.53]
    accident_rate = accident_rate[accident_rate['lgt'] < 122.12]
    accident_rate = accident_rate[accident_rate['lgt'] > 121]
    accident_rate = accident_rate[['lat', 'lgt', 'accident_rate']]
    loc_account = loc[~loc['speed'].isin([0])]
    loc_account = loc_account.drop_duplicates(['vin', 'year', 'month', 'day', 'hour', 'lat', 'lgt'])
    loc_account = loc_account[['vin', 'lat', 'lgt']]
    loc_account = pd.merge(loc_account, accident_rate, on=['lat', 'lgt'])
    loc_account = loc_account[['vin', 'accident_rate']]
    loc_account = loc_account.groupby('vin').sum()
    loc_account.reset_index(inplace=True)
    return(loc_account)


#计算每个区域车辆的平均速度
def speed_limit(dataset):
    speed = dataset.copy()
    speed['lat'] = speed['latitude'].apply(lambda x:round(x, 2))
    speed['lgt'] = speed['longitude'].apply(lambda x:round(x, 2))
    speed = speed[['year', 'month', 'lat', 'lgt', 'speed']]
    speed.dropna(how='any', inplace=True)
    speed = speed[~speed['speed'].isin([0])]
    speed = speed.groupby(['year', 'month', 'lat', 'lgt']).mean()
    speed.reset_index(inplace=True)
    speed = speed[speed['lat'] > 30.40]
    speed = speed[speed['lat'] < 31.53]
    speed = speed[speed['lgt'] < 122.12]
    speed = speed[speed['lgt'] > 121]
    speed.columns = ['year', 'month', 'lat', 'lgt', 'speed_limit']
    return(speed)        

def overspeed(dataset, speed_limit):
    loc = dataset.copy()
    #add_timestamp(loc)
    loc['lat'] = loc['latitude'].apply(lambda x:round(x, 2))
    loc['lgt'] = loc['longitude'].apply(lambda x:round(x, 2))
    loc_account = loc[~loc['speed'].isin([0])]
    loc_account = loc_account[['vin', 'year', 'month', 'lat', 'lgt', 'speed']]
    loc_account = loc_account.groupby(['vin', 'year', 'month', 'lat', 'lgt']).mean()
    loc_account.reset_index(inplace=True)
    speed = pd.merge(loc_account, speed_limit, on=['year', 'month', 'lat', 'lgt'])
    speed['over_speed'] = speed['speed'] - speed['speed_limit']
    speed['over_speed_tag'] = speed['over_speed'].apply(lambda x:0 if(x<0) else 1)
    speed = speed[['vin','over_speed_tag']]
    speed = speed.groupby('vin').mean()
    speed.reset_index(inplace=True)
    return(speed)    
'''
#计算不同速度区间车辆的行驶里程
def km_accident(dataset):
    loc = dataset.copy()
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))    
    speed0 = loc[(loc['speed']>=0) & (loc['speed']<20)]
    speed1 = loc[(loc['speed']>=20) & (loc['speed']<50)]
    speed2 = loc[(loc['speed']>=50) & (loc['speed']<80)]
    speed3 = loc[(loc['speed']>=80) & (loc['speed']<130)]
    speed4 = loc[(loc['speed']>=130)]
    ksum0 = 0
    ksum1 = 0
    ksum2 = 0
    ksum3 = 0
    ksum4 = 0

    for vin in tqdm(speed0['vin'].unique()):
        dt_user = speed0[speed0['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)-1):
            time = (dt_user.loc[i+1, 'record_time'] - dt_user.loc[i, 'record_time']).seconds
            if time <= 120:
                ksum0 += haversine(dt_user.loc[i+1, 'longitude'], dt_user.loc[i, 'longitude'],
                                   dt_user.loc[i+1, 'latitude'], dt_user.loc[i, 'latitude'])
            else:
                continue
    for vin in tqdm(speed1['vin'].unique()):
        dt_user = speed1[speed1['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)-1):
            time = (dt_user.loc[i+1, 'record_time'] - dt_user.loc[i, 'record_time']).seconds
            if time <= 120:
                ksum1 += haversine(dt_user.loc[i+1, 'longitude'], dt_user.loc[i, 'longitude'],
                                   dt_user.loc[i+1, 'latitude'], dt_user.loc[i, 'latitude'])
            else:
                continue
    for vin in tqdm(speed2['vin'].unique()):
        dt_user = speed2[speed2['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)-1):
            time = (dt_user.loc[i+1, 'record_time'] - dt_user.loc[i, 'record_time']).seconds
            if time <= 120:
                ksum2 += haversine(dt_user.loc[i+1, 'longitude'], dt_user.loc[i, 'longitude'],
                                   dt_user.loc[i+1, 'latitude'], dt_user.loc[i, 'latitude'])
            else:
                continue   
    for vin in tqdm(speed3['vin'].unique()):
        dt_user = speed3[speed3['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)-1):
            time = (dt_user.loc[i+1, 'record_time'] - dt_user.loc[i, 'record_time']).seconds
            if time <= 120:
                ksum3 += haversine(dt_user.loc[i+1, 'longitude'], dt_user.loc[i, 'longitude'],
                                   dt_user.loc[i+1, 'latitude'], dt_user.loc[i, 'latitude'])
            else:
                continue       
    for vin in tqdm(speed4['vin'].unique()):
        dt_user = speed4[speed4['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)-1):
            time = (dt_user.loc[i+1, 'record_time'] - dt_user.loc[i, 'record_time']).seconds
            if time <= 120:
                ksum4 += haversine(dt_user.loc[i+1, 'longitude'], dt_user.loc[i, 'longitude'],
                                   dt_user.loc[i+1, 'latitude'], dt_user.loc[i, 'latitude'])
            else:
                continue      
    return(ksum0, ksum1, ksum2, ksum3, ksum4)
'''    
def weather_feature(dataset, wea):
    loc = dataset.copy()
    #add_timestamp(loc)
    loc = loc[loc['latitude'] > 30.40]
    loc = loc[loc['latitude'] < 31.53]
    loc = loc[loc['longitude'] < 122.12]
    loc = loc[loc['longitude'] > 120.51]   
    loc = loc[loc['year'] == 2018]
    loc.drop_duplicates(['vin', 'year', 'month', 'day'], inplace=True)
    loc = loc[['vin', 'year', 'month', 'day']]
    loc = pd.merge(loc, wea, on=['year', 'month', 'day'])
    loc = loc.groupby('vin').mean()
    loc.reset_index(inplace=True)
    loc = loc[['vin', 'cond', 'temp']]
    return(loc)
                  
    

if __name__ == '__main__':
    #result = data_preprocessing()
    #result = result[result['year'] == 2018]
    #to_csv(result)
    result = read_dataset()
    #result = flag2_2019(result)
    '''
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    flag1 = flag1[flag1['year'] == 2018]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]
    #add_timestamp(result)
    result = result[result['year'] == 2018]
    #result = pd.merge(result, flag1, on='vin')
    #result = result[result['month'] < result['crash_month']]
    #result = result[result['vin'].isin(list(flag2018['vin']))]
    #result.drop(['crash_year', 'crash_month'], axis=1, inplace=True)
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    gc.collect()
    '''
    
    #result.reset_index(drop=True, inplace=True)
    #result2018 = result[result['year'] == 2018]
    #result2018 = result2018[result2018['month'] <= 6]
    #result.reset_index(drop=True, inplace=True)
    #add_timestamp(result)
    #to_csv(result)
    
    
    #只分析2018年的轨迹数据
    #result2018 = result2018[result2018['year'] == 2018]
    #result2018.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    
    #driving_time = driving_time(result2018)
    #covering = covering(result2018)
    #place_dif = visiting_frequency(result2018)
    
    #turning_radius = turning_radius(result2018)
    #entropy = life_entropy(result2018)
    #weekdays_weekends = weekdays_weekends(result2018)
    #distance = driving_distance(result2018)
    #trip = driving_trip(result2018)
    #top_frq = most_visiting(result2018)
    #accident = accident_rate(result2018)
    #speed_limit = speed_limit(result2018)
    #over_speed = overspeed(result2018, speed_limit)
    #ksum0, ksum1, ksum2, ksum3, ksum4 = km_accident(result2018)
    #weather = weather_feature(result2018, wea)    
    
    #删除出行天数少于30天的车辆
    #ww_day_pro = dayofweek_pro(result2018)
    #ww_day_pro = ww_day_pro[(ww_day_pro['weekdays'] + ww_day_pro['weekends']) >= 5]
    #疲劳驾驶
    #fatigue = fatigue_driving_count(result2018)
    #weekends_visiting = weekends_travelling(result2018)
    '''
    flag2 = flag2[flag2['year'] == 2018]
    flag2.drop(['crash_time', 'day', 'flag'], axis=1, inplace=True)
    flag2.columns = ['vin', 'crash_year', 'crash_month']
    result = pd.merge(result, flag2, on='vin')
    result = result[result['year'] == 2018]
    result = result[result['month'] < result['crash_month']]
    result.drop(['crash_year', 'crash_month'], axis=1, inplace=True)
    '''
    '''
    driving_time = driving_time(result)
    driving_time.to_csv("G:/shanghaimotor/code/feature_1/driving_time.csv", index=False)
    #covering = covering(result)
    #covering.to_csv("G:/shanghaimotor/code/feature_1/covering.csv", index=False)
    place_dif = visiting_frequency(result)
    place_dif.to_csv("G:/shanghaimotor/code/feature_1/place_dif.csv", index=False)
    turning_radius = turning_radius(result)
    turning_radius.to_csv("G:/shanghaimotor/code/feature_1/turning_radius.csv", index=False)
    entropy = life_entropy(result)
    entropy.to_csv("G:/shanghaimotor/code/feature_1/life_entropy.csv", index=False)
    weekdays_weekends = weekdays_weekends(result)
    weekdays_weekends.to_csv("G:/shanghaimotor/code/feature_1/weekdays_weekends.csv", index=False)
    distance = driving_distance(result)
    distance.to_csv("G:/shanghaimotor/code/feature_1/distance.csv", index=False)
    trip = driving_trip(result)
    trip.to_csv("G:/shanghaimotor/code/feature_1/trip.csv", index=False)
    top_frq = most_visiting(result)
    top_frq.to_csv("G:/shanghaimotor/code/feature_1/top10_frq.csv", index=False)
    #accident = accident_rate(result)
    ww_day_pro = dayofweek_pro(result)
    ww_day_pro.to_csv("G:/shanghaimotor/code/feature_1/weekdays_weekends_count.csv", index=False)
    fatigue = fatigue_driving_count(result)
    fatigue.to_csv("G:/shanghaimotor/code/feature_1/fatigue_driving.csv", index=False)
    weekends_visiting = weekends_travelling(result)
    weekends_visiting.to_csv("G:/shanghaimotor/code/feature_1/weekends_visiting.csv", index=False)
    '''
    #night_driving_pro = night_driving_pro(result)
    #night_driving_time = night_driving_time(result)
    #night_driving_day = night_driving_day(result)
    
    
    
    
    
    
    
    
    
