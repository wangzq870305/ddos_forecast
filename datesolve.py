#! /usr/bin/env python
#coding=utf-8
import datetime
import random

def getMonthDict():
    month=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    return dict([(month[i],i+1) for i in range(12)])

def getTweetDate(text,monthDict):
    try:
        p=text.split(',')
        year=int(p[1])
        pp=p[0].split(' ')
        day=int(pp[1])
        month=int(monthDict[pp[0]])
        
        return datetime.date(year,month,day)
    except:
        return None

def getEventDate(text):
    year=int(text[:4])
    month=int(text[4:6])
    day=int(text[6:8])
    return datetime.date(year,month,day)

def randomChoiseDate():
    date=datetime.date(2016,4,1)
    delta=datetime.timedelta(days=random.randint(10,180))
    date=date-delta
    return date

def getRandomDate(text):
    year=int(text[:4])
    month=int(text[5:7])
    day=int(text[8:])
    return datetime.date(year,month,day)
    