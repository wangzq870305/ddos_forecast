#! /usr/bin/env python
#coding=utf-8
from __future__ import division
from cedict import CEDict
import numpy as np
import random
from datesolve import *
from event import *
import datetime

ceDict=CEDict()
MONTHS=set(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])

class CDocument:
    def __init__(self,words,polarity,id,text,eventName):
        self.words=words
        self.polarity=polarity
        if polarity==True:
            self.label=1
        else:
            self.label=0
            
        self.id=id
        self.text=text
        self.eventName=eventName

def readTweets(path):
    tweets=[]
    monthDict=getMonthDict()
    for line in open(path,'rb'):
        line=line.strip()
        if len(line)>0:
            line=line.lower()
            p=line.split('\t')
            if len(p)==3:
                id,date,text=p
                date=getTweetDate(date,monthDict)
                if date!=None:
                    tweets.append(Tweet(int(id),date,text))
    print len(tweets)
    return tweets

def readEventList():
    events=[]
    id=0
    for line in open(r'data/target-event.txt'):
        line=line.strip()
        if len(line)>0:
            p=line.split('\t')
            if len(p)==2:
                name,date=p
                date=getEventDate(date)
                events.append(Event(id,date,name))
            id+=1
    print len(events)
    return events

def unigram(text):
    words={}
    for w in text.split():
        words[w]=1
    return words

def readRandomDate():
    dates=[]
    for line in open(r'data/random-date.txt','rb'):
        line=line.strip()
        if len(line)>0:
            dates.append(getRandomDate(line))
    return dates

def readIDList():
    idList=[]
    for line in open(r'data/id-list.txt','rb'):
        idList.append(int(line))
    return idList

def isEnglishWord(word):
    cwords=ceDict.getChinese(word)
    if len(cwords)>0:
        if word.lower() in MONTHS:
            return False
        else:
            for c in word.lower():
                if ord(c)<97 or ord(c)>122:
                    return False            
        return True
    else:
        return False

def getVocabrary(documents):
    V={}
    for d in documents:
        for w in d.words:
            if w not in V:
                V[w]=len(V)
    return V
    
def DF(documents,n=10):
    df={}
    for d in documents:
        for w in d.words:
            if w not in df:
                df[w]=0
            df[w]+=1
    for d in documents:
        for w in d.words.keys():
            if df[w]<n:
                del d.words[w] 

def formatK(data,V):
    X=[]
    Y=[]
    
    for i,d in enumerate(data):
        x=[]
        for w in d.words:
            x.append(V[w])
        X.append(x)
        Y.append(d.label)
    return X,Y

def getPositiveSamples_ranges(tweets,events,idList,week_num,date_range=7):
    idSet=set(idList)
    eventDict=dict([(e.id,e.date) for e in events]) # ID,Date    
    eventDict2=dict([(e.id,e.name) for e in events]) # ID,Name
 
    documentsList=[]
    for i in range(week_num):
        beginRange=i*date_range
        endRange=(i+1)*date_range
        d={}
        for tweet in tweets:
            if tweet.id in eventDict and tweet.id in idSet:
                date=eventDict[tweet.id]
                delta=date-tweet.date
                if delta.days>beginRange and delta.days<=endRange:
                    if tweet.id not in d:
                        d[tweet.id]=[]
                    d[tweet.id].append(tweet)
        documents=[]
        for id in d:
            text=''
            for tweet in d[id]:
                text+=' '+tweet.text
            documents.append(CDocument(unigram(text),True,id,text,eventDict2[id]))
        documentsList.append(documents)
        
    idSet=set([d.id for d in documentsList[0]])
    for documents in documentsList:
        idSet0=set([d.id for d in documents])
        idSet=idSet&idSet0
    
    newDocumentsList=[]
    for documents in documentsList:
        idDict=dict([(d.id,d) for d in documents])
        newDocumentsList.append([idDict[id] for id in idSet])
    
    return newDocumentsList

def getNegativeSamples_ranges(tweets,date,events,idList,week_num,date_range):
    idSet=set(idList)
    eventDict=dict([(e.id,e.date) for e in events])   
    eventDict2=dict([(e.id,e.name) for e in events])

    documentsList=[]
    for i in range(week_num):
        beginRange=i*date_range
        endRange=(i+1)*date_range
        d={}
        for tweet in tweets:
            delta=date-tweet.date
            deltaEvent=eventDict[tweet.id]-date
            if deltaEvent.days!=0 and delta.days>beginRange and delta.days<=endRange and tweet.id in idSet:
                if tweet.id not in d:
                    d[tweet.id]=[]
                d[tweet.id].append(tweet)
        
        documents=[]
        for id in d:
            text=''
            for tweet in d[id]:
                text+=' '+tweet.text
            documents.append(CDocument(unigram(text),False,id,text,eventDict2[id]))
        
        documentsList.append(documents)
        
    idSet=set([d.id for d in documentsList[0]])
    for documents in documentsList:
        idSet0=set([d.id for d in documents])
        idSet=idSet&idSet0
    
    newDocumentsList=[]
    for documents in documentsList:
       # print len(documents)
        idDict=dict([(d.id,d) for d in documents])
        newDocumentsList.append([idDict[id] for id in idSet])
    
    return newDocumentsList

def initData(week_num,date_range):
    neg=readTweets(r'data/negative.txt')
    pos=readTweets(r'data/positive.txt')
    events=readEventList()
    
    negDates=readRandomDate()
    idList=readIDList()
    
    trainIDs=idList[:80]
    devIDs=idList[80:140]
    testIDs=idList[140:]

    print 'length of pos and neg tweets: %s %s' %(len(pos),len(neg))
    
    posTrainList=getPositiveSamples_ranges(pos,events,trainIDs,week_num,date_range)
    devList=getPositiveSamples_ranges(pos,events,devIDs,week_num,date_range)
    testList=getPositiveSamples_ranges(pos,events,testIDs,week_num,date_range)
    
    negTrainList=[[] for i in range(week_num)]
    for i in range(10,20):
        negTrains=getNegativeSamples_ranges(neg,negDates[i+10],events,trainIDs,week_num,date_range)
        negDev=getNegativeSamples_ranges(neg,negDates[i],events,devIDs,week_num,date_range)
        negTests=getNegativeSamples_ranges(neg,negDates[i],events,testIDs,week_num,date_range)
        
        for j in range(week_num):
            negTrainList[j]+=negTrains[j]
            devList[j]+=negDev[j]
            testList[j]+=negTests[j]
    
    trainList=[[] for i in range(week_num)]
    for i in range(week_num):
        print 'range: %d, pos: %d, neg: %d, dev: %d, test: %d' %(i,len(posTrainList[i]),len(negTrainList[i]),len(devList[i]),len(testList[i]))
       #	trainList[i]=posTrainList[i]+negTrainList[i][:len(posTrainList[i])] # balance
        trainList[i]=posTrainList[i]+negTrainList[i][:len(posTrainList[i])]

    for i in range(week_num):
	for d in trainList[i]+devList[i]+testList[i]:
	    for w in d.words.keys():
		if isEnglishWord(w)==False:
		    del d.words[w] 
   
    return trainList,devList,testList

def getSameDocumetnsList(documentsList):
    idSet=set([d.id for d in documentsList[0]])
    for documents in documentsList:
        idSet0=set([d.id for d in documents])
        idSet=idSet&idSet0
    
    newDocumentsList=[]
    for documents in documentsList:
       # print len(documents)
        idDict=dict([(d.id,d) for d in documents])
        newDocumentsList.append([idDict[id] for id in idSet])

    print 'length of new documents %s' %len(newDocumentsList[0])    

    return newDocumentsList
	
def initData_multi():
    neg=readTweets(r'data/negative.txt')
    pos=readTweets(r'data/positive.txt')
    events=readEventList()
    
    negDates=readRandomDate()
    idList=readIDList()
    
    trainIDs=idList[:80]
    devIDs=idList[80:140]
    testIDs=idList[140:]

    print 'length of pos and neg tweets: %s %s' %(len(pos),len(neg))
    
    posTrainList=getPositiveSamples_ranges(pos,events,trainIDs,4,7)+getPositiveSamples_ranges(pos,events,trainIDs,2,14)
    devList=getPositiveSamples_ranges(pos,events,devIDs,4,7)+getPositiveSamples_ranges(pos,events,devIDs,2,14)
    testList=getPositiveSamples_ranges(pos,events,testIDs,4,7)+getPositiveSamples_ranges(pos,events,testIDs,2,14)

    posTrainList=getSameDocumetnsList(posTrainList)
    devList=getSameDocumetnsList(devList)
    testList=getSameDocumetnsList(testList)
    
    week_num=6
    negTrainList=[[] for i in range(week_num)]
    for i in range(10,20):
        negTrains=getNegativeSamples_ranges(neg,negDates[i+10],events,trainIDs,4,7)+getNegativeSamples_ranges(neg,negDates[i+10],events,trainIDs,2,14)
        negDev=getNegativeSamples_ranges(neg,negDates[i],events,devIDs,4,7)+getNegativeSamples_ranges(neg,negDates[i],events,devIDs,2,14)
        negTests=getNegativeSamples_ranges(neg,negDates[i],events,testIDs,4,7)+getNegativeSamples_ranges(neg,negDates[i],events,testIDs,2,14)

        negTrains=getSameDocumetnsList(negTrains)
        negDev=getSameDocumetnsList(negDev)
        negTests=getSameDocumetnsList(negTests)
        
        for j in range(week_num):
            negTrainList[j]+=negTrains[j]
            devList[j]+=negDev[j]
            testList[j]+=negTests[j]
    
    trainList=[[] for i in range(week_num)]
    for i in range(week_num):
        print 'range: %d, pos: %d, neg: %d, dev: %d, test: %d' %(i,len(posTrainList[i]),len(negTrainList[i]),len(devList[i]),len(testList[i]))
       	trainList[i]=posTrainList[i]+negTrainList[i][:len(posTrainList[i])] # balance

    for i in range(week_num):
	for d in trainList[i]+devList[i]+testList[i]:
	    for w in d.words.keys():
		if isEnglishWord(w)==False:
		    del d.words[w] 
   
    return trainList,devList,testList
