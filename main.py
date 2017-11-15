#! /usr/bin/env python
#coding=utf-8
from __future__ import division
from document import *
from embedding import Embedding
from cnn_rnn import cnn_prediction, lstmMulti, lstmMulti2, lstm_prediction
from svmclassify import svm_classify

# ------------------------- Init Data -------------------------
embModel=Embedding()
#trainList,devList,testList=initData(4,7)

trainList,devList,testList=initData_multi()

week_num=len(trainList)
print 'week num: %s' %week_num

documents=[]
for i in range(week_num):
    documents+=trainList[i]+devList[i]+testList[i]
DF(documents)

V=getVocabrary(documents)
print 'length of V: %s' %len(V)

# ------------------------- SVM ------------------------- 
#svm_classify(trainList[0],devList[0])

# ------------------------- Vanilla Stream Model ------------------------- 

#X_train,y_train=formatK(trainList[0],V)
#X_test,y_test=formatK(devList[0],V)

# CNN
#cnn_prediction(X_train,y_train,X_test,y_test,len(V))

# LSTM
#lstm_prediction(X_train,y_train,X_test,y_test,len(V))

# -------------------------  Hierarchical Models ------------------------- 

n=week_num
   
X_train_list=[]
y_train=[]
X_test_list=[]
y_test=[]

for i in range(n):
    X_train,y_train=formatK(trainList[i],V)
    X_train_list.append(X_train)

    X_test,y_test=formatK(devList[i],V)
    X_test_list.append(X_test)

# Short- and Long-Term Stream Model
lstmMulti(X_train_list,y_train,X_test_list,y_test,len(V),n)

# Hierarchical Stream Model
#lstmMulti2(X_train_list,y_train,X_test_list,y_test,len(V),n)


