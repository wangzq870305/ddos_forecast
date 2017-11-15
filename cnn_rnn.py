from __future__ import print_function
from __future__ import division
from functools import reduce
import re
import tarfile
import math

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten
from keras.layers import recurrent, Merge
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D

from sklearn.metrics import average_precision_score

from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence

RNN = recurrent.LSTM

EMBED_SIZE = 32
HIDDEN_SIZE= 32
MAX_LEN=50 # 500
BATCH_SIZE = 16
EPOCHS = 5 

nb_filter = 10
filter_length = 5

def readResult(y_test,results):
    index=0
    p=n=tp=tn=fp=fn=0
    for prob in results:
        if prob>0.5:
            predLabel=1
        else:
            predLabel=0
        if y_test[index]>0:
            p+=1
            if predLabel>0:
                tp+=1
            else:
                fn+=1
        else:
            n+=1
            if predLabel==0:
                tn+=1
            else:
                fp+=1
        index+=1

    acc=(tp+tn)/(p+n)
    precisionP=tp/(tp+fp)
    precisionN=tn/(tn+fn)
    recallP=tp/(tp+fn)
    recallN=tn/(tn+fp)
    gmean=math.sqrt(recallP*recallN)
    f_p=2*precisionP*recallP/(precisionP+recallP)
    f_n=2*precisionN*recallN/(precisionN+recallN)
    print ('{gmean:%s recallP:%s recallN:%s} {precP:%s precN:%s fP:%s fN:%s} acc:%s' %(gmean,recallP,recallN,precisionP,precisionN,f_p,f_n,acc))
    print('AUC %s' %average_precision_score(y_test,results))

    output=open('result.output','w')
    output.write('\n'.join(['%s' %r for r in results]))
	
def cnn_prediction(X_train,y_train,X_test,y_test,vocab_size): 
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape) 
    
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN, dropout=0.2))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
     
    model.add(MaxPooling1D(pool_length=2))
    
    model.add(Flatten())
    
    model.add(Dense(HIDDEN_SIZE))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              nb_epoch=EPOCHS,
              validation_data=(X_test, y_test))
            
    X_pred=model.predict(X_test)
    results=[result[0] for result in X_pred]
    
    return readResult(y_test,results)
	
def lstm_prediction(X_train,y_train,X_test,y_test,vocab_size): 
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape) 
    
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN, dropout=0.2))
    
    
    model.add(RNN(HIDDEN_SIZE))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              nb_epoch=EPOCHS, 
              validation_data=(X_test, y_test))
            
    X_pred=model.predict(X_test)
    results=[result[0] for result in X_pred]
    
    return readResult(y_test,results)
	
def buildModel_max(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN, dropout=0.2))
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=2))
      
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())
    model.add(RepeatVector(HIDDEN_SIZE))    
    return model  
	
def lstmMulti(X_train_list,y_train,X_test_list,y_test,vocab_size,n):
    X_new_train_list=[]
    X_new_test_list=[]
    for i in range(n):
        X_train = sequence.pad_sequences(X_train_list[i], maxlen=MAX_LEN)
        X_test = sequence.pad_sequences(X_test_list[i], maxlen=MAX_LEN)
        X_new_train_list.append(X_train)
        X_new_test_list.append(X_test)    

    print('Build model...')

    firstLayers=[buildModel_max(vocab_size) for i in range(n)]
    
    model = Sequential()
    model.add(Merge(firstLayers, mode='concat'))
    model.add(RNN(HIDDEN_SIZE))

#    model.add(RepeatVector(HIDDEN_SIZE))
#    model.add(RNN(HIDDEN_SIZE))

#    model.add(RepeatVector(HIDDEN_SIZE))
#    model.add(RNN(HIDDEN_SIZE))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    print('Training')

    
    model.fit(X_new_train_list, y_train, batch_size=BATCH_SIZE,
                  nb_epoch=EPOCHS)

    X_pred=model.predict(X_new_test_list)
    results=[result[0] for result in X_pred]
    
    return readResult(y_test,results)

def lstmMulti2(X_train_list,y_train,X_test_list,y_test,vocab_size,n=4):
    X_new_train_list=[]
    X_new_test_list=[]
    for i in range(n):
        X_train = sequence.pad_sequences(X_train_list[i], maxlen=MAX_LEN)
        X_test = sequence.pad_sequences(X_test_list[i], maxlen=MAX_LEN)
        X_new_train_list.append(X_train)
        X_new_test_list.append(X_test)    

    print('Build model...')

    # Layer 0
    firstLayers=[buildModel_max(vocab_size) for i in range(n)]
    
    # Layer 1
    model0 = Sequential()
    model0.add(Merge(firstLayers[:2], mode='sum'))
    model0.add(RNN(HIDDEN_SIZE))
    model0.add(RepeatVector(vocab_size))
    model1 = Sequential()
    model1.add(Merge(firstLayers[2:], mode='sum'))
    model1.add(RNN(HIDDEN_SIZE))
    model1.add(RepeatVector(vocab_size))
    
    # Layer 2
    model = Sequential() 
    model.add(Merge([model0,model1], mode='sum'))
    model.add(RNN(HIDDEN_SIZE))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    print('Training')

    
    model.fit(X_new_train_list, y_train, batch_size=BATCH_SIZE,
                  nb_epoch=EPOCHS)

    X_pred=model.predict(X_new_test_list)
    results=[result[0] for result in X_pred]
    
    return readResult(y_test,results)
