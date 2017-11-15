#! /usr/bin/env python
#coding=utf-8
import numpy as np

class Embedding:
    def __init__(self):
        self.d={}
        
        count=0
        for line in open(r'data/model.w2v.txt','rb'):
            line=line.strip()
            if len(line)>0 and count>0:
                p=line.split(' ')
                emb_len=len(p)-1
                word=p[0].strip()
                v=np.empty((emb_len),dtype="float32")
                for i in range(emb_len):
                    v[i]=float(p[i+1])
                self.d[word]=v
            count+=1
    
    def embed_word(self,w):
        if w not in self.d:
            return None
        else:
            return self.d[w]
