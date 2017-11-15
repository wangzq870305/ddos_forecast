#! /usr/bin/env python
#coding=utf-8

class Tweet:
    def __init__(self,id,date,text):
        self.id=id
        self.date=date
        self.text=text
        
class Event:
    def __init__(self,id,date,name):
        self.id=id
        self.date=date
        self.name=name