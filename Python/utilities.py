#!/usr/bin/env python3
# -*- coding: utf-8 -*-

    
    
############################### STORAGE ############################
## Some useful functions to store and load data

import pickle
def dump(obj,name):
	pickle.dump(obj,open(name+'.p',"wb")) 
def load(name):
	obj=pickle.load( open( name+".p", "rb" ) ) 
	return obj