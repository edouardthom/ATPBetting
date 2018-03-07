#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

############################### STRATEGY ASSESSMENT ############################

def xgbModelBinary(xtrain,ytrain,xtest,ytest,p,sample_weights=None):
	#Here xtest and ytest are also used for early stopping (validation set)
	if sample_weights==None:
		dtrain=xgb.DMatrix(xtrain,label=ytrain)
	else:
		dtrain=xgb.DMatrix(xtrain,label=ytrain,weight=sample_weights)
	dtest=xgb.DMatrix(xtest,label=ytest)
	eval_set = [(dtrain,"train_loss"),(dtest, 'eval')]
	params={'eval_metric':"logloss","objective":"binary:logistic",'subsample':0.8,
		 'min_child_weight':p[2],'alpha':p[6],'lambda':p[5],'max_depth':int(p[1]),
		 'gamma':p[3],'eta':p[0],'colsample_bytree':p[4]}
	model=xgb.train(params, dtrain, int(p[7]),evals=eval_set,early_stopping_rounds=int(p[8]))
	prediction= model.predict(dtest)
	return prediction,model


def sel_match_confidence(x):
    """
    One possible betting strategy
    """
    if x[0]>x[1]: # ie. if we bet on player1
        return x[0]/x[2] 
    else:
        return x[1]/x[3] 

def assessStrategyGlobal(test_beginning_match,duration_train_matches,duration_val_matches,duration_test_matches,xgb_params,nb_players,nb_tournaments,xtrain,data,model_name="0"):
    """
    Given the ids of the first match of the testing set (id=index in the dataframe "data"),
    outputs the confidence dataframe.
    The confidence dataframe tells for each match is our prediction is right, and for
    the outcome we chose, the confidence level.
    the confidence level is simply the probability we predicted divided by the probability
    implied by the bookmaker (=1/odd).
    """
    ########## Training/validation/testing set generation
    
    # Number of matches we in our training set 
    nm=int(len(xtrain)/2)
    
    beg_test=test_beginning_match
    end_test=min(test_beginning_match+duration_test_matches-1,nm-1)
    end_val=min(beg_test-1,nm-1)
    beg_val=beg_test-duration_val_matches
    end_train=beg_val-1
    beg_train=beg_val-duration_train_matches
       
    train_indices=range(2*beg_train,2*end_train+2)
    val_indices=range(2*beg_val,2*end_val+2)
    test_indices=range(2*beg_test,2*end_test+2)
    
    # We keep only a limited number of players/tournaments
    tournament_colums=[c for c in xtrain.columns if (c[:18]=="famous_tournament_")&(c[18:]!="other")]
    to_drop=[c for c in tournament_colums if int(c[18:])>=nb_tournaments]
    xtrain=xtrain.drop(to_drop,1)
    player_colums=[c for c in xtrain.columns if (c[:14]=="famous_player_")&(c[14:]!="other")]
    to_drop=[c for c in player_colums if int(c[14:])>=nb_players]
    xtrain=xtrain.drop(to_drop,1)
    
    # Split in train/validation/test
    xval=xtrain.iloc[val_indices,:].reset_index(drop=True)
    xtest=xtrain.iloc[test_indices,:].reset_index(drop=True)
    xtrain=xtrain.iloc[train_indices,:].reset_index(drop=True)
    ytrain=pd.Series([1,0]*int(len(train_indices)/2))
    yval=pd.Series([1,0]*int(len(val_indices)/2))
    
    if (len(test_indices)==0)|(len(train_indices)==0):
        return 0
    
    # ML
    pred_val,model=xgbModelBinary(xtrain,ytrain,xval,yval,xgb_params,sample_weights=None)
    
    # Prediction for the testing set
    dtest=xgb.DMatrix(xtest,label=None)
    pred_test= model.predict(dtest)
    prediction_test=pred_test[range(0,len(pred_test),2)]
    
    ## Strategy : Betting on better odds - TESTING SET
    cotes_full=data[["PSW","PSL"]].values.flatten()
    cotes_full=pd.Series(cotes_full,name="cotes")
    pred_book=(1/cotes_full).iloc[test_indices].reset_index(drop=True).fillna(1)
    pred_book_win=pred_book[list(range(0,len(pred_book),2))].values
    pred_book_loser=pred_book[list(range(1,len(pred_book),2))].values
    pred_p1=pred_test[range(0,len(pred_test),2)]
    pred_p2=pred_test[range(1,len(pred_test),2)]
    p=pd.Series(list(zip(pred_p1,pred_p2,pred_book_win,pred_book_loser)))

    # The model opinion on each match + the confidence
    bet_confidence_matches=p.apply(lambda x:sel_match_confidence(x))
    matches_bet_good=(prediction_test>=0.5).astype(int)
    test_indices_notduplicated=np.array(np.array(test_indices)[range(0,len(test_indices),2)]/2).astype(int)
    confidence=pd.DataFrame({"match":test_indices_notduplicated,"win"+model_name:matches_bet_good,"confidence"+model_name:bet_confidence_matches})
    confidenceTest=confidence.sort_values("confidence"+model_name,ascending=False)
   
    #### For both we join the cote for the winners
    c=data[["PSW"]].reset_index()
    c.columns=["match","PSW"]
    confidenceTest=confidenceTest.merge(c,how="left",on="match")
    
    return confidenceTest

def vibratingAssessStrategyGlobal(km,dur_train,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data):
    """
    The Roi is very sensistive to the training set. One day more or less can change it 
    substantially. Therefore it is preferable to run assessStrategyGlobal several times
    with slights changes in the training set lenght, and then combine the predictions.
    This is what this function does (majority voting for the cobination and average of the
    confidences of the models).
    """
    confTest1=assessStrategyGlobal(km,dur_train,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"1")
    confTest2=assessStrategyGlobal(km,dur_train-10,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"2")
    confTest3=assessStrategyGlobal(km,dur_train+10,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"3")
    confTest4=assessStrategyGlobal(km,dur_train-30,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"4")
    confTest5=assessStrategyGlobal(km,dur_train+30,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"5")
    confTest6=assessStrategyGlobal(km,dur_train-45,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"6")
    confTest7=assessStrategyGlobal(km,dur_train+45,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data,"7")
    if (type(confTest1)!=int)&(type(confTest2)!=int)&(type(confTest3)!=int)&(type(confTest4)!=int)&(type(confTest5)!=int):
        c=confTest1.merge(confTest2,on=["match","PSW"])
        c=c.merge(confTest3,on=["match","PSW"])
        c=c.merge(confTest4,on=["match","PSW"])
        c=c.merge(confTest5,on=["match","PSW"])
        c=c.merge(confTest6,on=["match","PSW"])
        c=c.merge(confTest7,on=["match","PSW"])
        c=pd.Series(list(zip(c.win1,c.win2,c.win3,c.win4,c.win5,
                             c.win6,c.win7,
                             c.confidence1,c.confidence2,c.confidence3,
                             c.confidence4,c.confidence5,
                             c.confidence6,c.confidence7)))
        c=pd.DataFrame.from_records(list(c.apply(mer)))
        conf=pd.concat([confTest1[["match","PSW"]],c],1)
        conf.columns=["match","PSW","win0","confidence0"]
    else:
        conf=0
    return conf
def mer(t):
    w=np.array([t[0],t[1],t[2],t[3],t[4],t[5],t[6]]).astype(bool)
    conf=np.array([t[7],t[8],t[9],t[10],t[11],t[12],t[13]])
    if w.sum()>=4:
        return 1,conf[w].mean()
    else:
        return 0,conf[~w].mean()

############################### PROFITS COMPUTING AND VISUALIZATION ############

def profitComputation(percentage_matchs,conf,model_name="0"):
    """
    Given a confidence dataset and a percentage of matches, computes the ROI 
    if we bet only on the percentage of matches we have the most confidence in
    (same amount of money for each match).
    """
    coeff=percentage_matchs/100
    lim=int(coeff*len(conf))
    conf=conf.sort_values("confidence"+model_name,ascending=False)
    conf=conf.iloc[:lim,:]
    profit=100*(conf.PSW[conf["win"+model_name]==1].sum()-len(conf))/len(conf)
    return profit

def plotProfits(conf,title=""):
    """
    Given a confidence dataset, plots the ROI according to the percentage of matches
    we bet on. 
    """
    profits=[]
    ticks=range(5,101)
    for i in ticks:
        p=profitComputation(i,conf)
        profits.append(p)
    plt.plot(ticks,profits)
    plt.xticks(range(0,101,5))
    plt.xlabel("% of matches we bet on")
    plt.ylabel("Return on investment (%)")
    plt.suptitle(title)


############################### STORAGE ############################

import pickle
def dump(obj,name):
	pickle.dump(obj,open(name+'.p',"wb")) 
def load(name):
	obj=pickle.load( open( name+".p", "rb" ) ) 
	return obj