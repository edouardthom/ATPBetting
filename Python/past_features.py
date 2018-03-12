#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import timedelta

###################### FEATURES BASED ON THE PAST OF THE PLAYERS ###############

def playerFeatures(sense,row,past):
    features=[]
    ##### Extraction
    player=row.Winner if sense==1 else row.Loser
    surface=row.Surface
    ##### General stats
    wins=past[past.Winner==player]    
    losses=past[past.Loser==player]    
    todo=pd.concat([wins,losses],0)
    features+=[len(wins),len(losses),len(todo)]
    per_victory=100*len(wins)/len(todo) if len(todo)>0 else np.nan
    features.append(per_victory)
    ##### Surface
    past_surface=past[past.Surface==surface]
    wins_surface=past_surface[past_surface.Winner==player]    
    losses_surface=past_surface[past_surface.Loser==player]    
    todo_surface=pd.concat([wins_surface,losses_surface],0)
    features+=[len(wins_surface),len(losses_surface),len(todo_surface)]
    per_victory_surface=100*len(wins_surface)/len(todo_surface) if len(todo_surface)>0 else np.nan
    features.append(per_victory_surface)
    return features

def playerRecentMatchesFeatures(sense,row,past):
    features=[]
    ##### Extraction
    player=row.Winner if sense==1 else row.Loser
    date=row.Date
    ##### Last matches
    wins=past[past.Winner==player]    
    losses=past[past.Loser==player]    
    todo=pd.concat([wins,losses],0)
    if len(todo)==0:
        return [np.nan]*7
    # Days since last match
    dslm=(date-todo.iloc[-1,:].Date).days
    # Was the last match won ?
    wlmw=int(todo.iloc[-1,:].Winner==player)
    # Ranking of the last player played
    rlpp=todo.iloc[-1,:].WRank
    # Number of sets of last match played
    nslmp=todo.iloc[-1,:]['Best of']
    # Number of sets won during last match played
    nswlmp=todo.iloc[-1,:]['Wsets'] if wlmw==1 else todo.iloc[-1,:]['Lsets']
    # Injuries - iitp + injury last match
    if len(losses)!=0:
        ilm=int(losses.iloc[-1,:].Comment=="Completed")
        iitp=1 if (losses.Comment!="Completed").sum()>0 else 0
    else:
        ilm=np.nan
        iitp=np.nan
    features+=[dslm,wlmw,rlpp,nslmp,nswlmp,ilm,iitp]
    return features

def duoFeatures(sense,row,past):
    duo_features=[]
    ##### Extraction
    player1=row.Winner if sense==1 else row.Loser
    player2=row.Loser if sense==1 else row.Winner
    ##### General duo features
    # % of the previous matches between these 2 players won by each.
    duo1=past[(past.Winner==player1)&(past.Loser==player2)]    
    duo2=past[(past.Winner==player2)&(past.Loser==player1)]    
    duo=pd.concat([duo1,duo2],0)
    duo_features+=[len(duo),len(duo1),len(duo2)]
    per_victory_player1=100*len(duo1)/len(duo) if len(duo)>0 else np.nan
    duo_features.append(per_victory_player1)
    return duo_features

def generalFeatures(sense,row,past):
    general_features=[]
    ##### Extraction
    player1=row.Winner if sense==1 else row.Loser
    r1=row.WRank if sense==1 else row.LRank
    r2=row.LRank if sense==1 else row.WRank
    general_features+=[r1,r2,r2-r1,int(r2>r1)]
    # Best ranking in the last x days
    rankw=past[(past.Winner==player1)].WRank.min()
    rankl=past[(past.Loser==player1)].LRank.min()
    best_rank=min(rankw,rankl)
    general_features.append(best_rank)
    return general_features

def getFeatures(data,nb_row,features_creation,days):   
    row=data.iloc[nb_row,:]
    sel=(data.Date<row.Date)&(data.Date>=row.Date-timedelta(days=days))
    past=data[sel]
    sense1=features_creation(1,row,past)
    sense2=features_creation(2,row,past)
    return sense1,sense2

def build_dataset(features_creation,days,feature_names_prefix,data,indices):
    """
    Creates features based on the past of the players. 
    Basically a for loop. Takes 1 match at a time, selects the matches that occurred during 
    its close past (usually 150 days before max) and computes some features.
    Each match will appear twice in the dataset : 1 time per outcome of the match.
    Example : 02/03/2016 Djoko-Zverev ,Djoko won
        During the 150 days before the match, Djoko won 80% of its matches and Zverev 40%.
        We encode the outcome "Djoko wins" like that : [80,40], and tell the model this outcome happened (1).
        We encode the outcome "Zverev wins" like that : [40,80], and tell the model it didn't happen (0).
    And we do that with some more features , based on the players past stats on the surface
    of the match, on the recent injuries, ...
    In the inputs of the function, "indices" contains the indices of the matches we want to encode.
    The output of the functions is twice as long as "indices".
    (these features introduce many hyperparameters to be tuned...)
    """
    train_examples=[]
    for count,i in enumerate(indices):
        sense1,sense2=getFeatures(data,i,features_creation,days)
        train_examples.append(sense1)
        train_examples.append(sense2)
        if count%100==0:
            print(str(count)+"/"+str(len(indices))+" matches treated.")
    train=pd.DataFrame(train_examples)
    train.columns=[feature_names_prefix+str(i) for i in range(len(train.columns))]
    return train
