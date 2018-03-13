#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime

###################### FEATURES BASED ON THE PAST OF THE PLAYERS ###############

def features_past_generation(features_creation_function,
                             days,
                             feature_names_prefix,
                             data,
                             indices):
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
    matches_outcomes=[]
    for i,match_indice in enumerate(indices):
        match=data.iloc[match_indice,:]
        past_matches=data[(data.Date<match.Date)&(data.Date>=match.Date-datetime.timedelta(days=days))]
        match_features_outcome_1=features_creation_function(1,match,past_matches)
        match_features_outcome_2=features_creation_function(2,match,past_matches)
        matches_outcomes.append(match_features_outcome_1)
        matches_outcomes.append(match_features_outcome_2)
        if i%100==0:
            print(str(i)+"/"+str(len(indices))+" matches treated.")
    train=pd.DataFrame(matches_outcomes)
    train.columns=[feature_names_prefix+str(i) for i in range(len(train.columns))]
    return train


def features_player_creation(outcome,match,past_matches):
    features_player=[]
    ##### Match information extraction (according to the outcome)
    player=match.Winner if outcome==1 else match.Loser
    surface=match.Surface
    ##### General stats
    wins=past_matches[past_matches.Winner==player]    
    losses=past_matches[past_matches.Loser==player]    
    todo=pd.concat([wins,losses],0)
    features_player+=[len(wins),len(losses),len(todo)]
    per_victory=100*len(wins)/len(todo) if len(todo)>0 else np.nan
    features_player.append(per_victory)
    ##### Surface
    past_surface=past_matches[past_matches.Surface==surface]
    wins_surface=past_surface[past_surface.Winner==player]    
    losses_surface=past_surface[past_surface.Loser==player]    
    todo_surface=pd.concat([wins_surface,losses_surface],0)
    features_player+=[len(wins_surface),len(losses_surface),len(todo_surface)]
    per_victory_surface=100*len(wins_surface)/len(todo_surface) if len(todo_surface)>0 else np.nan
    features_player.append(per_victory_surface)
    return features_player

def features_recent_creation(outcome,match,past_matches):
    ##### Match information extraction (according to the outcome)
    player=match.Winner if outcome==1 else match.Loser
    date=match.Date
    ##### Last matches
    wins=past_matches[past_matches.Winner==player]    
    losses=past_matches[past_matches.Loser==player]    
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
    features_recent=[dslm,wlmw,rlpp,nslmp,nswlmp,ilm,iitp]
    return features_recent

def features_duo_creation(outcome,match,past):
    features_duo=[]
    ##### Match information extraction (according to the outcome)
    player1=match.Winner if outcome==1 else match.Loser
    player2=match.Loser if outcome==1 else match.Winner
    ##### General duo features
    # % of the previous matches between these 2 players won by each.
    duo1=past[(past.Winner==player1)&(past.Loser==player2)]    
    duo2=past[(past.Winner==player2)&(past.Loser==player1)]    
    duo=pd.concat([duo1,duo2],0)
    features_duo+=[len(duo),len(duo1),len(duo2)]
    per_victory_player1=100*len(duo1)/len(duo) if len(duo)>0 else np.nan
    features_duo.append(per_victory_player1)
    return features_duo

def features_general_creation(outcome,match,past_matches):
    features_general=[]
    ##### Match information extraction (according to the outcome)
    player1=match.Winner if outcome==1 else match.Loser
    rank_player_1=match.WRank if outcome==1 else match.LRank
    rank_player_2=match.LRank if outcome==1 else match.WRank
    
    features_general+=[rank_player_1,rank_player_2,
                       rank_player_2-rank_player_1,
                       int(rank_player_1>rank_player_2)]
    best_ranking_as_winner=past_matches[(past_matches.Winner==player1)].WRank.min()
    best_ranking_as_loser=past_matches[(past_matches.Loser==player1)].LRank.min()
    best_ranking=min(best_ranking_as_winner,best_ranking_as_loser)
    features_general.append(best_ranking)
    return features_general