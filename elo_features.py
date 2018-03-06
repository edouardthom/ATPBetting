#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def compute_elo_rankings(data):
    """
    Given the list on matches in chronological order, for each match, computes 
    the elo ranking of the 2 players at the beginning of the match
    
    """
    players=list(pd.Series(list(data.Winner)+list(data.Loser)).value_counts().index)
    elo=pd.Series(np.ones(len(players))*1500,index=players)
    ranking_elo=[(1500,1500)]
    for i in range(1,len(data)):
        w=data.iloc[i-1,:].Winner
        l=data.iloc[i-1,:].Loser
        elow=elo[w]
        elol=elo[l]
        pwin=1 / (1 + 10 ** ((elol - elow) / 400))    
        K_win=32
        K_los=32
        new_elow=elow+K_win*(1-pwin)
        new_elol=elol-K_los*(1-pwin)
        elo[w]=new_elow
        elo[l]=new_elol
        ranking_elo.append((elo[data.iloc[i,:].Winner],elo[data.iloc[i,:].Loser])) 
        if i%1000==0:
            print(i)
    ranking_elo=pd.DataFrame(ranking_elo,columns=["elo_winner","elo_loser"])    
    ranking_elo["proba_elo"]=1 / (1 + 10 ** ((ranking_elo["elo_loser"] - ranking_elo["elo_winner"]) / 400))   
    return ranking_elo


########### Comparison ATP ranking vs. Elo vs. bookmakers for the winner prediction
# Interval for the comparison
#beg=datetime(2015,1,1)
#end=data.Date.iloc[-1]
#indices=data[(data.Date>=beg)&(data.Date<=end)].index
## classical ATP ranking
#test=data[["WRank","LRank"]].iloc[indices,:]
#100*(test.LRank>test.WRank).sum()/len(indices)
## Elo ranking
#test=ranking_elo.iloc[indices,:]
#100*(test.elo_winner>test.elo_loser).sum()/len(indices)
## Bookmakers
#test=data.iloc[indices,:]
#100*(test.PSW<test.PSL).sum()/len(indices)
