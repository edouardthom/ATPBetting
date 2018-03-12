#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

############################### CATEGORICAL FEATURES ENCODING ##################

### The features "player1", "player2" and "Tournament" are treated differently
### from the other features. 

def categorical_features_encoding(cat_features):
    """
    Categorical features encoding.
    Simple one-hot encoding.
    """
    cat_features=cat_features.apply(preprocessing.LabelEncoder().fit_transform)
    ohe=OneHotEncoder()
    cat_features=ohe.fit_transform(cat_features)
    cat_features=pd.DataFrame(cat_features.todense())
    cat_features.columns=["cat_feature_"+str(i) for i in range(len(cat_features.columns))]
    cat_features=cat_features.astype(int)
    return cat_features

def players_features_encoding(nb_players,data):
    """
    Encoding of the players . 
    We limit the number of players : we'll keep only the 
    players that won the most matches to avoid overfitting (usually 80 players).
    The other players are encoded in the column "famous_player_other".
    The players are not encoded like the other categorical features because for each
    match we encode both players at the same time (we put a 1 in each row corresponding 
    to the players playing the match for each match).
    """
    players=list(data.Winner.value_counts().index[:nb_players])
    winners=data.Winner.apply(lambda x:x if x in players else "other")
    losers=data.Loser.apply(lambda x:x if x in players else "other")
    le = preprocessing.LabelEncoder()
    winners=le.fit_transform(winners)
    losers=le.transform(losers)
    encod=np.zeros([len(winners),nb_players+1])
    for i in range(len(winners)):
        encod[i,winners[i]]+=1
    for i in range(len(losers)):
        encod[i,losers[i]]+=1
    players=pd.Series(range(len(players)),index=players)
    ## The number in the column name indicates the ranking of the player (in number of matches won)
    columns=["famous_player_"+str(players[le.classes_[i]]) if le.classes_[i]!="other" else "famous_player_other" for i in range(nb_players+1)]
    players_encoded=pd.DataFrame(encod,columns=columns)
    return players_encoded

def tournaments_features_encoding(nb_tournaments,data):
    """
    Encoding of the tournaments . We limit the number of tournaments to avoid overfitting.
    (see players_features_encoding)
    """
    tournaments=list(data.Tournament.value_counts().index[:nb_tournaments])
    tournament=data.Tournament.apply(lambda x:x if x in tournaments else "other")
    le = preprocessing.LabelEncoder()
    tournament=le.fit_transform(tournament)
    encod=np.zeros([len(tournament),nb_tournaments+1])
    for i in range(len(tournament)):
        encod[i,tournament[i]]+=1
    ## The number in the column name indicates the ranking of the player (in number of matches won)
    tournaments=pd.Series(range(len(tournaments)),index=tournaments)
    columns=["famous_tournament_"+str(tournaments[le.classes_[i]]) if le.classes_[i]!="other" else "famous_tournament_other" for i in range(nb_tournaments+1)]
    tournaments_encoded=pd.DataFrame(encod,columns=columns)
    return tournaments_encoded
