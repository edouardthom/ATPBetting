#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime,timedelta

from past_features import *
from elo_features import *
from categorical_features import *
from stategy_assessment import *
from utilities import *



################################################################################
######################### Building of the raw dataset ##########################
################################################################################

### Importation of the Excel files - 1 per year (from tennis.co.uk)
# Some preprocessing is necessary because for several years the odds are not present
# We consider only the odds of Bet365 and Pinnacle.

import glob
filenames=list(glob.glob("../Data/20*.xls*"))
l = [pd.read_excel(filename,encoding='latin-1') for filename in filenames]
no_b365=[i for i,d in enumerate(l) if "B365W" not in l[i].columns]
no_pi=[i for i,d in enumerate(l) if "PSW" not in l[i].columns]
for i in no_pi:
    l[i]["PSW"]=np.nan
    l[i]["PSL"]=np.nan
for i in no_b365:
    l[i]["B365W"]=np.nan
    l[i]["B365L"]=np.nan
l=[d[list(d.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]] for d in [l[0]]+l[2:]]
data=pd.concat(l,0)

### Data cleaning
data=data.sort_values("Date")
data["WRank"]=data["WRank"].replace(np.nan,0)
data["WRank"]=data["WRank"].replace("NR",2000)
data["LRank"]=data["LRank"].replace(np.nan,0)
data["LRank"]=data["LRank"].replace("NR",2000)
data["WRank"]=data["WRank"].astype(int)
data["LRank"]=data["LRank"].astype(int)
data["Wsets"]=data["Wsets"].astype(float)
data["Lsets"]=data["Lsets"].replace("`1",1)
data["Lsets"]=data["Lsets"].astype(float)
data=data.reset_index(drop=True)

### Elo rankings data
ranking_elo=compute_elo_rankings(data)
data=pd.concat([data,ranking_elo],1)

### Storage of the raw dataset
data.to_csv("../Generated Data/atp_data.csv",index=False)





################################################################################
####################### Study of simple winner prediction ######################
################################################################################

# Comparison of different techniques for the winner prediction
# Interval for the comparison : [beg,end]
beg=datetime(2013,1,1)
end=data.Date.iloc[-1]
indices=data[(data.Date>=beg)&(data.Date<=end)].index
# classical ATP ranking
test=data[["WRank","LRank"]].iloc[indices,:]
atp=100*(test.LRank>test.WRank).sum()/len(indices)
# Elo ranking
test=ranking_elo.iloc[indices,:]
elo=100*(test.elo_winner>test.elo_loser).sum()/len(indices)
# Bookmakers
test=data.iloc[indices,:]
book_pi=100*(test.PSW<test.PSL).sum()/len(indices)
book_365=100*(test.B365W<test.B365L).sum()/len(indices)
# Our prediction
conf=pd.read_csv("../Generated Data/confidence_data.csv")
our=100*conf.win0.sum()/len(conf)
# Plot
labels=["Prediction with ATP ranking","with Elo ranking",
        "with smallest odd (Bet365)","with smallest odd (Pinnacle)",
        "our prediction"]
values=[atp,elo,book_pi,book_365,our]
xaxis_label="% of matches correctly predicted"
title="Prediction of all matches since Jan. 2013 ("+str(len(indices))+" matches)"
xlim=[65,70]
basic_horizontal_barplot(values,labels,xaxis_label,title,xlim,figsize=None)







################################################################################
###################### Building of the enriched dataset ########################
################################################################################
### We'll add some features to the dataset

data=pd.read_csv("../Generated Data/atp_data.csv")
data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

########################## Encoding of categorical features ####################

cat_features=data[["Series","Court","Surface","Round","Best of"]]
cat_features=categorical_features_encoding(cat_features)
nb_players=120
nb_tournaments=20
tournaments_encoded=tournaments_features_encoding(nb_tournaments,data)
players_encoded=players_features_encoding(nb_players,data)
cat_features=pd.concat([cat_features,tournaments_encoded,players_encoded],1)

######################### The period that interests us #########################
### We'll perform the training and testing during this period. But the matches before 
### are used, for example to compute the elo rankings.
beg=datetime(2008,1,1) 
end=data.Date.iloc[-1]
indices=data[(data.Date>beg)&(data.Date<=end)].index

################### Building of some features based on the past ################

#player_features=build_dataset(playerFeatures,5,"playerft5",data,indices)
#duo_features=build_dataset(duoFeatures,150,"duoft",data,indices)
#general_features=build_dataset(generalFeatures,150,"generalft",data,indices)
#recent_features=build_dataset(playerRecentMatchesFeatures,150,"recentft",data,indices)
#dump(player_features,"player_features")
#dump(duo_features,"duo_features")
#dump(general_features,"general_features")
#dump(recent_features,"recent_features")
player_features=load("player_features")
duo_features=load("duo_features")
general_features=load("general_features")
recent_features=load("recent_features")

########################### Selection of our period ############################

data=data.iloc[indices,:].reset_index(drop=True)
cotes_features=data[["PSW","PSL"]]
cat_features=cat_features.iloc[indices,:].reset_index(drop=True)
elo_features=ranking_elo.iloc[indices,:].reset_index(drop=True)

############################### Duplication of rows ############################
## For the moment we have one row per match. 
## We "duplicate" each row to have one row for each outcome of the match. 
## Of course it isn't a simple duplication of  each row, we need to "invert" some features

# Elo data
elo_1=elo_features
elo_2=elo_1[["elo_loser","elo_winner","proba_elo"]]
elo_2.columns=["elo_winner","elo_loser","proba_elo"]
elo_2.proba_elo=1-elo_2.proba_elo
elo_2.index=range(1,2*len(elo_1),2)
elo_1.index=range(0,2*len(elo_1),2)
elo_features=pd.concat([elo_1,elo_2]).sort_index(kind='merge')

# Categorical features
cat_features=pd.DataFrame(np.repeat(cat_features.values,2, axis=0),columns=cat_features.columns)

# cotes features
cotes_features=pd.Series(cotes_features.values.flatten(),name="odds")
cotes_features=pd.DataFrame(cotes_features)

### Building of the final dataset
# You can remove some features to see the effect on the ROI
xtrain=pd.concat([cotes_features,
                  #elo_features,
                  cat_features],1)
                  #player_features,duo_features,general_features,recent_features],1)

xtrain.to_csv("../Generated Data/atp_data_features.csv",index=False)





################################################################################
#################### Strategy assessment - ROI computing #######################
################################################################################

## We adopt a rolling method. We predict the outcome of delta consecutive matches , 
## with the N previous matches. A small subset of the training set is devoted to
## validation (the consecutive matches right before the testing matches)

######################### Confidence computing for each match ############################

start_date=datetime(2013,1,1) #first day of test set
start_match=data[data.Date==start_date].index[0] #id of the first match of the testing set
span_matches=len(data)-start_match+1
duration_val_matches=300
delta=2000

## Number of tournaments and players encoded directly in one-hot (the most important only)
nb_players=50
nb_tournaments=5

## XGB parameters
learning_rate=[0.295] 
max_depth=[19]
min_child_weight=[1]
gamma=[0.8]
csbt=[0.5]
lambd=[0]
alpha=[2]
num_rounds=[300]
early_stop=[5]
params=np.array(np.meshgrid(learning_rate,max_depth,min_child_weight,gamma,csbt,lambd,alpha,num_rounds,early_stop)).T.reshape(-1,9).astype(np.float)
xgb_params=params[0]

## We predict the confidence in each outcome, delta matches at each iteration
key_matches=np.array([start_match+delta*i for i in range(int(span_matches/delta)+1)])
confs=[]
for test_beginning_match in key_matches:
    conf=vibratingAssessStrategyGlobal(test_beginning_match,10400,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data)
    confs.append(conf)
confs=[el for el in confs if type(el)!=int]
conf=pd.concat(confs,0)
## We add the date to the confidence dataset (can be useful for analysis later)
dates=data.Date.reset_index()
dates.columns=["match","date"]
conf=conf.merge(dates,on="match")
conf=conf.sort_values("confidence0",ascending=False)
conf=conf.reset_index(drop=True)


## We store this dataset
conf.to_csv("../Generated Data/confidence_data.csv",index=False)

## Plot of ROI according to the % of matches we bet on
plotProfits(conf,"Test on the period Jan. 2013 -> March 2018")




################################################################################
######################### ROI variability along time ###########################
################################################################################

## We bet only on 35% of the matches
confconf=conf.iloc[:int(0.35*len(conf)),:]

def profitsAlongTime(conf,matches_delta):
    span_matches=span_matches=conf.match.max()-conf.match.min()-1
    N=int(span_matches/matches_delta)+1
    milestones=np.array([conf.match.min()+matches_delta*i for i in range(N)])
    profits=[]
    lens=[]
    for i in range(N-1):
        beg=milestones[i]
        end=milestones[i+1]-1
        conf_sel=confconf[(conf.match>=beg)&(conf.match<=end)]
        l=len(conf_sel)
        lens.append(l)
        if l==0:
            profits.append(0)
        else:    
            p=profitComputation(100,conf_sel)
            profits.append(p)
    profits=np.array(profits)
    return profits,lens

matches_delta=117
profits,lens=profitsAlongTime(confconf,matches_delta)

fig=plt.figure(figsize=(5.5,3))
ax = fig.add_axes([0,0,1,0.9])  
ax.plot(profits,linewidth=2,marker="o")
plt.suptitle("Betting on sections of 100 matches")
ax.set_xlabel("From 2013 to 2018")
ax.set_ylabel("ROI")

fig=plt.figure(figsize=(5.5,3))
ax = fig.add_axes([0,0,1,0.9])  
ax.plot(lens,linewidth=2,marker="o")
plt.suptitle("Betting on sections of 100 matches")
ax.set_xlabel("From 2013 to 2018")
ax.set_ylabel("For each section, number of matches we bet on")
