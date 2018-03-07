#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### TEST souuldn't be here!!!

from datetime import datetime,timedelta

from past_features import *
from elo_features import *
from categorical_features import *
from stategy_assessment import *

data=pd.read_csv("atp_data.csv")
data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

############################ Encoding of categorical features ######################

cat_features=data[["Series","Court","Surface","Round","Best of"]]
cat_features=categorical_features_encoding(cat_features)
nb_players=120
nb_tournaments=20
tournaments_encoded=tournaments_features_encoding(nb_tournaments,data)
players_encoded=players_features_encoding(nb_players,data)
cat_features=pd.concat([cat_features,tournaments_encoded,players_encoded],1)

## Elo rankings data
ranking_elo=compute_elo_rankings(data)



############################ The period that interests us ############################

beg=datetime(2008,1,1) #prior to 2003, Date is the starting date of the tournament
end=data.Date.iloc[-1]
indices=data[(data.Date>beg)&(data.Date<=end)].index

################### Building of some features based on the past ################

player_features=build_dataset(playerFeatures,5,"playerft5",data,indices)
duo_features=build_dataset(duoFeatures,150,"duoft",data,indices)
general_features=build_dataset(generalFeatures,150,"generalft",data,indices)
recent_features=build_dataset(playerRecentMatchesFeatures,150,"recentft",data,indices)
dump(player_features,"player_features")
dump(duo_features,"duo_features")
dump(general_features,"general_features")
dump(recent_features,"recent_features")
#playerft5=load("playerft5")
#duoft=load("duoft")
#generalft=load("generalft")
#recentft=load("recentft")

############################ Selection of our period ############################

data=data.iloc[indices,:].reset_index(drop=True)
cotes_features=data[["PSW","PSL"]]
cat_features=cat_features.iloc[indices,:].reset_index(drop=True)
elo_features=ranking_elo.iloc[indices,:].reset_index(drop=True)

############################ Duplication of rows ############################

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

# Building of the full dataset
xtrain=pd.concat([cotes_features,
                  elo_features,
                  cat_features,
                  player_features,duo_features,general_features,recent_features],1)


######################## BEST POSSIBLE CONFIDENCE DATA ########################

start_date=datetime(2013,1,1) #first day of test set
conf=pd.concat([data[["Date","PSW","PSL"]]],1)
conf["win0"]=[1]*len(conf)
conf["confidence0"]=conf.PSW
perfect_conf=conf[conf.Date>start_date]
perfect_conf=perfect_conf.rename(columns={"Date":"date"})
plotProfits(perfect_conf)


######################## PROFIT COMPUTING ########################

start_date=datetime(2014,1,1) #first day of test set
start_match=data[data.Date==start_date].index[0]
span_matches=len(data)-start_match+1
duration_val_matches=300
xgb_params=params[0]

delta=2000

key_matches=np.array([start_match+delta*i for i in range(int(span_matches/delta)+1)])
nb_players=50
nb_tournaments=5
confs=[]
for test_beginning_match in key_matches:
    conf=vibratingAssessStrategyGlobal(test_beginning_match,10400,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments,xtrain,data)
    confs.append(conf)
confs=[el for el in confs if type(el)!=int]
conf=pd.concat(confs,0)


plotProfits(conf,"Strategy by Edo - tested on the period Jan. 2013 -> Feb 2018 (13,476 matches)")







#### Hyperparameter tuning

## To be tuned
duration_train_matches=[365*2,365*3,365*4,365*5]
max_depth=[15,17,19]
learning_rate=[0.255,0.275,0.295]
nb_players=[10,20,50,80,100,120]
nb_tournaments=[5,12]
hyperparams=np.array(np.meshgrid(duration_train_matches,max_depth,learning_rate,nb_players,nb_tournaments)).T.reshape(-1,5).astype(np.float)

# Not tuned
start_date=datetime(2013,1,1) #first day of test set
start_match=data[data.Date==start_date].index[0]
span_matches=len(data)-start_match+1
duration_val_matches=300
delta=2000

profits30=[]
profits40=[]
profits50=[]

for p in hyperparams:
    sd=start_date
    key_matches=[start_match+delta*i for i in range(int(span_matches/delta)+1)]
    duration_train_matches=int(p[0])
    nb_players=int(p[3])
    nb_tournaments=int(p[4])
    
    learning_rate=[p[2]] 
    max_depth=[int(p[1])]
    min_child_weight=[1]
    gamma=[0.8]
    csbt=[0.5]
    lambd=[0]
    alpha=[2]
    num_rounds=[300]
    early_stop=[6]
    params=np.array(np.meshgrid(learning_rate,max_depth,min_child_weight,gamma,csbt,lambd,alpha,num_rounds,early_stop)).T.reshape(-1,9).astype(np.float)
    xgb_params=params[0]
    
    confs=[]
    for km in key_matches:
        conf=vibratingAssessStrategyGlobal(km,duration_train_matches,duration_val_matches,delta,xgb_params,nb_players,nb_tournaments)
        confs.append(conf)
    confs=[el for el in confs if type(el)!=int]
    conf=pd.concat(confs,0)
    p30=profitComputation(30,conf)
    p40=profitComputation(40,conf)
    p50=profitComputation(50,conf)
    profits30.append(p30)
    profits40.append(p40)
    profits50.append(p50)
dump(profits30,"profits30")
dump(profits40,"profits40")
dump(profits50,"profits50")


hp=pd.DataFrame(hyperparams).iloc[:len(profits30),:]
hp.columns=["train","md","lr","players","tournaments"]
results=pd.concat([hp,pd.Series(profits30,name="profits30")],1)
results.sort_values("profits30",ascending=False)
dump(results,"tuning_delta300")







############# See along time the profit
### The perfect conf
perfect_conf=perfect_conf.sort_values("confidence0",ascending=False)
confconf=perfect_conf.iloc[:int(0.3*len(conf)),:]

### Our conf
dates=data.Date.reset_index()
dates.columns=["match","date"]
conf=conf.merge(dates,on="match")
conf=conf.sort_values("confidence0",ascending=False)
confconf=conf.iloc[:int(0.5*len(conf)),:]
profitComputation(100,confconf)


start_date=datetime(2013,1,1) #it's a monday
start_match=data[data.Date==start_date].index[0]
span_matches=len(data)-start_match+1
matches_delta=100
N=int(span_matches/matches_delta)+1
milestones=np.array([start_match+matches_delta*i for i in range(N)])
profits=[]
lens=[]
for i in range(N-1):
    beg=milestones[i]
    end=milestones[i+1]-1
    conf_sel=confconf[(confconf.match>=beg)&(confconf.match<=end)]
    l=len(conf_sel)
    lens.append(l)
    if l==0:
        profits.append(0)
    else:    
        p=profitComputation(100,conf_sel)
        profits.append(p)
profits=np.array(profits)


fig=plt.figure(figsize=(5.5,3))
ax = fig.add_axes([0,0,1,1])  
ax.plot(profits,linewidth=2,marker="o")

plt.plot(lens)


stds=[]
for nm in range(5,600):
    start_date=datetime(2013,1,1) #it's a monday
    start_match=data[data.Date==start_date].index[0]
    span_matches=len(data)-start_match+1
    matches_delta=nm
    N=int(span_matches/matches_delta)+1
    milestones=np.array([start_match+matches_delta*i for i in range(N)])
    profits=[]
    lens=[]
    for i in range(N-1):
        beg=milestones[i]
        end=milestones[i+1]-1
        conf_sel=confconf[(confconf.match>=beg)&(confconf.match<=end)]
        l=len(conf_sel)
        lens.append(l)
        if l==0:
            profits.append(0)
        else:    
            p=profitComputation(100,conf_sel)
            profits.append(p)
    profits=np.array(profits)
    s=np.nanstd(profits)
    stds.append(s)

