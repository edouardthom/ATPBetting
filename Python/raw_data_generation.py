#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime,timedelta

########################## RAW DATA GENERATION ##########################

## 
data1=pd.read_csv("data.csv",encoding='latin-1')
data1.Date = data1.Date.apply(lambda x:datetime.strptime(x, '%d/%m/%Y'))
data1=data1[data1.Date.apply(lambda x:x.year)<2016]
data2=pd.read_excel("2016.xlsx",encoding='latin-1')
data3=pd.read_excel("2017.xlsx",encoding='latin-1')
data4=pd.read_excel("2018.xlsx",encoding='latin-1')

data1=data1[list(data1.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]]
data2=data2[list(data2.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]]
data3=data3[list(data3.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]]
data4=data4[list(data4.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]]

data=pd.concat([data1,data2,data3,data4],0).reset_index(drop=True)
del data1,data2,data3,data4

## Data cleaning
data=data.sort_values("Date")
data["WRank"]=data["WRank"].replace(np.nan,0)
data["WRank"]=data["WRank"].replace("NR",2000)
data["LRank"]=data["LRank"].replace(np.nan,0)
data["LRank"]=data["LRank"].replace("NR",2000)
data["WRank"]=data["WRank"].astype(int)
data["LRank"]=data["LRank"].astype(int)
data=data.reset_index(drop=True)

data.to_csv("atp_data.csv",index=False)

########################## ANALYSIS ##########################

beg=datetime(2011,1,1) 
end=data.Date.iloc[-1]
indices=data[(data.Date>beg)&(data.Date<=end)].index

data_sel=data[["B365W","B365L","PSW","PSL","WRank","LRank"]]
data_sel=data.iloc[indices,:]

## Comparison of some basic strategies
roi_smallest_odd_ps=100*(data_sel.PSW[data_sel.PSW<data_sel.PSL].sum()-len(data_sel))/len(data_sel)
roi_best_ranking_ps=100*(data_sel.PSW[data_sel.WRank<data_sel.LRank].sum()-len(data_sel))/len(data_sel)
roi_random_ps=100*(data_sel.sample(int(len(data_sel)/2)).PSW.sum()-len(data_sel))/len(data_sel)
roi_smallest_odd_365=100*(data_sel.B365W[data_sel.B365W<data_sel.B365L].sum()-len(data_sel))/len(data_sel)
roi_best_ranking_365=100*(data_sel.B365W[data_sel.WRank<data_sel.LRank].sum()-len(data_sel))/len(data_sel)
roi_random_365=100*(data_sel.sample(int(len(data_sel)/2)).PSW.sum()-len(data_sel))/len(data_sel)
values=[roi_smallest_odd_ps,roi_best_ranking_ps,roi_random_ps,
        roi_smallest_odd_365,roi_best_ranking_365,roi_random_365]
labels=["Pinnacle\nsmallest odds strategy","Pinnacle\nbest ranking strategy","Pinnacle\nhead or tail betting",
       "Bet365\nsmallest odds strategy","Best365\nbest ranking strategy","Bet365\nhead or tail betting"]
xaxis_label="Return on investment (ROI) in %"
title="Betting on all ATP matches since 2011"
basic_horizontal_barplot(values,labels,xaxis_label,title,[0,-8],(4,4))

## Maximum possible ROI
max_roi_ps=100*(data_sel.PSW.sum()-len(data_sel))/len(data_sel)
max_roi_365=100*(data_sel.B365W.sum()-len(data_sel))/len(data_sel)
values=[max_roi_ps,max_roi_365]
labels=["Pinnacle\ngood prediction for all matches","Bet365\ngood prediction for all matches"]
xaxis_label="Return on investment (ROI) in %"
title="Betting on all ATP matches since 2011"
basic_horizontal_barplot(values,labels,xaxis_label,title,[0,100],(4,2.5))








import seaborn as sns
def basic_horizontal_barplot(values,labels,xaxis_label,title,xlim=None,figsize=None):
    """
    Please provide the labels corresponding to the values, the plot title, and the xaxis label.
    """
    # Possible colors to use - if not enough, colors are reused
    cs=["coral","tomato","peachpuff","orange","gold","firebrick","peru","khaki","chocolate"]
    cs=cs*(len(values)//len(cs)+1)
    # The figure
    if figsize==None:
        fig=plt.figure(figsize=(4,3))
    else:
        fig=plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,0.9])
    color=cs[:len(values)]
    ax.barh(range(len(values)),values,color=color)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    if xlim!=None:
        ax.set_xlim(xlim)
    plt.suptitle(title)
    ax.set_xlabel(xaxis_label)

