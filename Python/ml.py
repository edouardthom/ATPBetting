import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.model_selection import StratifiedKFold,KFold

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
