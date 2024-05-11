# Credit-Card-Fraud-Detection
#project for credit card fraud detection
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec

dataset=pd.read_csv("creditcard.csv")
print(dataset.head())

#BELOW PROG FOR FINDING MISSING VALUES IN DATA 

#print("Non missing values:",str(dataset.isnull().shape[0]))
#print("Missing values:",str(dataset.shape[0]-dataset.isnull().shape[0]))

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler().fit (dataset[["Time","Amount"]])
dataset[["Time","Amount"]]=scaler.transform(dataset[["Time","Amount"]])

print(dataset.head())

y=dataset["Class"]#TARGET
x=dataset.iloc[:,0:30]#FETAURES

from sklearn.model_selection import _search, train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,cross_val_score,RandomizedSearchCV

kf=StratifiedKFold(n_splits=5,random_state=None,shuffle=False)

#imbalance learn module, only 20 is of test

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,recall_score,precision_score,f1_score 

#Importing all Classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
#rf.fit(Q_train,P_train)
#yp=rf.predict(Q_test)
# rf.fit(x_train,y_train)
# yp = rf.predict(x_test)
# print("accuracy is :",accuracy_score(y_test,yp))
# print("precision:",precision_score(y_test,yp))
# print("recall:",recall_score(y_test,yp))
# print("f1 score",f1_score(y_test,yp))

#under sampling function

def get_model_best_estimator(estimator,params,kf=kf,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,is_gridsearch=True,sampling=NearMiss(),scoring="f1",n_jobs=2):
   if sampling is None:
        pipeline=make_pipeline(estimator)
   else:
      pipeline=make_pipeline(sampling,estimator)
   estimator_name=estimator.__class__.__name__.lower()
   # new_param = {f"{estimator_name}__{key}": params[key] for key in params}
   new_param={f"{estimator_name}__{key}":params [key] for key in params}
   if is_gridsearch:
      search=GridSearchCV(pipeline,param_grid=new_param,cv=kf,return_train_score=True,n_jobs=n_jobs,verbose=2)
   else:
      search=RandomizedSearchCV(pipeline,param_distributions=new_param,cv=kf,scoring=scoring,return_train_score=True,n_jobs=n_jobs,verbose=1)
   search.fit(x_train,y_train)
   cv_score=cross_val_score(search,x_train,y_train,scoring=scoring,cv=kf)
   y_pred=search.best_estimator_.named_steps[estimator_name].predict(x_test) 
   recall=recall_score(y_test,y_pred,pos_label=1)
   accuracy=accuracy_score(y_test,y_pred)
   f1=f1_score(y_test,y_pred)
   y_prob=search.best_estimator_.named_steps[estimator_name].predict_proba(x_test)[::,1]
   fpr,tpr,_=roc_curve(y_test,y_prob)
   auc=roc_auc_score(y_test,y_prob)
   return{
      "best_estimator":search.best_estimator_,
      "estimator_name":estimator_name,
      "cv_score":cv_score,
      "recall":recall,
      "accuracy":accuracy,
      "f1_score":f1,
      "fpr":fpr,
      "tpr":tpr,
      "auc":auc,
            }
res_table=pd.DataFrame(columns=['classifier','fpr','tpr','auc'])
result=get_model_best_estimator(estimator=LogisticRegression(),params={"penalty":['l1','l2'],'C':[0.01,0.1,1,100],'solver':['liblinear']},sampling=NearMiss())
#print(result)
# print("Estimator_name",_search.best_estimator_)
# print("Accuracy", _search.CV_results_['mean_test_score'][_search.best_index_])
# print("Recall", _search.CV_results_['mean_test_recall'][_search.best_index_])
# print("F1 score", _search.CV_results_['mean_test_f1'][_search.best_index_])
# print("Estimator_name",result["search.best_estimator"])
# print("Accuracy",result["accuracy"])
# print("recall",result["recall"])
# print("f1 score",result["f1_score"]) .

res_table=res_table.add({'classifier':result["estimator_name"],'fpr':result["fpr"],'tpr':result["tpr"],'auc':result["auc"]})

#below starts with over sampling(o_result is over sampling)

res_table=res_table.add({'classifier':result["estimator_name"],
                         'fpr':result["fpr"],
                         'tpr':result["tpr"], 
                         'auc':result["auc"]})
o_result=get_model_best_estimator(estimator=LogisticRegression(),params={"penalty":['l1','l2'],'C':[0.01,0.1,1,100,100],'solver':['liblinear']},sampling=SMOTE(random_state=42),scoring='f1',is_gridsearch=False,n_jobs=2)
print(o_result) 
res_table=res_table.add({'classifier':o_result["estimator_name"],
                         'fpr':o_result["fpr"],
                         'tpr':o_result["tpr"],
                         'auc':o_result["auc"]})
print(res_table)
