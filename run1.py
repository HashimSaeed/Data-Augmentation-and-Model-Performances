# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:37:30 2019

@author: hashi
"""

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

def comp_accuracy(lr_probs,y_test,labels):
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.', label=labels)

def do_cv_class(df, num_folds, test_model):
    output = pd.DataFrame()         
    kfold_data = KFold(num_folds, shuffle=True, random_state=1000)
    j=1
    for ind1,ind2 in kfold_data.split(df):
        temp = pd.DataFrame()
        x_train, x_test = df.iloc[ind1,:-1], df.iloc[ind2,:-1]
        y_train, y_test = df.iloc[ind1,-1], df.iloc[ind2,-1]
        train = pd.concat([x_train,y_train],axis=1)
        test = pd.concat([x_test,y_test],axis=1)
        m,n = train.shape
        m1,n1 = test.shape
        test_model.fit(train.iloc[:,:n-1], train.iloc[:,-1])
        y_pred = test_model.predict_proba(test.iloc[:,:n1-1])
        y_act = test.iloc[:,-1]
        y_act = [float(i) for i in y_act]
        temp['prediction'] = y_pred[:,1]
        temp['actual'] = y_act
#        temp['class'] = y_class
        arr = j*np.ones(len(y_pred))
        temp['fold']=arr
        output = pd.concat([output,temp],axis=0,ignore_index=True)
        j+=1
          
    return output

def roc_auc(df,label):
    y_pred = df.iloc[:,0].values
    y_true = df.iloc[:,1].values
    fpr,tpr,thr = roc_curve(np.array(y_true),np.array(y_pred),pos_label=1)
    roc_auc = auc(fpr,tpr)
    print("AUC is: ",roc_auc)
#    ax1.plot(fpr, tpr, label=label+' ROC curve (area = %0.4f)'% roc_auc)
#    ax3.plot(fpr, tpr, label=label+' ROC curve (area = %0.4f)'% roc_auc)   
#    plt.figure(figsize = (4,4))
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<=0.5]=0
    print(confusion_matrix(y_true,y_pred))
    return roc_auc

def output(data1,i):
    X = data1.iloc[:,:-1]
    y = data1.iloc[:,-1]
    data = pd.concat([X,y],axis=1)
    f_model = RandomForestClassifier(n_estimators=100, random_state=1000)
    output = do_cv_class(data,10,f_model)
    auc = roc_auc(output,'RF Iter - '+str(i))
    return auc

def step(data1, data2, i):
    m,n = data1.shape
    m1,n1 = data2.shape
    X = data1.iloc[:,:-1]
    y = data1.iloc[:,-1]
    X_t = data2.iloc[:,:-1]
    y_t = data2.iloc[:,-1]
    model = RandomForestClassifier(n_estimators=100, random_state=1000)
    model.fit(X,y)
#    out1= av_cross_val(X,y)
    out1 = model.predict_proba(X)
    data1.insert(n-1,'prob'+str(i),out1[:,1])
    out2 = model.predict_proba(X_t)
    data2.insert(n1-1,'feature'+str(i),out2[:,1])
    return data1,data2

def shapechange(df):
    for i in range(df.shape[1]):
        p_cols = [col for col in df.columns if 'prob'+str(i) in col]
        if len(p_cols)!=0:
            p_cols = p_cols[0]
            df.drop(p_cols, axis=1, inplace=True)           
    return df
    
def av_cross_val(X,y):
    model = RandomForestClassifier(n_estimators=100, random_state=1000)
    ind = []
    output = []
    kf = KFold(2, shuffle=True, random_state=1000)
    y_pred = np.zeros_like(y)
    for train_index,test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]        
        model.fit(X_train,y_train)
        y_p = model.predict_proba(X_test)
        ind.extend(test_index)
        output.extend(y_p[:,1])

    ind = np.asarray(ind)
    output = np.asarray(output)
    idx = np.argsort(ind)
    y_pred = output[idx]
    return y_pred

#fig1, ax1 = plt.subplots()    
#fig3,ax3 = plt.subplots()
arr= []
data1 = pd.read_csv("heart.csv")
data2 = pd.read_csv("cardio_train.csv",sep=";")
## Data1 is Training 
data_new = pd.DataFrame()
data_new['age'] = data2['age']
data_new['gender'] = data2['gender']
data_new['cholestrol'] = data2['cholesterol']
data_new['ap_hi'] = data2['ap_hi']
data_new['target'] = data2['cardio']

## Data 2 is Testing
data_2 = pd.DataFrame()
data_2['age'] = data1['age']
data_2['gender'] = data1['sex']
data_2['cholestrol'] = data1['chol']
data_2['ap_hi'] = data1['thalach']
data_2['target'] = data1['target']
i=0
# Initial Result
print(data_2.shape)
a = output(data_2,i)
arr.append(a)
#corr = data_new.corr()
#sm.graphics.plot_corr(corr, xnames=list(corr.columns))
data1 = data_2
data2 = data_new
############### Iteration 1 ########################
for i in range(1,5):
    data1, data2 = step(data1,data2,i)
    data2, data1 = step(data2,data1,i)
    data1 = shapechange(data1)
    data2 = shapechange(data2)
    a = output(data1,i)
    arr.append(a)
#    corr = data1.corr()
#    sm.graphics.plot_corr(corr, xnames=list(corr.columns))


ax1.plot([0,1],[0,1],linestyle='--')
ax1.legend()
ax1.set_title("Random Forest with Additional Features",fontsize=16)
ax1.set_ylabel('tpr')
ax1.set_xlabel('fpr')

ax3.set_title("Random Forest with Additional Features (Log scale)",fontsize=16)
ax3.set_ylabel('tpr')
ax3.set_xlabel('fpr')
ax3.set_xscale('log')
ax3.legend()

x = np.arange(1,len(arr)+1)
fig2, ax2 = plt.subplots()
ax2.plot(x, arr, 'ro--')
ax2.set_ylabel('AUC')
ax2.set_xlabel('Iterations')
ax2.set_title("AUC with respect to Iterations",fontsize=16)