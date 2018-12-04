def m_analysis(dat):
    '''
    计算每天的所需特征，比如：目前计算的是入睡时间(分钟)
    '''
    sleep_cost=0
    for i in range(len(dat)-2):
        if(dat[i][3]==0 and dat[i+1][3]==1 and dat[i+2][3]==2):
            sleep_cost=dat[i+1][2]-dat[i][2]
        #elif(dat[i][3]==0 and dat[i+1][3]==1):
            #sleep_cost=dat[i+1][2]-dat[i][2]
            
            
    return round(sleep_cost/60,2)
    
 
import pymysql
import numpy as np
conn=pymysql.connect('localhost','root','19951024','sleep')
cur=conn.cursor()
cur.execute('select distinct(date) from tb_gu_sleep where uid=2  order by date asc')
date=cur.fetchall()

sleep_cost_all=[]
for da in date:
    cur.execute('select * from tb_gu_sleep where uid=2 and date={} order by time asc'.format(da[0]))
    dat=cur.fetchall()
    sleep_cost=m_analysis(dat)
    sleep_cost_all.append(sleep_cost)
cur.close()
conn.close()

import matplotlib.pyplot as plt
plt.plot(range(len(sleep_cost_all)),sleep_cost_all)
plt.show()


# -*- coding:utf-8 -*-
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
testStationarity(sleep_cost_all)



import datetime
import pandas as pd
import pandas_datareader.data as web
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import statsmodels.tsa.stattools as ts
from scipy import  stats
import pywt
def show_acf_pacf(ts):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=20,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout();

    fig = plt.figure(figsize=(12,8))
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=20, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout();
show_acf_pacf(sleep_cost_all)


import matplotlib.pyplot as plt
ts_log = np.log(sleep_cost_all)
def show_figure(ts):
    plt.plot(range(len(ts)),ts)
    plt.show()


import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(sleep_cost_all,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order
#结果是（0，0）

from statsmodels.tsa.arima_model import ARMA
model = ARMA(sleep_cost_all, order=(2,1))#滑动平稳
result_arma = model.fit(disp=-1, method='css')#拟合
train_predict = result_arma.predict(7)#预测


#均方根误差
np_arr=np.array(sleep_cost_all)
RMSE = np.sqrt(((train_predict-np_arr[7:])**2).sum()/np_arr[7:].size)
print(RMSE)


def show_compare(old_series,prodict_series):
    plt.plot(range(len(old_series)), old_series, color='green', label='old_series')
    plt.plot(range(len(prodict_series)), prodict_series, color='red', label='prodict_series')
    plt.show()
show_compare(np_arr,train_predict)  


from PyEMD import EMD
import numpy as np
emd=EMD()
sleep=np.array(sleep_cost_all)
imfs=emd(sleep)


ts_imp0=pd.Series(imfs[0])
testStationarity(ts_imp0)
show_acf_pacf(ts_imp0)
from statsmodels.tsa.arima_model import ARMA
model_imp0 = ARMA(ts_imp0, order=(1,1))#滑动平稳
result_arma_imp0 = model_imp0.fit(disp=-1, method='css')#拟合
train_predict_imp0 = result_arma_imp0.predict(7)#预测

RMSE = np.sqrt(((train_predict_imp0-ts_imp0[7:])**2).sum()/ts_imp0[7:].size)
print(RMSE)
show_compare(ts_imp0,train_predict_imp0)

all_prodict={}
all_pq={}
all_pq['imf0']=(1,1)
all_prodict['imf0']=train_predict_imp0

ts_imp1=pd.Series(imfs[1])
testStationarity(ts_imp1)
show_acf_pacf(ts_imp1)

import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(imfs[1],max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order

from statsmodels.tsa.arima_model import ARMA
model_imp1= ARMA(ts_imp1, order=(1,4))#滑动平稳
result_arma_imp1 = model_imp1.fit(disp=-1, method='css')#拟合
train_predict_imp1 = result_arma_imp1.predict(7)#预测

RMSE = np.sqrt(((train_predict_imp1-ts_imp1[7:])**2).sum()/ts_imp1[7:].size)
print(RMSE)
show_compare(ts_imp1,train_predict_imp1)

all_pq['imf1']=(1,4)
all_prodict['imf1']=train_predict_imp1

ts_imp2=pd.Series(imfs[2])
testStationarity(ts_imp2)
show_acf_pacf(ts_imp2)

import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(imfs[2],max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order


from statsmodels.tsa.arima_model import ARMA
model_imp2= ARMA(ts_imp2, order=(2,5))#滑动平稳
result_arma_imp2 = model_imp2.fit(disp=-1, method='css')#拟合
train_predict_imp2 = result_arma_imp2.predict(7)#预测

RMSE = np.sqrt(((train_predict_imp2-ts_imp2[7:])**2).sum()/ts_imp2[7:].size)
print(RMSE)
show_compare(ts_imp2,train_predict_imp2)

all_pq['imf2']=(2,5)
all_prodict['imf2']=train_predict_imp2


ts_imp3=pd.Series(imfs[3])
testStationarity(ts_imp3)
show_acf_pacf(ts_imp3)

import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(imfs[3],max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order

from statsmodels.tsa.arima_model import ARMA
model_imp3= ARMA(ts_imp3, order=(3,3))#滑动平稳
result_arma_imp3 = model_imp3.fit(disp=-1, method='css')#拟合
train_predict_imp3 = result_arma_imp3.predict(7)#预测

RMSE = np.sqrt(((train_predict_imp3-ts_imp3[7:])**2).sum()/ts_imp3[7:].size)
print(RMSE)
show_compare(ts_imp3,train_predict_imp3)

all_pq['imf3']=(3,3)
all_prodict['imf3']=train_predict_imp3

ts_imp4=pd.Series(imfs[4])
testStationarity(ts_imp4)
show_acf_pacf(ts_imp4)


import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(imfs[4],max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order


from statsmodels.tsa.arima_model import ARMA
model_imp4= ARMA(ts_imp4, order=(4,5))#滑动平稳
result_arma_imp4 = model_imp4.fit(disp=-1, method='css')#拟合
train_predict_imp4 = result_arma_imp4.predict(7)#预测

RMSE = np.sqrt(((train_predict_imp4-ts_imp4[7:])**2).sum()/ts_imp4[7:].size)
print(RMSE)
show_compare(ts_imp4,train_predict_imp4)

all_pq['imf4']=(4,5)
all_prodict['imf4']=train_predict_imp4

ts_imp5=pd.Series(imfs[5])
testStationarity(ts_imp5)
show_acf_pacf(ts_imp5)


import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(imfs[5],max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order

from statsmodels.tsa.arima_model import ARMA
model_imp5= ARMA(ts_imp5, order=(4,1))#滑动平稳
result_arma_imp5 = model_imp5.fit(disp=-1, method='css')#拟合
train_predict_imp5 = result_arma_imp5.predict(7)#预测

RMSE = np.sqrt(((train_predict_imp5-ts_imp5[7:])**2).sum()/ts_imp5[7:].size)
print(RMSE)
show_compare(ts_imp5,train_predict_imp5)


ts_imp6=pd.Series(imfs[6])
testStationarity(ts_imp6)
show_acf_pacf(ts_imp6)


import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(imfs[6],max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order


#确定预测的长度和滞后项，我们采用预测的长度为1，滞后项为7
pred_dat=[]
for i in range(len(imfs[6])-7):
    pred_dat.append(imfs[6][i:i+8])



import numpy as np
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import svm
n_pred_dat=np.array(pred_dat)
dat_train,dat_test = train_test_split(n_pred_dat, train_size=0.8)
#分测试,训练，x,y
X_train=dat_train[:,0:7]
Y_train=dat_train[:,7]

X_test=dat_test[:,0:7]
Y_test=dat_test[:,7]
clf = svm.SVR(kernel='linear')
clf.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
import numpy as np
#Y_pred=clf.predict(X_test)
Y_pred=clf.predict(n_pred_dat[:,0:7])
# accuracy_score(Y_testt,Y_predd)

accuracy_score(np.round(n_pred_dat[:,7],0),np.round(Y_pred,0))

import matplotlib.pyplot as plt
plt.plot(range(len(Y_pred)),Y_pred,color='r')
plt.plot(range(len(n_pred_dat[:,7])),n_pred_dat[:,7],color='g')
plt.show()


RMSE = np.sqrt(((Y_pred-Y_test)**2).sum()/Y_test.size)
print(RMSE)


all_prodict['imf6']=Y_pred

all_prodic_count=all_prodict['imf0']+all_prodict['imf1']+all_prodict['imf2']+all_prodict['imf3']+all_prodict['imf4']+all_prodict['imf5']+all_prodict['imf6']

show_compare(n_sleep_cost_all,all_prodic_count)















