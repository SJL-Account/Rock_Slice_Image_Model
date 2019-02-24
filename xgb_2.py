import xgboost as xgb
import  import_data
import pandas as pd
from sklearn.utils import  shuffle
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.cross_validation import train_test_split


def vote_result(e_list,test):
    pre_result=np.zeros((test_size,len(e_list)))
    for i,e in enumerate(e_list):
        pre_result[:,i]=e.predict(test)

    result_df=pd.DataFrame(pre_result)

    vote_retult=np.zeros((test_size))

    for i in range(test_size):
        vote_retult[i] = result_df.ix[i].value_counts().index[0]

    print ('merror',((y_test == vote_retult).astype('int').sum()) / test_size)

    print ('save final result。。。。')

    result_df['real_result_xgb']=y_test

    result_df.to_csv('predict_result_xgb.csv',index=None)



print ('--------------start import data----------------')
#数据量5879075
data = import_data.import_data('Align_Pixel_RGB1.csv')
data_test=import_data.import_data('Align_Pixel_test.csv')
data = shuffle(data)
train_size=100000
test_size=3000

print('--------------start split data----------------')
y = data.pop('o_label')
x = data

y_test = data_test.pop('o_label')
x_test = data_test

dtest=xgb.DMatrix(x_test.values,label=y_test.values)


print ('--------------start create model----------------')



print ('--------------start train------------------------')

start_time=time.time()
evals_result={}


param={'max_depth':4, 'eta':1.0, 'silent':1, 'objective':'multi:softmax','num_class':19,'alpha':10,'subsample':0.6,'colsample_bytree':0.6}
bst_list=[]
for i in range(1,11):

    x_train = x.values[(i - 1) * train_size:i * (train_size)]
    y_train = y.values[(i - 1) * train_size:i * (train_size)]
    dtrain = xgb.DMatrix(x_train, label=y_train)

    bst = xgb.train(param, dtrain, num_boost_round=40, evals=[(dtrain, 'dtrain'), (dtest, 'dtest')], obj=None, feval=None,
                    maximize=False, early_stopping_rounds=None, evals_result=evals_result,
                    verbose_eval=True, xgb_model=None, callbacks=None)

    bst_list.append(bst)

end_time=time.time()

print ('all_time:',end_time-start_time)


print ('-------------------start predict-------------------')

vote_result(bst_list,dtest)

print ('--------------start plot---------------------------')

