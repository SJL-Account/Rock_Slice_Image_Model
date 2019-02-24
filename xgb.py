import xgboost as xgb
import  import_data
import pandas as pd
from sklearn.utils import  shuffle
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.cross_validation import train_test_split

print ('--------------start import data----------------')
#数据量5879075
data = import_data.import_data('Align_Pixel_RGB1.csv')
data = shuffle(data)
train_size=100000
test_size=5000

print('--------------start split data----------------')
y = data.pop('o_label')
x = data
x_train = x.values[(1 - 1) * train_size:1 * (train_size)]
y_train = y.values[(1 - 1) * train_size:1 * (train_size)]
x_test = x.values[-test_size:]
y_test = y.values[-test_size:]


dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test,label=y_test)


print ('--------------start create model----------------')





print ('--------------start train------------------------')

start_time=time.time()
evals_result={}

'''
#参数扰动
params =[ {'max_depth':2, 'eta':1.0, 'silent':1, 'objective':'multi:softmax','num_class':19,'alpha':0.1,'subsample':0.6,'colsample_bytree':0.6},
          {'max_depth':2, 'eta':0.5, 'silent':1, 'objective':'multi:softmax','num_class':19,'alpha':10,'subsample':0.7,'colsample_bytree':0.7},
          {'max_depth': 2, 'eta': 0.5, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 19, 'lambda':10,'subsample':0.8,'colsample_bytree':0.8},
          {'max_depth': 2, 'eta': 0.5, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 19, 'lambda': 0.1,'subsample': 0.9, 'colsample_bytree': 0.9},]

train_predict_result=np.ones((train_size,len(params)))
predict_result=np.ones((test_size,len(params)))
for i,param in enumerate(params):
    bst=xgb.train(param, dtrain, num_boost_round=40,evals=[(dtrain,'dtrain'),(dtest,'dtest')],  obj=None, feval=None,
              maximize=False, early_stopping_rounds=None, evals_result=evals_result,
              verbose_eval=True, xgb_model=None, callbacks=None)
    predict_result[:,i]=bst.predict(dtest)
    train_predict_result[:,i]=bst.predict(dtrain)


#逻辑回归
lr=LogisticRegression()

lr.fit(train_predict_result,y_train)


'''

param={'max_depth':4, 'eta':1.0, 'silent':1, 'objective':'multi:softmax','num_class':19,'alpha':10,'subsample':0.6,'colsample_bytree':0.6}
for i in range(1,11):

    bst = xgb.train(param, dtrain, num_boost_round=40, evals=[(dtrain, 'dtrain'), (dtest, 'dtest')], obj=None, feval=None,
                    maximize=False, early_stopping_rounds=None, evals_result=evals_result,
                    verbose_eval=True, xgb_model=None, callbacks=None)



end_time=time.time()

print ('all_time:',end_time-start_time)


print ('-------------------start predict-------------------')
result_df = pd.DataFrame(lr.predict(predict_result),columns=['pre_result'])
result_df['real_result'] = y_test
result_df.to_csv('predict_result_xgb.csv', index=None)

print ((result_df[result_df['pre_result'].astype('int32')==result_df['real_result']].astype('int').sum())/test_size)
print ('--------------start plot-------------------------')
plt.plot(1.0-np.array(evals_result['dtest']['merror']))
plt.plot(1.0-np.array(evals_result['dtrain']['merror']))
plt.show()










'''
clf=xgb.XGBClassifier(max_depth=3, learning_rate=0.1,
                 n_estimators=20, silent=False,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=-1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None)

clf.fit(x_train, y_train,
        #eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric='error',
        verbose=True)

'''
'''
dtrain.save_binary('dtrain.train')
dtrain.save_binary('dtest.test')

'''
