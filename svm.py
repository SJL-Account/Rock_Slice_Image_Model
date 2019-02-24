import  import_data
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.cross_validation import  train_test_split
from multiprocessing import Process,Pool
import time
import numpy as np

def svm_paralle(svm,x_train,y_train):
    svm.fit(x_train, y_train)
    print ('over')
    return svm
if __name__=='__main__':
    print ('import data')
    data = import_data.import_data('Align_Pixel_RGB1.csv')
    data = shuffle(data)

    print('--------------start split data----------------')

    y = data.pop('o_label')
    x = data
    train_size=10000
    test_size=3000

    data_test = import_data.import_data('Align_Pixel_test.csv')
    y_test = data_test.pop('o_label').values
    x_test = data_test.values

    print('--------------start create model----------------')
    svm_rbf = SVC(C=10.0, kernel='rbf', degree=3, gamma=0.00001,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None)
    svm_poly = SVC(C=1.0, kernel='poly', degree=2, gamma=0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None)


    print('--------------start train----------------')

    start_time=time.time()

    svm_list=[]

    pool=Pool(processes=20)
    for i in range(1,21):
        x_train=x.values[(i-1)*train_size:i*(train_size)]
        y_train=y.values[(i-1)*train_size:i*(train_size)]
        svm_rbf_porcess = pool.apply_async(svm_paralle, args=[svm_rbf,x_train,y_train])
        svm_poly_process = pool.apply_async(svm_paralle, args=[svm_poly,x_train,y_train])
        print ('启动进程 svm_-----------',i,'---------------------------')
        svm_list.append(svm_rbf_porcess)
        svm_list.append(svm_poly_process)
    pool.close()
    pool.join()

    end_time=time.time()

    print ('all_time:',end_time-start_time)


    print ('--------------start predict----------------')
    predict_result=np.zeros((test_size,20))
    for i,svm in enumerate(svm_list):
        svm_instance= (svm.get())
        if i<20:
            predict_result[:, i]=svm_instance.predict(x_test)

    result_df=pd.DataFrame(predict_result)
    vote_retult=pre_result=np.zeros((test_size))
    for i in range(test_size):
        vote_retult[i] = result_df.ix[i].value_counts().index[0]

    print (((y_test == vote_retult).astype('int').sum()) / test_size)

    print ('save final result。。。。')
    result_df['real_result_svm']=y_test

    result_df.to_csv('predict_result_svm.csv',index=None)