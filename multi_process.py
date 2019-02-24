from multiprocessing import Process

import time
def calc_hugenum(order):
    num=0
    for i in range(1000000000):
        num+=i
    print (order,'finish')

if __name__ == '__main__':

    start_time=time.time()

    for i in range(8):
        calc_pro=Process(target=calc_hugenum,args=[i,])
        calc_pro.start()
        calc_pro.join()
    end_time=time.time()
    print('all time ',end_time-start_time)