from multiprocessing import Process
from multiprocessing import Pool,Queue
import time
def calc_hugenum(order):
    num=0
    for i in range(10000000):
        num+=i
    print (order,'finish')
    return num

if __name__=='__main__':
    pool=Pool(processes=8)
    start_time=time.time()
    relist=[]
    for i in range(8):

        relist.append(pool.apply_async(calc_hugenum,[i]))

    pool.close()
    pool.join()

    end_time=time.time()

    for re in relist:
        print (re.get())

    print('all time ',end_time-start_time)
