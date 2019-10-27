import  pickle as cPickle
import time
DEFAULTPREDICT='./predict'
dict={'a':1,'b':2,'c':3}
with open(DEFAULTPREDICT + '/%d' % int(time.time()), 'wb') as savepre:
    cPickle.dump(dict, savepre)