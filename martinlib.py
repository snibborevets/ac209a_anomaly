#!/usr/bin/env python3

import numpy as np
import scipy as sp
import pandas as pd
import sys
from sklearn.utils import shuffle


def strangeness_nn(xtr, ytr, x, y):
    '''Return the Nearest-Neighbor strangeness measure of the data (x,y).
    
    Params:
    xtr (list of numpy.ndarray): A list of n training predictors
    ytr (list of numpy.ndarray): A list of n training classifications
    x (numpy.ndarray): The predictors for which to calculate strangeness
    y (numpy.ndarray): The predictions for which to calculate strangeness

    Returns:
    alpha (float): A numerical measure of strangeness for the point (x,y). '''

    # Initialize distances at infinity
    min_n = np.inf
    min_d = np.inf

    # Find the distances `min_n` and `min_d` between (x,y) and
    # 1) The training point closest to (x,y) with the same y
    # 2) The training point closest to (x,y) with a different y
    for i in range(len(xtr)):
        norm = np.linalg.norm(xtr[i]-x, ord=2)
        if ytr[i] == y and min_n < norm: min_n = norm
        elif min_d < norm: min_d = norm

    # Divide and truncate values to bound numeric error
    alpha = round(min_n/min_d, 3)

    # Set alpha to infinity if min_d is 0n
    if np.isnan(alpha): alpha = np.inf

    return alpha

# Tranduces examples into strangeness values given a strangeness function
class EU_Transducer():
    
    '''This class transduces vectors observed on-line into p-values using a strangeness function.
       If 'randomized' is true, this will return a randomized p-value, otherwise it will be deterministic.

    Usage:
    1) Create transducer object: transducer = EU_Transducer(strangeness_fn)
    2) Fit the transducer with some initial data: transducer.partial_fit(xtr, ytr)
    3) As you iterate through a dataset, produce p-values for new observations (x,y):
       transducer.transduce(x, y)'''

    def __init__(self, strangeness_fn, randomized=True):
        '''Create a new transducer.

        Params:
        strangeness_fn (xtr, ytr, x, y): Return a strangeness value alpha for observation (x,y)
        randomized (bool)(optional): True if the p-value should be randomized, false otherwise'''
                   
        try:
            a = np.array([0])
            strangeness_fn(a, a, a, a)
            self._sfn = strangeness_fn
        except:
            raise ValueError('Strangeness function must accept parameters fn(xtr, ytr, x, y)')

        if randomized:
            self._theta = lambda: np.random.rand()
        else:
            self._theta = lambda: 1

        self._isfit = False
            
    def partial_fit(self, train_x, train_y):
        '''Create initial alphas for the transducer with training data.

        Params:
        train_x (list of numpy.ndarray): training predictors
        train_y (list of numpy.ndarray): training classes'''

        # Validate input
        try:
            if len(train_x) != len(train_y):
                raise ValueError('''The numbers of training predictors must 
                   be equal to the number of training classes.''')
        except:
            raise ValueError('`examples` must be a sequence type')
        
        # Compute and store the strangeness function for each training point
        self._xs = []
        self._ys = []
        alphas = []
        for i in range(len(train_x)):
            alphas.append(self._sfn(self._xs, self._ys, train_x[i], train_y[i]))
            print(type(train_x[i]))
            self._xs.append(train_x[i])
            self._ys.append(train_y[i])
        self._alphas = alphas
        self._isfit = True

    def transduce(self, x, y):
        '''Return the p-value for the data point (x,y) given the points observed so far.

        Params:
        x (numpy.ndarray): Predictors for current data point.
        y (nump.ndarray): Prediction for current data point.'''

        # Validate fit
        if not self._isfit:
            raise ValueError('The tranducer must be trained using `partial_fit`')

        # Calculate strangeness value
        alpha = self._sfn(self._xs, self._ys, x, y)

        # Delete oldest strangeness value and store new one
        self._xs = self._xs[1:]
        self._xs.append(x)
        self._ys = self._ys[1:]
        self._ys.append(y)
        self._alphas.append(alpha)

        # Count the number of strangeness values equal to and larger than the observed
        a_greater = 0
        a_equal = 0
        for a in self._alphas:
            if a > alpha: a_greater += 1
            elif a == alpha: a_equal += 1

        # Caluclate p-value (_theta() is random if this transducer is randomized)
        pval = (a_greater + self._theta() * a_equal) / len(self._alphas)
        
        return pval

    def fitted(self):
        '''Return True if the transducer has been fitted, false otherwise.'''
        return self._isfit

class PowerMartingale():

    '''This class implements a power martingale based on a given transducer.

    Usage:
    1) Create PowerMartingale object with a fitted transducer: pm = PowerMartingale(transducer)
    2) To generate the log10 martingale value for a new point, do pm.martingale(x,y) '''
    
    def __init__(self, transducer, epsilon=0.92):
        '''Create new PowerMartingale object

        Params:
        transducer(EUTransducer): The transduce to use when calculating p-values
        epsilon (float)(optional): The power martingale epsilon to use. Default is 0.92.
'''
        
        if not transducer.fitted():
            raise ValueError('Tranducer must be fitted.')
                             
        self._trans = transducer
        self._epsilon = epsilon
        self._step = 0
        self._log_mi = 1.0
                             
    def martingale(self, x, y):
        '''Return the value of the power martingal at new point (x,y)

        Params:
        x (numpy.ndarray): The predictors of the observed data point
        y (numpy.ndarray): The prediction at the observed data point

        Returns:
        (float) The log(base 10) of the martingale at the new point.'''
        
        pi = self._trans.transduce(x, y)
        self._log_mi = self._log_mi - np.log10(self._epsilon*pi**(self._epsilon-1))
        self._step += 1
        return self._log_mi 
                             
if __name__ == '__main__':
    # Use examples


    # Shuffled usps dataset
#    data = shuffle(pd.read_csv('usps_train.csv'))

    # Shuffled satellite data
#    data = shuffle(pd.read_csv('sat.tst', delim_whitespace=True))

    # Unshuffled satellite data
    data = pd.read_csv('sat.trn', delim_whitespace=True)

    # Create transducer
    trans = EU_Transducer(strangeness_nn)

    # Train transducer on first datapoint
    xtr = [np.array(data.ix[0][:-1])]
    ytr = [np.array(data.ix[0][-1])]
    trans.partial_fit(xtr, ytr)

    # Create martingale
    pm = PowerMartingale(trans)

    # Print martingale for every point in data
    i = 0
    for row in data.values:
        x = np.array(row[:-1])
        y = np.array(row[-1])
        print(pm.martingale(x, y))
        i += 1
    
