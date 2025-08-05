# Based on https://gist.github.com/Teagum/460a508cda99f9874e4ff828e1896862

import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import math
import time

def hellinger(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Original, ugly version.
    """
    return sum([(math.sqrt(t[0])-math.sqrt(t[1]))*(math.sqrt(t[0])-math.sqrt(t[1]))\
                for t in zip(p,q)])/math.sqrt(2.)
    
def hellinger2(p, q):
    """Hellinger distance between two discrete distributions. 
       In pure Python.
       Some improvements.
    """
    return math.sqrt(sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q) ]) / 2)
  
def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Same as hellinger2 but without list comprehension
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2
        
        # append 
        list_of_squares.append(s)
    
    # calculate sum of squares
    sosq = sum(list_of_squares)    
    
    return math.sqrt(sosq / 2)
  
def hellinger_dot(p, q):
    """Hellinger distance between two discrete distributions. 
       Using numpy.
       For Python >= 3.5 only"""
    z = np.sqrt(p) - np.sqrt(q)
    return np.sqrt(z @ z / 2)
  
def hellinger_fast(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Fastest version.
    """
    return sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p,q) ])