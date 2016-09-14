import numpy as np
import scipy.spatial.distance

#
# exp(-0.5 ||x1-x2||_2^2 / (2*sqrt(length_scale)))
#
class RBFKernel:
    
    def __init__(self,length_scale=1.0):  
        
        self.length_scale=length_scale       
        
    def compute(self,X,Y):
        
        sqdist=scipy.spatial.distance.cdist(X,Y,'euclidean')
        sqdist=sqdist*sqdist
    
        return np.exp(sqdist*(-1/(2*self.length_scale*self.length_scale)))         