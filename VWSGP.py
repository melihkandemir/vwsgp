# 
# The Variational Weakly Supervised Gaussian Process
#
# Source: M. Kandemir et al., Variational Weakly Supervised Gaussian Processes, BMVC, 2016
#
# Copyright: Melih Kandemir, melihkandemir@gmail.com
#
# All rights reserved, 14.09.2016
# 

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import logistic
import cPickle as pickle

class VWSGP:

    #
    # data_kernel: Should be a kernel object instance with "compute(X,Y)" function implemented. See RBFKernel.py for an example.
    # num_inducing: Number of inducing points used by the sparse Gaussian process prior
    # max_iter: Maximum number of variational inference iterations
    # verbose: Whether more computation steps should be reported on the command line                
    # normalize: Whether the data z-score normalized
    # memofriendly: 1-> the entire data set is loaded into memory, 0-> the bags are loaded into memory one by one (recommended for large data sets)
    #
    def __init__(self, data_kernel, num_inducing = 10, max_iter = 50,verbose=0,normalize=1,memofriendly=0):
               
        self.data_kernel=data_kernel        
        self.P = num_inducing        
        self.max_iter = max_iter         
        self.verbose = verbose
        self.memofriendly = memofriendly
        self.normalize=normalize
        
    #
    # datapath: Path to the directory containing each bag in a separate file (see the example under the"elephant_bags" folder)
    # bagnames: A list object containing the names of bag files
    # trainingbags: The index set for training bags
    #
    # Each bag file should contain the following numpy arrays:
    #   DataBag: NxD matrix with N: number of bag instances and D: number of features
    #   label: A binary scalar (i.e. coded as 0 and 1) in any int or float type indicating the bag label
    #
    def train(self,datapath,bagnames,trainbags):
                       
        self.initialize(datapath,bagnames,trainbags)
        
        B = trainbags.shape[0]        
        Epsilons = np.ones([B,1]).ravel()
        Lambdas = np.ones([B,1]).ravel()
        
        for ii in range(self.max_iter):
            if self.verbose:
                print 'Iter: %i\r' % (ii+1)     


            if ii>0:
                mmTplusS = np.outer(self.m,self.m)+self.S
                LmmTplusS = np.linalg.cholesky(mmTplusS)                                  
                                                
            xtemp=np.zeros([self.P,1]).ravel()
            Sinv=np.zeros([self.P, self.P])
            for bb in range(B):
                if self.memofriendly == 1:
                    (Abb,Abb_summed,Bbb,Ctemp) = self.loadBag(bb,datapath,bagnames,trainbags)
                else:
                    Abb = self.A[self.bags[:,0]==bb,:]
                    Bbb = self.B[self.bags[:,0]==bb]
                    Abb_summed = np.sum(Abb,axis=0)                

                # Update S
                Sinv += 2*Lambdas[bb]*np.outer(Abb_summed,Abb_summed)
            
                # update m                
                xtemp += (self.T[bb]-0.5)*np.sum(Abb,axis=0)
                
                if ii>0:
                    CC = Abb.dot(LmmTplusS)
                    CC1 = np.sum(CC,axis=0).ravel()
                    Epsilons[bb] = np.sqrt( CC1.dot(CC1) +  sum(Bbb))  
            

            Sinv += self.Kzz_inv
            self.S = np.linalg.inv(Sinv+np.identity(self.P)*0.00000001) 
            self.m = self.S.dot(xtemp).ravel() 
            
            if ii>0:
                Lambdas = (logistic.cdf(Epsilons)-0.5)*(1./(2.*Epsilons)) 
                        
            # M-Step
                              
            # Update Epsilons
            mmTplusS = np.outer(self.m,self.m)+self.S
            LmmTplusS = np.linalg.cholesky(mmTplusS)
            
            if ii==0:
                for bb in range(B):  
                    if self.memofriendly == 1:
                        (Abb,Abb_summed,Bbb,Ctemp) = self.loadBag(bb,datapath,bagnames,trainbags)
                    else:
                        Abb = self.A[self.bags[:,0]==bb,:]
                        Bbb = self.B[self.bags[:,0]==bb]
                  
                    CC = Abb.dot(LmmTplusS)
                    CC1 = np.sum(CC,axis=0).ravel()
                    Epsilons[bb] = np.sqrt( CC1.dot(CC1) +  sum(Bbb))              
                  
                Lambdas = (logistic.cdf(Epsilons)-0.5)*(1./(2.*Epsilons))  
              
        self.A = 0
        self.B = 0            
        print '\r'
       
    def initialize(self,datapath,bagnames,trainbags):
        
        P = self.P
        Ppos = np.uint32(np.floor(P*0.5))
        B=trainbags.shape[0]
        
        T = np.zeros([B,1]).ravel()
        
        Nrepr_tot=0
        Ntot=0
                                 
        for bb in range(B):
                 
            bag = pickle.load(open(datapath + '/' + bagnames[trainbags[bb]],'rb'))               
            Xrepr_bb = np.array(bag["DataBag"])
            label = bag["label"]            
            
            T[bb] = label                
            
            Nrepr_tot += Xrepr_bb.shape[0]
            
            Ntot+=Xrepr_bb.shape[0]
            
            
            if bb==0:
                Xmeans = np.zeros([B,Xrepr_bb.shape[1]])
                
            Xmeans[bb,:] = np.mean(Xrepr_bb,axis=0)
            
            
            if bb == 0:
                Xrepr = Xrepr_bb
                Trepr = np.ones([Xrepr_bb.shape[0],1]).ravel()*T[bb]
                
            else:
                Xrepr = np.concatenate((Xrepr,Xrepr_bb),axis=0)
                Trepr_bb = np.ones([Xrepr_bb.shape[0],1]).ravel()*T[bb]
                Trepr = np.concatenate((Trepr,Trepr_bb),axis=0)                
            
             
            if self.verbose > 0:
                  print '    Bag %6i initialized;\r' % (bb)                  
            
        # normalize
        self.data_mean = np.mean(Xrepr,axis=0)
        self.data_std = np.std(Xrepr,axis=0)
        self.data_std[self.data_std==0.] = 1.
         
        self.T = T
                
        if self.normalize == 1 :
            Xmeans = (Xmeans - np.tile(self.data_mean,[B,1])) / np.tile(self.data_std,[B,1])
            Xrepr = (Xrepr - np.tile(self.data_mean,[Nrepr_tot,1])) / np.tile(self.data_std,[Nrepr_tot,1])
        
        Xzeros=Xrepr[Trepr==0,:]
        Xzeros[np.isnan(Xzeros)]=0.0
        Xzeros=np.float32(Xzeros)

        Xones=Xrepr[Trepr==1,:]
        Xones[np.isnan(Xones)]=0.0
        Xones=np.float32(Xones)
        
        kmeans_model0 = KMeans(init='k-means++', n_clusters=P-Ppos, n_init=1).fit(Xzeros)        
        kmeans_model1 = KMeans(init='k-means++', n_clusters=Ppos, n_init=1).fit(Xones)        
         
        self.Z = np.concatenate((kmeans_model0.cluster_centers_,  kmeans_model1.cluster_centers_))
        self.Kzz=self.data_kernel.compute(self.Z,self.Z)
        self.Kzz_inv = np.linalg.inv(self.Kzz + np.identity(self.P)*0.0000001)
        
        Tsign=Trepr*1.0;
        Tsign[Trepr==0] = -1.
        
        self.m = np.random.random([self.P,1]).ravel()
        self.S = np.identity(self.P) + np.random.random([self.P,self.P])*0.01
        
        if self.memofriendly == 0:
                    
            for bb in range(B):
                if self.verbose:
                    print '    Bag %6i kernel computed;\r' % (bb)   
                
                (Abb,Abb_summed,Bbb,Ctemp) = self.loadBag(bb,datapath,bagnames,trainbags)
                Nb = Abb.shape[0]
                
                if bb == 0:
                    self.bags = np.zeros([Nb,1])
                    self.A = Abb
                    self.B = Bbb
                    
                else:
                    self.bags = np.concatenate((self.bags, np.ones([Nb,1])*bb ))
                    self.A =  np.concatenate((self.A, Abb ))
                    self.B =   np.concatenate((self.B, Bbb ))                                        
        
    def loadBag(self,bagno,datapath,bagnames,trainbags):
        bag = pickle.load(open(datapath + '/' + bagnames[trainbags[bagno]],'rb'))                             
        X = np.array(bag["DataBag"])            
        X[np.isnan(X)] = 0.
        X[np.isinf(X)] = 0.
        Nb = X.shape[0]
                    
        if self.normalize == 1 :
            X = (X - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])                
        
        Kzx=self.data_kernel.compute(self.Z,X)        
        L_Kzz_inv = np.linalg.cholesky(self.Kzz_inv)
        Ctemp = Kzx.T.dot(L_Kzz_inv)
       
        Abb = Kzx.T.dot(self.Kzz_inv)
        Abb_summed = np.sum(Abb,axis=0)  
        Bbb = np.ones([Nb,1]).ravel()- np.sum(Ctemp*Ctemp,axis=1) 
        
        return (Abb,Abb_summed,Bbb,Ctemp)

    def logistic(self,x):
        return 1. / (1.+np.exp(-x))                
        
    def predict(self, datapath, bagnames,testbags):
        
      B = testbags.shape[0]  
      probabilities = np.zeros([B,1]).ravel()
      
      for bb in range(B):
          
            bag = pickle.load(open(datapath + '/' + bagnames[testbags[bb]],'rb'))                             
            Xts = np.array(bag["DataBag"])            
            Xts[np.isnan(Xts)] = 0.
            Xts[np.isinf(Xts)] = 0.
            Nb = Xts.shape[0]        
            
            if self.normalize == 1 :            
                Xts = (Xts - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])
        
            Kxz = self.data_kernel.compute(Xts,self.Z)
            Db = Kxz.dot(self.Kzz_inv)
            
            Ab = Db.dot(self.m)
            Cb = np.ones([Nb,1]).ravel() - np.sum(Db*Kxz,axis=1).ravel()
            
            Fsamples = np.zeros([Nb,100])
           
            for nn in range(Nb):
                Fsamples[nn,:] = np.random.normal(Ab[nn],np.sqrt(Cb[nn]),100)            
            
            probabilities[bb] = np.mean(self.logistic( np.sum(Fsamples,axis=0) ))
            
            if self.verbose > 0:
                  print '    Bag %6i predicted;\r' % (bb)      
            
      return (probabilities>0.5)*1
