# 
# The Variational Weakly Supervised Gaussian Process
#
# Source: M. Kandemir et al., Variational Weakly Supervised Gaussian Processes, BMVC, 2016
#
# Copyright: Melih Kandemir, melihkandemir@gmail.com
#
# All rights reserved, 14.09.2016
# 

from VWSGP import VWSGP
from RBFKernel import RBFKernel
import numpy as np
import cPickle as pickle

metadata=pickle.load(open('demo_dataset/elephant_metadata.pkl','rb'))
splits = metadata['splits']
bagnames = metadata['bagnames']
labels = metadata['baglabels']
datapath = "./demo_dataset/elephant_bags"

trainbags = np.where(splits[:,0]!=3)[0]
testbags = np.where(splits[:,0]==3)[0]

yts=labels[testbags].ravel()

#train_bag_indices=bags[np.in1d(bags,train_bags)]
#test_bag_indices=bags[np.in1d(bags,test_bags)]

length_scale = np.sqrt(230.0)
kernel = RBFKernel(length_scale)
testbags

model = VWSGP(kernel, num_inducing = 50, max_iter=10, normalize=0)

model.train(datapath, bagnames, trainbags)

ypred = model.predict(datapath, bagnames, testbags)

print "Accuracy:", np.mean(ypred==yts)*100.0, "per cent"


