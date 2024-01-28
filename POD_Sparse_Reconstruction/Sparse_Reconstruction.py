# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:32:51 2018

@author: Abdullah
"""

##### Sparse Recon ##########

import os

import matplotlib.pyplot as plt

import numpy as np

################# Variables ###########
name = "Input_"
ext = ".in"
fstart = 601
fend = 1186+1
finc = 1

PODMean       = 0  # 1 is to take mean out during POD procedure, 0 is not
select_k  = 15
PODTruncation = 1  # 1 is rank based cutoff; 0 is % based cutoff
PODoutput     = 1  # Output POD modes or not, 0 no; 1 yes


########## Collect Data ################
M = 0
for i in range (fstart,fend,finc):
    M = M+1


os.chdir(r'/media/cfsr123/My4TBHD1/MaterialForAbdullah/Converted Reduced Channel Data (256x36)')

for i in range (fstart,fend,finc):
    print("Collecting data: " + str(i))
    rawdata = np.loadtxt(name + str(i) + ext, delimiter=None,skiprows=2)
    
    if i == fstart:
        nrow=rawdata.shape[0]
        OriginX = np.zeros((nrow*2,M))
        
    OriginX[:,i-fstart]=np.concatenate((rawdata[:,2],rawdata[:,3]),axis=0)
    
   
######## Take mean out
X = np.zeros((nrow*2,M)) 
snapmean = np.mean(OriginX, axis=1)


for i in range(M):
    X[:,i]=OriginX[:,i]-snapmean
    
    
    
############# Call function #######
    
#os.chdir(r'D:\Abdullah _ A20138451\Course Book\practicepython.org')
#import function
#(a,b,c)=function.func(2,3,4)
    
###################################
    
os.chdir(r'/media/cfsr123/My4TBHD1/Abdullah_Windows/Abdullah _ A20138451/Sparse_Recon_Python/Library')
import Compute_POD_Mode
(phi,meanU,mydata_noMean,nmode,maxsin)=Compute_POD_Mode.Compute_Phi(X,PODMean,select_k,PODTruncation,PODoutput,rawdata[:,2:4])    
