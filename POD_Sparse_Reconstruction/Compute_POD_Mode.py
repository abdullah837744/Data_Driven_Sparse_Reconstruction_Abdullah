# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:15:29 2018

@author: Abdullah
"""

######## Compute POD Mode #############
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def Compute_Phi(mydata,flag,cutoff,option1,option2,coordin):
    
    (M,N) = mydata.shape
    
    if flag == 1:
        meanU = np.mean(mydata, axis=1)
        mydata_nomean = np.zeros((M,N))
        
        for i in range(N):
            mydata_nomean[:,i] = mydata[:,i] - meanU
                        
        
    elif flag == 0:
        mydata_nomean = mydata
        meanU = np.mean(mydata, axis=1)
        
         
        
    R = np.matmul((np.transpose(mydata_nomean)),mydata_nomean)
    
    D,V = la.eig(R)
    
    for i in range(N):
        if D[i] < 0:
            D[i] = 0
    D = np.sort(D)[::-1]
    np.savetxt('Singular.out',D,fmt='%15.8f',header= "Sigma")      
    TotalD = np.sum(D)
    PercD = np.divide(D,TotalD)
    TotalE = 0
    nmode = 0
    
########### Output Energy #####################    
    if option2 == 1:
        os.chdir('..')
        os.mkdir(r'O_POD')
        os.chdir(r'O_POD')
        
        TotalD = np.sum(D)
        Doutput1 = np.divide(D,TotalD)
        Doutput1 = np.reshape(Doutput1,(N,1))
        Doutput2 = np.zeros((N,1))
        num = np.zeros((N,1))
        for i in range(N):
            num[i]=i+1
            if i==0:
                Doutput2[i]=Doutput1[0]
            else:
                Doutput2[i]= Doutput1[i] + Doutput2[i-1]
        
        Energy = np.concatenate((num,Doutput1,Doutput2),axis=1)
        header1 = "Mode, Independent Energy, Cumulative Energy"
        #np.savetxt('Energy.out',Energy,fmt='%10d, %15.8f, %15.8f',header= header1)
        np.savetxt('Energy.out',Energy,header= header1)
        
        #### Make Energy Plot #####
        fig, ax1 = plt.subplots(figsize=(15, 10))
        ax1.plot(Doutput1,'ko-',markersize=15)
        ax1.set_yscale('log')
        ax1.set_xlabel(r'$K$',fontsize = 35)
        ax1.set_ylabel(r'$\sigma_n$',fontsize = 40)
        ax1.tick_params(size=10,width=2,labelsize=30)
        plt.grid(True)
        plt.savefig('I_energy.pdf',bbox_inches='tight')
        
        fig, ax2 = plt.subplots(figsize=(15, 10))
        ax2.plot(Doutput2,'ko-',markersize=15)
        ax2.set_xlabel(r'$K$',fontsize = 35)
        ax2.set_ylabel(r'CEF',fontsize = 35)
        ax2.tick_params(size=10,width=2,labelsize=30)
        plt.grid(True)
        plt.savefig('C_energy.pdf',bbox_inches='tight')
        
    if option1 == 1:
        cumulativeD = 0;
        for i in range(cutoff):
            cumulativeD = np.divide(D[i],TotalD) + cumulativeD
        cutoff = cumulativeD
    elif option1 == 0:
        cutoff = cutoff
    
    for i in range(N):
        if TotalE < cutoff:
            nmode = nmode + 1
            TotalE = TotalE + PercD[i]
        else:
            D[i] = 0
        
    #### Diagonal matrix containing the square root of the eigenvalues
    
    S = np.sqrt(D)
    Scutoff = S[0:nmode]
    np.savetxt('Scutoff.out',Scutoff,fmt='%15.8f',header= "phi")
    Scutoff_inv = np.diag(np.divide(1,Scutoff))
    np.savetxt('Scutoff_inv.out',Scutoff_inv,fmt='%15.8f',header= "phi")
    ##### Compute Phi ######
    
    phi = np.matmul((np.matmul(mydata_nomean,V[:,0:nmode])),Scutoff_inv)
    Maxsingularlr = np.amax(S)
    #np.savetxt('Phi.out',phi,fmt='%15.8f',header= "phi")
    ######## Output modes and energy ##########
    
    if option2 == 1:
        
        ## Output the first 10 modes
        l = int(M/2)
        writemode = np.zeros((l,4))
        writemode[:,0:2]= coordin
        
        if nmode > 10:
            number=10
        else:
            number = nmode
        
        for i in range(number):
            print('Writing out POD Mode data: ', str(i+1))
            writemode[:,2:4]= np.reshape(phi[:,i],(l,2),order = 'F')
            np.savetxt('PODMode_'+str(i+1)+'.out',writemode,fmt='%15.8f',delimiter=',',header="Energy")
    
    os.chdir('..')
            
            
    
        
        
        
        
    
    