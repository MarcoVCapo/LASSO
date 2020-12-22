import numpy as np
from time import time as now
import random
import path2 # Garrigues' Homotopy, Coordinate Descent and LARS
import copy
import time
from numpy import linalg as LA
from sklearn.datasets import make_regression


def LEF(X,y,theta1,nz,mu,d,tipo=0):
    ########################## LASSO Error Function ###########################    
    if tipo>0:
        theta2=np.array([[0.] for ll in range(d)]);
        theta2[nz]=theta1
    else:
        theta2=theta1    
    f=0.50*((X.dot(theta2)-y)**2).sum()+mu*(abs(theta2)).sum()    
    return f

def LassoOptCond(s,r,mu,v1,A,B,C,D,b,c,d,e):
    ################ Verify Optimality Conditions, Eq.13-14 ###################    
    coeff=np.matmul(np.linalg.inv((s**2)*C+A),b+(s**2)*d-s*(1-r)*mu*v1)
    sgrad=-(1.0/((r+s*(1-r))*mu))*((s**2)*(np.matmul(D,coeff)-e)+np.matmul(B,coeff)-c)
    logi1=np.sign(coeff)==v1;logi2=abs(sgrad)<=1
    l1=np.where(logi1==False)[0];l2=np.where(logi2==False)[0]
    return l1,l2,coeff,sgrad


def AFLH(X,X0,y,y0,r,mu,v1,A,B,C,D,b,c,d,e,ac,nac):
    ########################## AFLH Implementation ############################
    se=0;see=-1;
    en=0;sa=0; # Variables that enter (en) the active set of indices and exits it (sa)
    Convergence=False
    while Convergence==False:
        s=1;Transition=False
        while Transition==False:
            l1,l2,coeff,sgrad=LassoOptCond(s,r,mu,v1,A,B,C,D,b,c,d,e)         
            if len(l1)+len(l2)==0:
                se=s
                if s==1:
                    Convergence=True
                    Transition=True
                else:
                    s=0.50*(se+see)
            elif len(l1)+len(l2)==1:
                se=s;      
                if len(l2)==1:
                    en+=1
                    ac=np.append(ac,nac[l2[0]])
                    v1=np.concatenate((v1,np.array([np.sign(sgrad[l2[0]])]).reshape(1,1)),axis=0)
                    nac=np.delete(nac,l2[0])
             
                    X1=X[:,ac];X2=X[:,nac]
                    x1=X0[:,ac];x2=X0[:,nac]
                    V=r*np.matmul(X1[:,len(ac)-1],X1)
                    v=np.matmul(x1[:,len(ac)-1],x1)
                    W=r*np.matmul(X1[:,len(ac)-1],y)-r*mu*v1[len(ac)-1]
                    w=np.matmul(x1[:,len(ac)-1],y0)
                    U=r*np.matmul(X2.T,X1[:,len(ac)-1].T)
                    u=np.matmul(x2.T,x1[:,len(ac)-1].T)                    
                    
                    A=np.concatenate((A,V[range(len(ac)-1)].reshape(len(ac)-1,1)),axis=1)
                    A=np.concatenate((A,V.reshape(1,len(ac))),axis=0)
                    b=np.concatenate((b,W.reshape(1,1)),axis=0)                   
                    B=np.delete(B,l2[0],0);B=np.concatenate((B,U.reshape(len(nac),1)),axis=1)
                    c=np.delete(c,l2[0],0)
                    C=np.concatenate((C,v[range(len(ac)-1)].reshape(len(ac)-1,1)),axis=1)
                    C=np.concatenate((C,v.reshape(1,len(ac))),axis=0)                    
                    d=np.concatenate((d,w.reshape(1,1)),axis=0)
                    D=np.delete(D,l2[0],0);D=np.concatenate((D,u.reshape(len(nac),1)),axis=1)                    
                    e=np.delete(e,l2[0],0)
           
                    Transition=True                
                else:
                    sa+=1
                    nac=np.append(nac,ac[l1[0]])
                    ac=np.delete(ac,l1[0])

                    A=np.delete(A,l1[0],1);v=A[l1[0]];A=np.delete(A,l1[0],0)
                    t1=b[l1[0]]+r*mu*v1[l1[0]];b=np.delete(b,l1[0],0);v1=np.delete(v1,l1[0],0)
                    B=np.delete(B,l1[0],1);B=np.concatenate((B,v.reshape(1,len(ac))),axis=0)
                    c=np.concatenate((c,t1.reshape(1,1)),axis=0)
                    C=np.delete(C,l1[0],1);w=C[l1[0]];C=np.delete(C,l1[0],0)
                    t2=d[l1[0]];d=np.delete(d,l1[0],0)
                    D=np.delete(D,l1[0],1);D=np.concatenate((D,w.reshape(1,len(ac))),axis=0)
                    e=np.concatenate((e,t2.reshape(1,1)),axis=0)
                                               
                    Transition=True  
            else:
                see=s;s=0.50*(se+s)
    return coeff,ac,nac,en,sa
    

def Example(n,dd,r=0.5,n0=0.05,m=0.01):
    ###########################################################################
    ######################## Full Data Set ####################################       
    XM, yM=make_regression(n_samples=n, n_features=dd, random_state=0)
    yM=yM.reshape((n,1))
    ###########################################################################
    arr = np.arange(n)
    np.random.shuffle(arr)
    n1=int(np.floor((1-n0)*n))
    X=XM[arr[range(n1)]];y=yM[arr[range(n1)]] # Original batches
    X0=XM[arr[range(n1,n)]];y0=yM[arr[range(n1,n)]] # New batch
    ###########################################################################
    ################### Computing mu_max for the example ######################
    Xtotal=np.concatenate((X0,np.sqrt(r)*X)); #Data set with adaptative filter
    ytotal=np.concatenate((y0,np.sqrt(r)*y));         
    MU=LA.norm(np.matmul(Xtotal.T,ytotal), np.inf) #mu_max 
    ###########################################################################
    ################## LASSO solution on Original batch (X,y) #################
    mu=m*MU
    thet,nbr=path2.coordinate_descent(X, y, mu)
    ac=np.where(thet!=0)[0];X1=X[:,ac];v1=np.sign(thet[ac])
    nac=np.where(thet==0)[0];X2=X[:,nac]
    A=r*np.matmul(X1.T,X1)
    b=r*np.matmul(X1.T,y)-r*mu*v1
    B=r*np.matmul(X2.T,X1)
    c=r*np.matmul(X2.T,y)    
    
    x1=X0[:,ac];x2=X0[:,nac]
    C=np.matmul(x1.T,x1)
    d=np.matmul(x1.T,y0)
    D=np.matmul(x2.T,x1)
    e=np.matmul(x2.T,y0)    
    ###########################################################################
    ################################ AFLH #####################################
    coeff,ac2,nac2,en,sa=AFLH(X,X0,y,y0,r,mu,v1,A,B,C,D,b,c,d,e,ac,nac)
    f1n=LEF(Xtotal,ytotal,coeff,ac2,mu,dd,1)
    print('Active Variables:',len(ac2),'In:',en,'Out:',sa,'LASSO_error:',f1n)
    ###########################################################################              

