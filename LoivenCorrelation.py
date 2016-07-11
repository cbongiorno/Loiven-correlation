#!/usr/bin/python

import sys, getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
import scipy.stats as st
from matplotlib.pylab import flatten
from multiprocessing import Pool
import random as rd
from copy import deepcopy

def floatingPointError(H):
	H[np.logical_and(H<1e-12,H>-1e-12)]=0
	return H

def RMT(A,(N,M),method='Pos',rem_mode=False):
	'''
	'Input Correlation Matrix, Dimension of TimeSeries (N,M),method'+
	' "PosNeg" take out l{max} and l-<l<+l; "PosNeg_wMod" take out just l-<l<+l;'
	'''

	LM = (1+np.sqrt(N/float(M)))**2
	Lm = (1-np.sqrt(N/float(M)))**2
	
	l,v = np.linalg.eig(A)
	
	v = [v[:,i] for i in range(len(v))]
	
	l,v = zip(*sorted(zip(l,v),reverse=True,key=lambda x:x[0]))

	

	Cm =(np.outer(v[0],v[0])*l[0]).real
	Cr = np.zeros((len(l),len(l)))

	for i in range(len(l)):
		if l[i]<LM:
			S = np.outer(v[i],v[i])*l[i]
			Cr+= S.real
			
	l = np.array(l)	
	if method=='Pos' and rem_mode==False:
		xv = (sum(l[l>=LM])-max(l))/float(N)
		return floatingPointError(A-Cr-Cm),xv
	elif method=='Pos' and rem_mode==True:
		xv = (sum(l[l>=LM]))/float(N)
		return floatingPointError(A -Cr),xv
	elif method=='All' and rem_mode==False:
		return A,1.0
	elif method=='All' and rem_mode==True:
		return floatingPointError(A-Cm),1.0 - l[0]/float(N)
	else:
		print "BUG"
		return None


def My_RMT(X,method='All',rem_mode=False):
	
	if method=='All' and rem_mode==False:
		return np.corrcoef(X)
	
	Xm = X-X.mean(axis=0)

	return np.corrcoef(Xm)


def UpdateSigma(sigma,R):
    return np.array([[p for x in c for p in sigma[x]] for c in R ])

def to_Membership(sigma,N):
    M = np.zeros(N)
    for i,r in enumerate(sigma):
        M[r] = i
    return M
    

def Modulize(B):
	Q = np.diagonal(B).sum()	#Initial Modularity
	N = len(B)
	M = dict(zip(range(N),range(N)))	#The Membership
	
	'Create Membership Matrix'
	C = np.zeros((N,N))
	np.fill_diagonal(C,True)
	
	'Shuffle the priority list of nodes'
	Nx = range(N)
	rd.shuffle(Nx)

	count=0
	k=0
	xcount = 0
	Q0 = Q
	'Do it until no futher improvement are possile'
	while True:
		
		xcount+=1
		if xcount%N==0: 
			if Q>Q0: Q0 = Q	#if it is an improvement change Q -> Q0
			else: break 	#if no improvment exit
		
		
		i = Nx[k]	#The node
		ci = M[i]	#Th membership of the node
		
		'For any neighbors of node-i evaluate the increment of modularity'
		dQ = []
		for j in xrange(N):
			cj = M[j]		#The membership of the j-node 
			if ci==cj: continue
			
			dQ.append((2*B[i,np.where(C[cj])].sum() - 2*B[i,np.where(C[ci])].sum() + 2*B[i,i],j))	#The increment of modularity
			
		if len(dQ)==0: continue	#No possible movement
		dQ,j = max(dQ)	#Select the movement with the maximum modularity
		cj = M[j]		#the destiantion membership
		'If does not provide a sigificant improvement skip it'
		if dQ>1e-12:
			count=0
			C[cj,i] = True
			C[ci,i] = False
			M[i] = cj
			Q+=dQ
		else:
			count+=1
		'If no improvement break (why it is also up?)'
		if count>=N:
			break
		k+=1
		if k>=N: k=0
			 
	return C,Q

#~ 
#~ def Modulize(B):
	#~ Q = np.diagonal(B).sum()
	#~ N = len(B)
	#~ M = dict(zip(range(N),range(N)))
#~ 
	#~ C = np.zeros((N,N))
	#~ np.fill_diagonal(C,True)
	#~ Nx = range(N)
#~ 
	#~ rd.shuffle(Nx)
	#~ count,k=0,0
#~ 
	#~ while True:
		#~ 
		#~ i = Nx[k]
		#~ ci = M[i]
		#~ 
		#~ dQ = []
		#~ for j in xrange(N):
			#~ cj = M[j]
			#~ if ci==cj: continue
			#~ dQ.append((2*B[i,np.where(C[cj])].sum() - 2*B[i,np.where(C[ci])].sum() + 2*B[i,i],j))
			#~ 
		#~ if len(dQ)==0: continue
		#~ dQ,j = max(dQ)
		#~ cj = M[j]
		#~ if dQ>0:
			#~ count=0
			#~ C[cj,i] = True
			#~ C[ci,i] = False
			#~ M[i] = cj
			#~ Q+=dQ
		#~ else:
			#~ count+=1
			#~ 
		#~ if count>=N: break
		#~ 
		#~ k+=1
		#~ 
		#~ if k>=N: k=0
	#~ return C,Q

#~ def Modulize(B):
	#~ Q = np.diagonal(B).sum()
	#~ N = len(B)
	#~ M = dict(zip(range(N),range(N)))
#~ 
	#~ C = np.zeros((N,N))
	#~ np.fill_diagonal(C,True)
	#~ Nx = range(N)
	#~ rd.shuffle(Nx)
	#~ for i in Nx:
		#~ ci = M[i]
#~ 
		#~ dQ = []
		#~ for j in xrange(N):
			#~ cj = M[j]
			#~ if ci==cj: continue
			#~ dQ.append((2*B[i,np.where(C[cj])].sum() - 2*B[i,np.where(C[ci])].sum() + 2*B[i,i],j))
			#~ 
		#~ if len(dQ)==0: continue
		#~ dQ,j = max(dQ)
		#~ cj = M[j]
		#~ if dQ>0:
#~ 
			#~ C[cj,i] = True
			#~ C[ci,i] = False
			#~ M[i] = cj
			#~ Q+=dQ
	#~ return C,Q

def get_comm(C):
    R = defaultdict(list)
    for a,b in zip(*np.where(C)):
        R[a].append(b)
    R = np.array([np.array(R[k]) for k in R])
    return R

def renormlize(B,R):
    return np.array([[sum(B[l,m] for l in R[i] for m in R[j]) for i in xrange(len(R))] for j in xrange(len(R))])

def LoivenMod_Hier(B):
    N = len(B)
    sigma = np.array([[i] for i in xrange(N)])

    Q0 = 0
    Bt = deepcopy(B)
    while True:
        
        C,Q = Modulize(Bt)

        if Q<=Q0: break

        Q0 = Q
        R =get_comm(C)
        sigma = UpdateSigma(sigma,R)  
        Bt = renormlize(Bt,R)

    
    return to_Membership(sigma,N),Q
   
def LoivenMod(B):
	N = len(B)
	sigma = np.array([[i] for i in xrange(N)])
	  
	C,Q = Modulize(B)
	R =get_comm(C)
	sigma = UpdateSigma(sigma,R)

	return to_Membership(sigma,N),Q

def LoivenModM(B,n,ncpu=1,hierarchy=False):
	'Multicall (ncpu process) for loiven'
	if ncpu>1:
		p = Pool(ncpu)
		X = p.map(LoivenMod_Hier,[B]*n)
		p.close()
	else:
		X = map(LoivenMod_Hier,[B]*n)

	return max(X,key=lambda x:x[1])

def ToCorrelation(XR,n=10, ncpu=1,method='Pos',hierarchy=False):
	N,M = XR.shape
	A = np.corrcoef(XR)
	B,var = RMT(A,(N,M),method,rem_mode=False)
	
	H = [LoivenModM(B,n,ncpu)[0].astype(int)]
	V = [[(0,var)]]
	if hierarchy==False:
		return H

	while True:
		xvar = []
		M = H[-1]
		
		mx,h = 0,np.zeros(N)
		
		size = Counter(M)
		
		for c in set(M):
			if size[c]>1:
				Bs, XRs = deepcopy(B),deepcopy(XR)
				XRs = XRs[np.where(M==c)]
				As = np.corrcoef(XRs)
				
				Bs,var = RMT(As,XRs.shape,method,rem_mode=True)
				Ms,q = LoivenModM(Bs,n,ncpu)
				h[np.where(M==c)] = Ms+mx
				xvar.append((c,var))
				mx = h.max()+1
			else:
				h[np.where(M==c)] = mx
				mx+=1

		if len(set(h))==len(set(H[-1])): break
		H.append(h.astype(int))
		V.append(xvar)
		
		if set(h)==N: break
	return H,V


def ToCorrelation_My(XR,n=10, ncpu=1,method='All',hierarchy=False):
	N,M = XR.shape
	#A = np.corrcoef(XR)
	B = My_RMT(XR,method,rem_mode=False)
	
	H = [LoivenModM(B,n,ncpu)[0].astype(int)]
	#V = [[(0,var)]]
	if hierarchy==False:
		return H

	while True:
		#xvar = []
		M = H[-1]
		
		mx,h = 0,np.zeros(N)
		
		size = Counter(M)
		
		for c in set(M):
			if size[c]>1:
				Bs, XRs = deepcopy(B),deepcopy(XR)
				XRs = XRs[np.where(M==c)]
				As = np.corrcoef(XRs)
				
				Ns,Ms = XRs.shape
				
				LM = (1+np.sqrt(Ns/float(Ms)))**2
				l = max(np.linalg.eigvals(As))
				
				Bs = My_RMT(XRs,method,rem_mode=True)
				Ms,q = LoivenModM(Bs,n,ncpu)
					
				h[np.where(M==c)] = Ms+mx
				mx = h.max()+1
			else:
				h[np.where(M==c)] = mx
				mx+=1

		if len(set(h))==len(set(H[-1])): break
		H.append(h.astype(int))
		
		
		if set(h)==N: break
	
	return H


#~ 
#~ def ToCorrelation(XR,n=10, ncpu=1,hierarchy=False):
	#~ 
	#~ 
	#~ 'Va standardizzata?'
	#~ N,M = XR.shape
	#~ #XR = st.zscore(XR,axis=1) ##### occio!
	#~ A = np.corrcoef(XR)
	#~ 
	#~ B = RMT(A,(N,M),'PosNeg_wMod')
	#~ M,Q = LoivenModM(B,n,ncpu,hierarchy)
	#~ Q = Q/A.sum()
	#~ 
	#~ return M,Q

def main(argv):
   inputfile = ''
   outputfile = ''
   n = 10
   ncpu = 1
   hierarchy=False
   
   try:
      opts, args = getopt.getopt(argv,"hi:o:n:c:H",["ifile=","ofile=","nrun=","ncpu=","hier="])
   except getopt.GetoptError:
      print 'LoivenCorrelation.py -i <inputfile> -o <outputfile> --nrun <run> --ncpu <ncpu> --hier <bool>' 
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'LoivenCorrelation.py -i <inputfile> -o <outputfile> --nrun <run> --ncpu <ncpu> --hier <bool>' 
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-n", "--nrun"):
		n = int(arg)
      elif opt in ("-c", "--ncpu"):
         ncpu = int(arg) 
                      
      elif opt in ("-H", "--hier"):
		 hierarchy = arg=='1'
   
   return inputfile,outputfile,n,ncpu,hierarchy

	
if __name__=='__main__':
	
	file_r,file_w,n,ncpu,hierarchy = main(sys.argv[1:])

	XR = np.array(pd.read_table(file_r,header=None))
	M,V = ToCorrelation(XR,n,ncpu,hierarchy)

	OUT = pd.DataFrame(M).transpose()
	OUT.to_csv(file_w,header=False,index=False,sep='\t')
	

	

