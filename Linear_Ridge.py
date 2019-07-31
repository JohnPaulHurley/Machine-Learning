import numpy as np
import csv
import math
I = np.zeros((96,96))
for i in range(0,96):
	I[i,i] = 1

def eval(x,y,f,v,lambd):
	se = 0
	wrr = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x) + I*lambd),np.transpose(x)),y)
	for i in range(0,319):
		ytest = np.matmul(np.transpose(wrr),f[i,:])
		se += (v[i]-ytest)*(v[i]-ytest)
	return se
def crossval(x,y,f,v,lambd,w0):
	wt=w0
	wprev=np.zeros(96)
	l1 = 0.0
	while (True):
		wprev = wt
			
		grad = np.matmul(np.transpose(x),(y-np.matmul(x,wprev)))-lambd*wprev
		
		wt = wprev + grad*10e-6
		l2 = np.matmul(np.transpose(np.matmul(x,wprev) - y),(np.matmul(x,wprev) - y))+lambd*np.matmul(np.transpose(wprev),wprev)
	
		l1 = l2 
		l2 = np.matmul(np.transpose(np.matmul(x,wt) - y),(np.matmul(x,wt) - y))+lambd*np.matmul(np.transpose(wt),wt)
		diff = l2 - l1
		if (math.fabs(diff) < 10e-5):
			break
	se = 0
	for i in range(0,319):
		ytest = np.matmul(np.transpose(wt),f[i,:])
		se += (v[i]-ytest)*(v[i]-ytest)


	return se

with open('crime.txt') as csvfile:
	readCSV = csv.reader(csvfile,delimiter='\t')
	y = np.matrix([])
	data=np.zeros((1595,96))
	x = np.zeros((1595,96))
	w = np.matrix([])
	i = flag = 0
	for row in readCSV:
		if (flag == 0):
			flag += 1
			continue
		data[i] = (np.matrix(row))
		i += 1	
	y = data[:,0]
	x[:,0:95] = data[:,1:96]

	x[:,95] = 1
	
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)
	#print (w)
	with open('crimetest.txt') as csvfile2:
		readCSV2 = csv.reader(csvfile,delimiter='\t')
		ytest = np.matrix([])
		datatest=np.zeros((1595,96))
		xtest = np.zeros((1595,96))
		i = flag = 0
		for row in readCSV2:
			if (flag == 0):
				flag += 1
				continue
			datatest[i] = (np.matrix(row))
			i += 1	
		ytrue = datatest[:,0]
		xtest[:,0:95] = datatest[:,0:95]
		xtest[:,95] = 1 
		se = 0
		for i in range(0,399):
			ytest = np.matmul(np.transpose(w),xtest[i,:])
			se += (ytest-ytrue[i])*(ytest-ytrue[i])
		se /= 399
		se = math.sqrt(se)
		print ("LR test rmse =",se)	
		se =0
		for i in range(0,1595):
			ytrain = np.matmul(np.transpose(w),x[i,:])
			se += (ytrain-y[i])*(ytrain-y[i])

		se = math.sqrt(se/1595)
		print ("LR training rmse =",se)

# RR
		f1 = x[0:319,:]
		f2 = x[319:638,:]
		f3 = x[638:957,:]
		f4 = x[957:1276,:]
		f5 = x[1276:1595,:]
		t1 = np.concatenate((f1,f2,f3,f4))
		t2 = np.concatenate((f1,f2,f3,f5))
		t3 = np.concatenate((f1,f2,f4,f5))
		t4 = np.concatenate((f1,f3,f4,f5))
		t5 = np.concatenate((f2,f3,f4,f5))
		y1 = y[0:319]
		y2 = y[319:638]
		y3 = y[638:957]
		y4 = y[957:1276]
		y5 = y[1276:1595]
		yt1 = np.concatenate((y1,y2,y3,y4))
		yt2 = np.concatenate((y1,y2,y3,y5))
		yt3 = np.concatenate((y1,y2,y4,y5))
		yt4 = np.concatenate((y1,y3,y4,y5))
		yt5 = np.concatenate((y2,y3,y4,y5))

		lambd = 400.0
		minerror = math.inf
		minlambda = math.inf
		while (lambd > 0.781):
			se = eval(t1,yt1,f5,y5,lambd)
			se += eval(t2,yt2,f4,y4,lambd)
			se += eval(t3,yt3,f3,y3,lambd)
			se += eval(t4,yt4,f2,y2,lambd)
			se += eval(t5,yt5,f1,y1,lambd)
			se = math.sqrt(se/319)/5
			if(se<minerror):
				minerror = se
				minlambda = lambd

			lambd /= 2
		wrr = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x) + I*minlambda),np.transpose(x)),y)
		se = 0;
		for i in range(0,399):
			ytest = np.matmul(np.transpose(wrr),xtest[i,:])
			se += (ytrue[i]-ytest)*(ytrue[i]-ytest)

		rmse = math.sqrt(se/399)
		print ("RR test rmse =",rmse)

# GD LR
		w0 = np.random.standard_normal(96)
		wt=w0
		wprev=np.zeros(96)
		count = 0
		while(count!=96):
			count = 0;
			wprev = wt
			grad = np.matmul(np.transpose(x),(y-np.matmul(x,wprev)))
			
			wt = wprev + grad*10e-6
			
			diff = wt-wprev
			for i in range(0,96):
				if (math.fabs(diff[i]) < 10e-5):
					count += 1
		se = 0
		for i in range(0,399):
			ytest = np.matmul(np.transpose(wt),xtest[i,:])
			se += (ytest-ytrue[i])*(ytest-ytrue[i])
		se /= 399
		se = math.sqrt(se)
		print ("GDLR test rmse =",se)	
		se =0
		for i in range(0,1595):
			ytrain = np.matmul(np.transpose(wt),x[i,:])
			se += (ytrain-y[i])*(ytrain-y[i])

		se = math.sqrt(se/1595)
		print ("GDLR training rmse =",se)
		print ("Please wait... calculating GDRR") 
# GD RR
		minlambda = math.inf
		minerror = math.inf
		lambd = 400.0
		while (lambd > 0.781):
			se = math.sqrt(crossval(t1,yt1,f5,y5,lambd,w0))
			se += math.sqrt(crossval(t2,yt2,f4,y4,lambd,w0))
			se += math.sqrt(crossval(t3,yt3,f3,y3,lambd,w0))
			se += math.sqrt(crossval(t4,yt4,f2,y2,lambd,w0))
			se += math.sqrt(crossval(t5,yt5,f1,y1,lambd,w0))
			se /= math.sqrt(319)
			se /= 5
			if(se<minerror):
				minerror = se
				minlambda = lambd
			
			lambd /= 2
		wt = w0
		l1 = 0.0
		while(True):
			wprev = wt
				
			grad = np.matmul(np.transpose(x),(y-np.matmul(x,wprev)))-minlambda*wprev
			
			wt = wprev + grad*10e-6
			l2 = np.matmul(np.transpose(np.matmul(x,wprev) - y),(np.matmul(x,wprev) - y))+lambd*np.matmul(np.transpose(wprev),wprev)
	
			l1 = l2 
			l2 = np.matmul(np.transpose(np.matmul(x,wt) - y),(np.matmul(x,wt) - y))+lambd*np.matmul(np.transpose(wt),wt)
			diff = l2 - l1

			if (math.fabs(diff) < 10e-5):
				break
			

		se = 0
		for i in range(0,399):
			ytest = np.matmul(np.transpose(wt),xtest[i,:])
			se += (ytest-ytrue[i])*(ytest-ytrue[i])
		se /= 399
		se = math.sqrt(se)
		print ("GDRR test rmse =",se)	
	
