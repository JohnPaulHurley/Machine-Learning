import numpy as np
import csv
import math
def p(test,mu,sigma):
	x = test-mu
	#print (x)
	temp = np.matmul(np.transpose(x),np.linalg.inv(sigma))
	temp = np.matmul(temp,x)
	#try:
	temp = math.exp(-1/2*temp)
	#except OverflowError:
	#	temp = math.exp(-math.copysign(math.inf,temp))
	temp = temp/(2*math.pi)/(2*math.pi)/math.sqrt(np.linalg.det(sigma))
	#print (temp)
	return temp
with open('coen140_ss.csv') as csvfile:
	readCSV = csv.reader(csvfile,delimiter=',')
	sepalL = []
	sepalW = []
	petalL= []
	petalW = []
	for row in readCSV:
		sepL = row[0]
		sepW = row[1]
		petL = row[2]
		petW = row[3]

		sepalL.append(float(sepL))
		sepalW.append(float(sepW))
		petalL.append(float(petL))
		petalW.append(float(petW))

sepLmat = np.matrix(sepalL)
sepWmat = np.matrix(sepalW)
petLmat = np.matrix(petalL)
petWmat = np.matrix(petalW)


setosa = np.concatenate((sepLmat[0,0:50],sepWmat[0,0:50],petLmat[0,0:50],petWmat[0,0:50]))
versic = np.concatenate((sepLmat[0,50:100],sepWmat[0,50:100],petLmat[0,50:100],petWmat[0,50:100]))
nica = np.concatenate((sepLmat[0,100:150],sepWmat[0,100:150],petLmat[0,100:150],petWmat[0,100:150]))

# separate data into test and training
setosaTest = versicTest = nicaTest = setosaTrain = versicTrain = nicaTrain = np.matrix([]);
setosaTrain = setosa[:,0:40]
#print (setosaTrain)
setosaTest = setosa[:,40:50]
versicTrain = versic[:,0:40]
versicTest = versic[:,40:50]
nicaTrain = nica[:,0:40]
nicaTest = nica[:,40:50]
mu1 = np.sum(setosaTrain,axis=1)/40
mu2 = np.sum(versicTrain,axis=1)/40
mu3 = np.sum(nicaTrain, axis=1)/40
sigma1 = sigma2 = sigma3 = np.matrix('0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
for i in range(0,40):
	sigma1= np.add(sigma1, np.matmul((setosaTrain[:,i] - mu1),np.transpose(setosaTrain[:,i] - mu1)))
	sigma2= np.add(sigma2, np.matmul((versicTrain[:,i] - mu2),np.transpose(versicTrain[:,i] - mu2)))
	sigma3= np.add(sigma3, np.matmul((nicaTrain[:,i] - mu3),np.transpose(nicaTrain[:,i] - mu3)))


sigma1 = sigma1/40
sigma2 = sigma2/40
sigma3 = sigma3/40

correct=correctLDA=correctTrain=0

#testing
sigma = np.add(sigma1,sigma2,sigma3)/3

for i in range(0,10):

	# QDA
	p1= p(setosaTest[:,i],mu1,sigma1)
	p2 =p(setosaTest[:,i],mu2,sigma2)
	p3 =p(setosaTest[:,i],mu3,sigma3)
	
	if (p1>p2 and p1>p3):
		correct += 1

	p1 = p(versicTest[:,i],mu1,sigma1)
	p2 = p(versicTest[:,i],mu2,sigma2)
	p3 = p(versicTest[:,i],mu3,sigma3)

	if (p2>p1 and p2>p3):
		correct += 1

	p1 = p(nicaTest[:,i],mu1,sigma1)
	p2 = p(nicaTest[:,i],mu2,sigma2)
	p3 = p(nicaTest[:,i],mu3,sigma3)

	if (p3>p1 and p3>p2):
		correct += 1

	# LDA
	p1 = p(setosaTest[:,i],mu1,sigma)
	p2 =p(setosaTest[:,i],mu2,sigma)
	p3 =p(setosaTest[:,i],mu3,sigma)
	
	if (p1>p2 and p1>p3):
		correctLDA += 1

	p1 = p(versicTest[:,i],mu1,sigma)
	p2 = p(versicTest[:,i],mu2,sigma)
	p3 = p(versicTest[:,i],mu3,sigma)

	if (p2>p1 and p2>p3):
		correctLDA += 1

	p1 = p(nicaTest[:,i],mu1,sigma)
	p2 = p(nicaTest[:,i],mu2,sigma)
	p3 = p(nicaTest[:,i],mu3,sigma)

	if (p3>p1 and p3>p2):
		correctLDA += 1
for i in range(0,40):
	p1 = p(nicaTrain[:,i],mu1,sigma1)
	p2 = p(nicaTrain[:,i],mu2,sigma2)
	p3 = p(nicaTrain[:,i],mu3,sigma3)

	if( p1>p2 and p1>p3):
		correctTrain += 1

	p1 = p(versicTrain[:,i],mu1,sigma1)
	p2 = p(versicTrain[:,i],mu2,sigma2)
	p3 = p(versicTrain[:,i],mu3,sigma3)
	
	if( p2>p1 and p2>p3):
		correctTrain +=1

	p1 = p(setosaTrain[:,i],mu1,sigma1)
	p2 =p(setosaTrain[:,i],mu2,sigma2)
	p3 =p(setosaTrain[:,i],mu3,sigma3)
	
	if( p3>p1 and p3>p2):
		correctTrain +=1	
print (correct)
print (correctLDA)
print (1-correctTrain/120)
