#######################################################

###

### Sol Vitkin

### Netflix Recommendation Problem

### Last Modified: 10/8/2015

###

########################################################


#Numpy and Pandas Library
import numpy as np
import pandas as pd
import math


###Importing and Pre-processing Data###

#Training Data
training_data = pd.read_csv\
    ("/home/sol/PycharmProjects/CSC 576/Netflix/Data/training.txt", sep = '\t')

#dim of initial data
#print training_data.shape

#number of unique movies and users,
#taken from maximum of each column in original data
nmovies = max(training_data["Movie"])
nusers = max(training_data["User"])

#empty matrix to be training data matrix
tmatrix = np.zeros((nusers,nmovies))

#for each user and movie index in training data set
#obtain the correspoinding rating and put into
#empty training matrix for the respective indeces
for i in xrange(0,len(training_data)):
    u = training_data.loc[i,"User"]-1
    m = training_data.loc[i,"Movie"]-1
    rating = training_data.loc[i,"Rating"]
    tmatrix[u,m] = rating
#print trmatrix


#use trmatrix for ALS algorithm
trmatrix = np.copy(tmatrix)


#Testing Data
testing_data = pd.read_csv\
    ("/home/sol/PycharmProjects/CSC 576/Netflix/Data/testing.txt", sep = '\t')

#number of unique movies and users,
#taken from maximum of each column in original data
nmovieste = max(testing_data["Movie"])
nuserste = max(testing_data["User"])

#empty matrix to be testing data matrix
tematrix = np.zeros((nuserste,nmovieste))

#for each user and movie index in testing data set
#obtain the correspoinding rating and put into
#empty testing matrix for the respective indeces
for i in xrange(0,len(testing_data)):
    u = testing_data.loc[i,"User"]-1
    m = testing_data.loc[i,"Movie"]-1
    rating = testing_data.loc[i,"Rating"]
    tematrix[u,m] = rating


#Use for ALS later
testmatrix = np.copy(tematrix)

###Helper functions###

#L2 norm function
def l2norm(vector):

    norm = 0
    for i in xrange(0,vector.size):
        v2 = math.pow(vector.item(i),2)
        norm = norm + v2

    return math.sqrt(norm)

#Frobenius Norm Matrix
def fr_norm(mtrx):
    mtrx = mtrx + 0.0
    fmtrx = mtrx.flatten()
    return l2norm(fmtrx)


def objfunc(F,P,R):
    tmp = F - np.dot(P,np.transpose(R))
    frn = fr_norm(tmp)
    return .5*frn


#debug fr_norm
#A = np.matrix([[1,1],[1,1]])
#print fr_norm(A)

def fdiff(B,C,D,E,F):
    diff1 = objfunc(B,C,D)
    diff2 = objfunc(B,E,F)
    return diff1 - diff2



###ALS SETUP AND ALGORITHM###

#initialize H_0 and W_0 and set rank for training
r = 10
H = np.random.uniform(low=1,high=6,size=(nusers,r))
W = np.random.uniform(low=1,high=6,size=(nmovies,r))

#initialize H_0 and W_0 and set rank for testing
r = 10
H_0 = np.random.uniform(low=1,high=6,size=(nuserste,r))
W_0 = np.random.uniform(low=1,high=6,size=(nmovieste,r))

#Reformat original training matrix with average rating
#instead of missing value to make ALS more accurate
#because of matrix multiplication taking place

#Get mean of ratings
def avgm(array):
    sum = 0
    count = 0
    dim = array.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if round(array[i,j]) != 0:
                sum += array[i,j]
                count += 1

    return sum/count


#average rating for training data
trmean = avgm(trmatrix)
#print trmean

#average rating for testing data
temean = avgm(testmatrix)

#Replace missing 0's with matrix average rating for test matrix
d1 = testmatrix.shape
for i in range(d1[0]):
    for j in range(d1[1]):
        if testmatrix[i,j]==0:
            testmatrix[i,j] = temean

#Replace missing 0's with matrix average rating for training matrix
d2 = trmatrix.shape
for i in range(d2[0]):
    for j in range(d2[1]):
        if trmatrix[i,j] == 0:
            trmatrix[i,j] = trmean


def ALS(A,H,W,alpha):
    #H_k is previous H
    #W_k is previous W
    #alpha is user defined cutoff

    diff = alpha + 1

    #print fdiff(A,H_k,W_k,H,W)

    while diff > alpha:
        H_k = np.copy(H)
        W_k = np.copy(W)

        H = np.dot(A, np.transpose(np.linalg.pinv(W_k)))
        W = np.dot(np.transpose(A), np.transpose(np.linalg.pinv(H)))

        diff = fdiff(A,H_k,W_k,H,W)

        #f = (np.dot(H,np.transpose(W)))
        #print f.astype(int)

    return np.dot(H,np.transpose(W))

#predictive training matrix
ALS_tr = ALS(trmatrix, H, W, .001)

#predictive testing matrix
ALS_te = ALS(testmatrix, H_0, W_0, .001)

###RMSE###

#RMSE calculated with numerator as difference in prediction
#calculated if the testing matrix element is not equal to 0
#The denominator is the number of nonzero elements in
#the testing matrix
def RMSE(a1,a2):
    dim2 = a2.shape

    den = 0.0
    for x in np.nditer(a2):
        if x!=0:
            den += 1

    num = 0.0
    for i in range(dim2[0]):
        for j in range(dim2[1]):
            if a2[i,j] != 0:
                dif = (a1[i,j]-a2[i,j])**2
                num += dif

    return math.sqrt(num/den)

#debugging RMSE
#P = np.matrix([[1,3],[1,1]])
#Q = np.matrix([[1,2],[0,0]])
#print RMSE(P,Q)

#training matrix prediction accuracy
rmse2 = RMSE(ALS_tr,tmatrix)
print "The RMSE for the training data is " + str(rmse2)

#Output predicted testing matrix
tedata = pd.DataFrame(ALS_te)
stack = tedata.stack()
stack.to_csv("pred_testingdata.csv", sep = "\t")
