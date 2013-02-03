""" IMPORT STATEMENTS """
import scipy.linalg as scilinalg
import scipy.io
import sys
import numpy as np
from numpy import linalg
from matplotlib import pyplot
import matplotlib.cm
import Image
from pylab import *

#give path to features here
def getFeatures(path = 'e:\\EE5907R\\project1\\docs\\features.mat'):
	matfile = scipy.io.loadmat(path)
	features = matfile['fea']
	labels = matfile['labels']
	features = features.astype(np.float32)
	print "Primitive Features : ", features.shape
	fix_features = np.matrix(np.zeros(features.shape))
	fix_features.astype(np.float32)
	idx = 0;
	for feature in features[:]:
		fmat = feature.reshape(50,40,order='F')
		fix_features[idx] = fmat.reshape(-1)
		#showFeature(fix_features[idx], True, str(idx+1)+".png")
		idx += 1
	fix_features = np.transpose(fix_features)
	return fix_features,labels

#give math to splits directory here
def loadSplitN(n):
    splitn = scipy.io.loadmat('E:\\EE5907R\\project1\\docs\\splits\\'+str(n)+'.mat')
    train = splitn['trainIdx']
    test = splitn['testIdx']
    return train,test

def float2int8(A):
	# function im = float2int8(A)
	# convert an float array image into grayscale image with contrast from 0-255
	amin = np.amin(A)
	amax = np.amax(A)
	im = ((A - amin) / (amax - amin)) * 255
	im = np.trunc(im)
	return im.astype(np.int8)

#PCA function
def myPCA(A):
	# function [W,LL,m]=mypca(A)
	# computes PCA of matrix A
	# A: D by N data matrix. Each column is a random vector
	# W: D by K matrix whose columns are the principal components in decreasing order
	# LL: eigenvalues
	# m: mean of columns of A
	# Note: "lambda" is a Python reserved word
	# compute mean, and subtract mean from every column
	[r,c] = A.shape
	m = np.mean(A,1)
	mmat = np.tile(m, (1,c))
	print "MMAT SHAPE ",mmat.shape
	A = A - mmat
	B = np.dot(np.transpose(A), A)
	[d,v] = linalg.eig(B)
	# v is in descending sorted order
	# compute eigenvectors of scatter matrix
	W = np.dot(A,v)
	Wnorm = ComputeNorm(W)

	W1 = np.tile(Wnorm, (r, 1))
	W2 = W / W1
	LL = d[0:-1]
	W = W2[:,0:-1]      #omit last column, which is the nullspace
	return W, LL, m

def LDA(Sw,Sb):
	[W,m] = scilinalg.eig(Sw,Sb)
	return W,m

def projectImagesOntoLDA(gestures,lda):
	Y = np.dot(lda.T,gestures)
	return Y

def projectImagesOntoPCA(gestures,pca,mean,K):
	W = pca[:,0:K]
	[r,c] = gestures.shape
	gestures = gestures - np.tile(mean,(1,c))
	Y = np.dot(W.T,gestures)
	return Y

def getMeanPerClass(gestures, RANGE = 10):
	means = []
	for x in range(RANGE):
		features = gestures[:,x : x + 10]
		means.append(features.mean(axis = 1))
	return means

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

# compute accuracy
def accuracy(A):
	print "Trace : ", np.trace(A), " Sum : ",  np.sum(A.reshape(-1))
	return np.trace(A)/(0.0+np.sum(A.reshape(-1)))

def getTrainingSet(trainSet, features):
	print "Creating Training Set"
	[fr,fc] = features.shape
	""" Initialize training set matrix to contain 10 ft vectors from each of 10 classes """
	""" to access, use tset[class][i,:] """
	tset = []
	mset = []
	c = 0
	idx = 0
	tfeatures = np.matrix(np.zeros([fr,10]))
	alltogether = np.matrix(np.zeros([fr,len(trainSet[0])]))
	for x in trainSet[0,:]:
		#print " X : ", x
		tfeatures[:,idx%10] = features[:,x-1]
		alltogether[:,idx] = features[:,x-1]
		idx += 1
		if idx/((c + 1)*10) >= 1:
			tset.append(tfeatures)
			mset.append(tfeatures.mean(axis=1))
			tfeatures = np.matrix(np.zeros([fr,10]))
			c += 1
	return tset,mset,alltogether

#performs 16-connected and 8-connected approximation
# of covariance matrix
def getMaskVector(i,LEVELS = 2,ROWS = 50, COLUMNS = 40):
	c = ((i + 1) % COLUMNS)
	r = ceil((i + 1) / COLUMNS)
	if c == 0:
		c = COLUMNS
	r -= 1
	c -= 1
	#print "Mask : ", r, c
	maskMat = np.matrix(np.zeros([ROWS,COLUMNS]))
	maskMat[r,c] = 1

	if r > 1 and LEVELS > 1:
		maskMat[r-2,c] = 1
		if c != 0:
			maskMat[r-2,c-1] = 1
		if c > 1:
			maskMat[r-2,c-2] = 1
		if c != COLUMNS-1:
			maskMat[r-2,c+1] = 1
		if c < COLUMNS-2:
			maskMat[r-2,c+2] = 1

	if r != 0:
		maskMat[r-1,c] = 1
		if c != 0:
			maskMat[r-1,c-1] = 1
		if c > 1 and LEVELS > 1:
			maskMat[r-1,c-2] = 1
		if c != COLUMNS-1:
			maskMat[r-1,c+1] = 1
		if c < COLUMNS-2 and LEVELS > 1:
			maskMat[r-1,c+2] = 1

	if c != 0:
		maskMat[r,c-1] = 1
	if c != COLUMNS-1:
		maskMat[r,c+1] = 1
	if c > 1 and LEVELS > 1:
		maskMat[r,c-2] = 1
	if c < COLUMNS-2 and LEVELS > 1:
		maskMat[r,c+2] = 1

	if r != ROWS-1:
		maskMat[r+1,c] = 1
		if c != 0:
			maskMat[r+1,c-1] = 1
		if c > 1 and LEVELS > 1:
			maskMat[r+1,c-2] = 1
		if c != COLUMNS-1:
			maskMat[r+1,c+1] = 1
		if c < COLUMNS-2 and LEVELS > 1:
			maskMat[r+1,c+2] = 1

	if r < ROWS-2 and LEVELS > 1:
		maskMat[r+2,c] = 1
		if c != 0:
			maskMat[r+2,c-1] = 1
		if c > 1:
			maskMat[r+2,c-2] = 1
		if c != COLUMNS-1:
			maskMat[r+2,c+1] = 1
		if c < COLUMNS-2:
			maskMat[r+2,c+2] = 1

	maskVector = maskMat.reshape(-1)
	#print maskMat
	#wait = raw_input("Press enter..")
	return maskVector

def approxToDiagonal(A):
	[r,c] = A.shape
	i = 0
	mask_mat = np.identity(r)
	while i < r:
		mask_vector = mask_mat[i]
		dotp = np.array(A[i,:]) * np.array(mask_vector)
		A[i,:] = dotp
		i += 1
	return A

def getCovarianceMatrix(A,mask = False,LEVELS = 2,approxAlone = False):
	[r,c] = A.shape
	S = np.cov(A, rowvar = 1, bias = 1)
	print "Covar Shape : ", S.shape
	if mask or approxAlone:
		i = 0
		while i < r:
			if mask:
				rowi = S[i,:]
				mask_vector = np.array(getMaskVector(i, LEVELS))
				dotp = np.array(rowi) * mask_vector
				S[i,:] = dotp
			if S[i,i] < 0.7:
				S[i,i] = 0.7
			i += 1
	return S

def getTestSet(testSet, features):
	print "Creating Test Set"
	[fr,fc] = features.shape
	
	""" Initialize test set matrix to contain 2 ft vectors from each of 10 classes """
	idx = 0
	tfeatures = np.matrix(np.zeros([fr,20]))
	for x in testSet[0,:]:
		tfeatures[:,idx] = features[:,x-1]
		idx += 1
	#print tfeatures
	return tfeatures

def getImageFromArray(arr):
	print arr.shape
	mat = arr.reshape([50,40]).copy()
	mat = float2int8(mat)
	image = Image.fromarray(mat,"L")
	return image

#shows a particular image
def showFeature(arr, save = False, path = ""):
	image = getImageFromArray(arr)
	if save:
		image.save("E:\\EE5907R\\train\\" + path)
	else:
		image.show()

#for reconstructing Image from PCA space
def reconstruct(Y, pca, mean):
	figure = pyplot.figure()

    # adding gestures to figure
	for i in range(10):
		print i
		y = Y[:,10 * i]
		x = np.dot(pca,y) + mean
		image = getImageFromArray(x)
		image = image.rotate(180)
		pic = figure.add_subplot(5,2,i+1)
		pic.xaxis.set_visible(False)
		pic.yaxis.set_visible(False)
		pic.set_title("Recon Hand : " + str(i+1))
		pic.imshow(image, matplotlib.cm.get_cmap('gist_gray'))
	figure.show()
	return

def bayesianClassifyCase4(test_set, means, Sc_inv, Sc_det):
	[r,c] = test_set.shape
	idx = 0
	ldiscval = np.matrix(np.zeros([10,c]))
	maxindices = np.array(np.zeros([c]))
	for i in range(10):
		mi = means[i]
		for idx in range(c):
			x = test_set[:,idx]
			ldiscval[i,idx] = linear_disc_case_4(x,Sc_inv[i],mi,Sc_det[i])
	for i in range(c):
		w = np.array(ldiscval[:,i])
		wi = np.argmax(w) + 1
		maxindices[i] = wi
	return ldiscval, maxindices

def linear_disc_case_4(x,Sinv, mean, det):
	x_minus_mu = x - mean
	gi = (-0.5) * (np.dot(x_minus_mu.T,np.dot(Sinv,x_minus_mu)) + math.log1p(math.fabs(det)))
	return gi

def getMinEigenValues(eigenVal):
	eigsum = np.sum(eigenVal)
	k = 0
	for x in range(len(eigenVal)):
		buildsum = np.sum(eigenVal[0:x])
		if buildsum/eigsum >= 0.95:
			k = x + 1
			break;
	return k

#KNN Classification method
def KNNClassify(test_set, train_set, train_idx, labels, K,CLASSES = 10):
	# KNN Method
	# for each test sample t
	#	for each training sample tr
	#		compute norm |t - tr|
	#	choose top K norms
	#	class which dominates is classification
	[tr,tc] = test_set.shape
	[trr,trc] = train_set.shape
	classes = np.array(np.zeros([CLASSES]))
	result = np.array(np.zeros([tc]))
	i = 0
	#print "KNN : with K = ",K
	while i < tc:
		classes *= 0
		x = test_set[:,i]
		xmat = np.tile(x,(1,trc))
		xmat = xmat - train_set
		norms = ComputeNorm(xmat)
		args = np.argsort(norms)
		kargs = args[0,0:K]
		for c in kargs[:]:
			which_train = train_idx[0,c]
			c = labels[which_train - 1]
			#print "C : ",c
			classes[c - 1] += 1
		result[i] = np.argmax(classes) + 1
		#print "Classes : ", classes
		#print "Class : ",result[i]
		i += 1
	return result

#print classification on screen
# also returns conf matrix and accuracy
def printClassification(result, values, test_idx, labels, print_stat = True):
	if print_stat:
		print "Classification : "
		misclas = 0
		for i in range(len(test_idx[0])):
			print "Test : ", i + 1," Classify : ", result[i], " Actual : ", labels[test_idx[0,i] - 1]
			if result[i] != labels[test_idx[0,i] - 1]:
				misclas += 1
		print "Misclassifications : ", misclas
	[conf, accu] = getConfusionMatrixAndAccuracy(result, test_idx, labels)
	if print_stat:
		print "Accuracy : ",accu
	return conf, accu

# Given classification, actual indices and labels
# return confusion matrix and accuracy
def getConfusionMatrixAndAccuracy(result, actual, labels):
	conf = np.matrix(np.zeros([10,10]))
	for i in range(len(result)):
		ai = labels[actual[0,i] - 1]
		conf[ai[0]-1,result[i]-1] += 1
	return conf, accuracy(conf)

# Trains and tests classifiers with optimal parameters as determined from the various plots'
# Q1, Q2.1, Q2.2, Q3.1, Q3.2
def train_classify(question, split, features, labels, method = 1):
	[train_idx,test_idx] = loadSplitN(split)
	print "------------------------------"
	print "SPLIT : ",split
	print "Train : ",train_idx
	print "Test : ",test_idx

	[train_set,mean_set,whole_train_set] = getTrainingSet(train_idx,features)
	print "Training Set Shape : ",whole_train_set.shape
	test_set = getTestSet(test_idx,features)
	print "Test Set Shape : ",test_set.shape

	if question == 1:
		Sc_inv = []
		Sc_det = np.array(np.zeros(10))
		for x in range(10):
			#print " ---------------------------"
			print "Class : ", x+1
			A = getCovarianceMatrix(train_set[x],True,2)
			detA = linalg.det(A)
			if detA == 0:
				A = getCovarianceMatrix(train_set[x],True,1)
				detA = linalg.det(A)
				if detA == 0:
					A = approxToDiagonal(A)
					detA = linalg.det(A)

			print "Covar diagonals : ", np.diag(A)
			print "Covar diagonals - MAX : ", np.max(np.diag(A)), " - MIN : ", np.min(np.diag(A))
			print "Covar Det : ", detA
			Ainv = linalg.inv(A)
			Sc_inv.append(Ainv)
			Sc_det[x] = detA
			print " ---------------------------"
		[vals,indi] = bayesianClassifyCase4(test_set, mean_set, Sc_inv, Sc_det)
		return printClassification(indi, vals, test_idx,labels)
	elif question == 2 and method == 1:
		K = 5 #5NN best for R^2000 feature space
		indi = KNNClassify(test_set,whole_train_set,train_idx,labels,K)
		return printClassification(indi,'null',test_idx,labels)
	elif question == 2 and method == 2:
		[pca, eigenVal, mean] = myPCA(whole_train_set)
		print "PCA Shape = ", pca.shape
		print "Num Eigen Vals = ",len(eigenVal)
		print "Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)

		PCS = 20
		print "Principal Components : ", PCS
		Y = projectImagesOntoPCA(whole_train_set,pca,mean,PCS)
		T = projectImagesOntoPCA(test_set,pca,mean,PCS)

		K = 1
		indi = KNNClassify(T,Y,train_idx,labels,K)
		return printClassification(indi,'null',test_idx,labels)
	elif question == 3 and method == 1:
		[pca, eigenVal, mean] = myPCA(whole_train_set)
		print "PCA Shape = ", pca.shape
		print "Num Eigen Vals = ",len(eigenVal)
		print "Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)

		PCS = len(eigenVal) 
		print "Principal Components : ", PCS
		Y = projectImagesOntoPCA(whole_train_set,pca,mean,PCS)
		T = projectImagesOntoPCA(test_set,pca,mean,PCS)

		#LDA Code
		Sw = np.matrix(np.zeros([PCS,PCS]))
		class_means_list = getMeanPerClass(Y)
		for x in range(10):
			cov = getCovarianceMatrix(Y[:,x * 10 : (x + 1) * 10 ],False,2,True) / 10
			Sw += cov

		class_means_mat = np.matrix(np.zeros([PCS,10]));
		for c in range(10):
			class_means_mat[:,c] = class_means_list[c]

		Sb = getCovarianceMatrix(class_means_mat,False,2,True)

		print "LDA : Sw Shape : ", Sw.shape
		print "LDA : Sb Shape : ", Sb.shape

		[geig_vals,geig_vectors] = LDA(Sb, Sw)
		print "LDA : eigen values : ", len(geig_vals)
		print "LDA : eigen vectors : ", geig_vectors.shape

		LD = 11
		#number of LDA features to use
		geig_vectors = geig_vectors[:,0:LD]
		geig_vals = geig_vals[0:LD]
		geig_norm = ComputeNorm(geig_vectors);

		print "LDA : eigen values : ", geig_vals.shape
		print "LDA : eigen vectors : ", geig_vectors.shape
		print "LDA : eigen norms : ",geig_norm.shape

		for geig in range(LD):
			geig_vectors[:,geig] = geig_vectors[:,geig] / geig_norm[0,geig];
		YL = projectImagesOntoLDA(Y, geig_vectors)
		TL = projectImagesOntoLDA(T, geig_vectors)
		#LDA ends here

		KNN = 1
		indi = KNNClassify(TL,YL,train_idx,labels,KNN)
		return printClassification(indi,'null',test_idx,labels)
	elif question == 3 and method == 2:
		[pca, eigenVal, mean] = myPCA(whole_train_set)
		print "PCA Shape = ", pca.shape
		print "Num Eigen Vals = ",len(eigenVal)
		print "Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)
	
		PCS = 20
		print "Principal Components : ", PCS
		Y = projectImagesOntoPCA(whole_train_set,pca,mean,PCS)
		T = projectImagesOntoPCA(test_set,pca,mean,PCS)
	
		K1 = 1 #1NN proved to be best for PCA,KNN
		indi = KNNClassify(T,Y,train_idx,labels,K1)
		printClassification(indi, 'null', test_idx, labels, True)
		#priliminary results obtained
	
		#--------------------
		# CLASSIFY G, H again
		#--------------------
	
		gh_count = 0
		gh_indices = []
		for ind in range(len(indi)):
			if indi[ind] == 5 or indi[ind] == 6:
				gh_count += 1
				gh_indices.append(ind)
		gh_test_matrix = np.matrix(np.zeros([2000,gh_count]))
	
		i = 0
		for index in gh_indices:
			gh_test_matrix[:,i] = test_set[:,index]
			i += 1
		
		gh_train_matrix = whole_train_set[:,4*10 : 6*10]
	
		[pca, eigenVal, mean] = myPCA(gh_train_matrix)
		print "GH PCA Shape = ", pca.shape
		print "GH Num Eigen Vals = ",len(eigenVal)
		print "GH Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)
	
		PCS = len(eigenVal)
		print "GH Principal Components : ", PCS
	
		Y = projectImagesOntoPCA(gh_train_matrix,pca,mean,PCS)
		T = projectImagesOntoPCA(gh_test_matrix,pca,mean,PCS)
	
		#LDA Code
		Sw = np.matrix(np.zeros([PCS,PCS]))
		class_means_list = getMeanPerClass(Y,2)
		for x in range(2):
			cov = getCovarianceMatrix(Y[:,x * 10 : (x + 1) * 10 ],False,2,True) / 2
			Sw += cov
	
		class_means_mat = np.matrix(np.zeros([PCS,2]));
		for c in range(2):
			class_means_mat[:,c] = class_means_list[c]
	
		Sb = getCovarianceMatrix(class_means_mat,False,2,True)
	
		print "LDA : Sw Shape : ", Sw.shape
		print "LDA : Sb Shape : ", Sb.shape
	
		[geig_vals,geig_vectors] = LDA(Sb, Sw)
		print "LDA : eigen values : ", len(geig_vals)
		print "LDA : eigen vectors : ", geig_vectors.shape
	
	
		geig_norm = ComputeNorm(geig_vectors);
	
		KNN = 1
		LD = 1
		#number of LDA vectors to use
		geig_vectors_sm = geig_vectors[:,0:LD]
				
		YL = projectImagesOntoLDA(Y, geig_vectors_sm)
		TL = projectImagesOntoLDA(T, geig_vectors_sm)
		#LDA ends here
		print "Train indices : ", train_idx[0,4*10:6*10]
		print "Labels : ",labels[4*10:6*10]
		indi1 = KNNClassify(TL,YL,np.matrix(train_idx[0,4*10:6*10]),labels,KNN)
		k = 0
		for index in gh_indices:
			indi[index] = indi1[k]
			k += 1
		[conf,accu] = printClassification(indi,'null',test_idx,labels,True)
		print "Q3 2Level Accuracy : ",accu*100,"\nConfusion matrix\n",conf
		return conf,accu
	return

#Run classification over various parameters
# Fpr Q2.2
def Q2Dot2(split, features, labels, par_knn, par_pcs):
	[train_idx,test_idx] = loadSplitN(split)
	print "------------------------------"
	print "SPLIT : ",split
	print "Train : ",train_idx
	print "Test : ",test_idx

	[train_set,mean_set,whole_train_set] = getTrainingSet(train_idx,features)
	print "Training Set Shape : ",whole_train_set.shape
	test_set = getTestSet(test_idx,features)
	print "Test Set Shape : ",test_set.shape
	#print len(mean_set)
	#print len(train_set)
	#print whole_train_set.shape
	#showFeature(test_set[:,0])
	result_matrix = np.matrix(np.zeros([len(par_knn),len(par_pcs)]))

	[pca, eigenVal, mean] = myPCA(whole_train_set)
	[row,col] = pca.shape
	print "PCA Shape = ", pca.shape
	print "Num Eigen Vals = ",len(eigenVal)
	print "Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)

	for i in range(len(par_knn)):
		par = par_knn[i]
		for j in range(len(par_pcs)):
			PCS = par_pcs[j]
			Y = projectImagesOntoPCA(whole_train_set,pca,mean,PCS)
			T = projectImagesOntoPCA(test_set,pca,mean,PCS)
			#		K = 1
			K = par
			indi = KNNClassify(T,Y,train_idx,labels,K)
			[conf,accu] = printClassification(indi,'null',test_idx,labels,False)
			result_matrix[i,j] = accu*100
	return result_matrix

#Run classification over various parameters
# Fpr Q2.1
def Q2Dot1(split, features, labels, par_knn):
	[train_idx,test_idx] = loadSplitN(split)
	print "------------------------------"
	print "SPLIT : ",split
	print "Train : ",train_idx
	print "Test : ",test_idx

	[train_set,mean_set,whole_train_set] = getTrainingSet(train_idx,features)
	print "Training Set Shape : ",whole_train_set.shape
	test_set = getTestSet(test_idx,features)
	print "Test Set Shape : ",test_set.shape

	result_matrix = np.array(np.zeros([len(par_knn)]))
	for i in range(len(par_knn)):
		K = par_knn[i]
		indi = KNNClassify(test_set,whole_train_set,train_idx,labels,K)
		[conf,accu] = printClassification(indi,'null',test_idx,labels)
		result_matrix[i] = accu*100
	return result_matrix

#Run classification over various parameters
# Fpr Q3.1
def Q3Dot1(split,features,labels,par_knn,par_lda):
	[train_idx,test_idx] = loadSplitN(split)
	print "------------------------------"
	print "SPLIT : ",split
	print "Train : ",train_idx
	print "Test : ",test_idx

	[train_set,mean_set,whole_train_set] = getTrainingSet(train_idx,features)
	print "Training Set Shape : ",whole_train_set.shape
	test_set = getTestSet(test_idx,features)
	print "Test Set Shape : ",test_set.shape

	[pca, eigenVal, mean] = myPCA(whole_train_set)
	[row,col] = pca.shape
	print "PCA Shape = ", pca.shape
	print "Num Eigen Vals = ",len(eigenVal)
	print "Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)

	PCS = len(eigenVal) 

	print "Principal Components : ", PCS
	Y = projectImagesOntoPCA(whole_train_set,pca,mean,PCS)
	T = projectImagesOntoPCA(test_set,pca,mean,PCS)

	#LDA Code
	Sw = np.matrix(np.zeros([PCS,PCS]))
	class_means_list = getMeanPerClass(Y)
	for x in range(10):
		cov = getCovarianceMatrix(Y[:,x * 10 : (x + 1) * 10 ],False,2,True) / 10
		Sw += cov

	class_means_mat = np.matrix(np.zeros([PCS,10]));
	for c in range(10):
		class_means_mat[:,c] = class_means_list[c]

	Sb = getCovarianceMatrix(class_means_mat,False,2,True)

	print "LDA : Sw Shape : ", Sw.shape
	print "LDA : Sb Shape : ", Sb.shape

	[geig_vals,geig_vectors] = LDA(Sb, Sw)
	print "LDA : eigen values : ", len(geig_vals)
	print "LDA : eigen vectors : ", geig_vectors.shape

	result_matrix = np.matrix(np.zeros([len(par_knn),len(par_lda)]))

	geig_norm = ComputeNorm(geig_vectors);
	for geig in range(len(geig_vals)):
		geig_vectors[:,geig] = geig_vectors[:,geig] / geig_norm[0,geig];

	for i in range(len(par_knn)):
		KNN = par_knn[i]
		for j in range(len(par_lda)):
			LD = par_lda[j]
			#number of LDA vectors to use
			geig_vectors_sm = geig_vectors[:,0:LD]
			
			YL = projectImagesOntoLDA(Y, geig_vectors_sm)
			TL = projectImagesOntoLDA(T, geig_vectors_sm)
			indi = KNNClassify(TL,YL,train_idx,labels,KNN)
			[conf,accu] = printClassification(indi,'null',test_idx,labels,False)
			result_matrix[i,j] = accu*100
	return result_matrix

#Run classification over various parameters
# Fpr Q3.2
def Q3Dot2(split,features,labels,par_knn,par_lda):
	[train_idx,test_idx] = loadSplitN(split)
	print "------------------------------"
	print "SPLIT : ",split
	print "Train : ",train_idx
	print "Test : ",test_idx

	[train_set,mean_set,whole_train_set] = getTrainingSet(train_idx,features)
	print "Training Set Shape : ",whole_train_set.shape
	test_set = getTestSet(test_idx,features)
	print "Test Set Shape : ",test_set.shape

	[pca, eigenVal, mean] = myPCA(whole_train_set)

	print "PCA Shape = ", pca.shape
	print "Num Eigen Vals = ",len(eigenVal)
	print "Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)

	PCS = 20
	print "Principal Components : ", PCS
	Y = projectImagesOntoPCA(whole_train_set,pca,mean,PCS)
	T = projectImagesOntoPCA(test_set,pca,mean,PCS)

	K1 = 1 #1NN proved to be best for PCA,KNN
	indi = KNNClassify(T,Y,train_idx,labels,K1)
	printClassification(indi, 'null', test_idx, labels, True)
	#priliminary results obtained

	#--------------------
	# CLASSIFY G, H again
	#--------------------

	gh_count = 0
	gh_indices = []
	for ind in range(len(indi)):
		if indi[ind] == 5 or indi[ind] == 6:
			gh_count += 1
			gh_indices.append(ind)
	gh_test_matrix = np.matrix(np.zeros([2000,gh_count]))

	i = 0
	for index in gh_indices:
		gh_test_matrix[:,i] = test_set[:,index]
		i += 1
	
	gh_train_matrix = whole_train_set[:,4*10 : 6*10]

	[pca, eigenVal, mean] = myPCA(gh_train_matrix)
	print "GH PCA Shape = ", pca.shape
	print "GH Num Eigen Vals = ",len(eigenVal)
	print "GH Capture 95% variance whole_train_set : K = ",getMinEigenValues(eigenVal)

	PCS = len(eigenVal)
	print "GH Principal Components : ", PCS

	Y = projectImagesOntoPCA(gh_train_matrix,pca,mean,PCS)
	T = projectImagesOntoPCA(gh_test_matrix,pca,mean,PCS)

	#LDA Code
	Sw = np.matrix(np.zeros([PCS,PCS]))
	class_means_list = getMeanPerClass(Y,2)
	for x in range(2):
		cov = getCovarianceMatrix(Y[:,x * 10 : (x + 1) * 10 ],False,2,True) / 2
		Sw += cov

	class_means_mat = np.matrix(np.zeros([PCS,2]));
	for c in range(2):
		class_means_mat[:,c] = class_means_list[c]

	Sb = getCovarianceMatrix(class_means_mat,False,2,True)

	print "LDA : Sw Shape : ", Sw.shape
	print "LDA : Sb Shape : ", Sb.shape

	[geig_vals,geig_vectors] = LDA(Sb, Sw)
	print "LDA : eigen values : ", len(geig_vals)
	print "LDA : eigen vectors : ", geig_vectors.shape

	result_matrix = np.matrix(np.zeros([len(par_knn),len(par_lda)]))

	geig_norm = ComputeNorm(geig_vectors);

	for i in range(len(par_knn)):
		KNN = par_knn[i]
		for j in range(len(par_lda)):
			LD = par_lda[j]
			#number of LDA features to use
			geig_vectors_sm = geig_vectors[:,0:LD]
			
			YL = projectImagesOntoLDA(Y, geig_vectors_sm)
			TL = projectImagesOntoLDA(T, geig_vectors_sm)
			#LDA ends here
			print "Train indices : ", train_idx[0,4*10:6*10]
			print "Labels : ",labels[4*10:6*10]
			#KNN = 1
			indi1 = KNNClassify(TL,YL,np.matrix(train_idx[0,4*10:6*10]),labels,KNN)
			k = 0
			for index in gh_indices:
				indi[index] = indi1[k]
				k += 1
			[conf,accu] = printClassification(indi,'null',test_idx,labels,True)
			print "Q3 2Level Accuracy : ",accu*100,"\nConfusion matrix\n",conf
			result_matrix[i,j] = accu*100
	return result_matrix

#plot graph for accuracy at various parameters
# For method Q3.1
def evaluateQ3Dot1(features,labels,SPLITS):
	par_knn = np.arange(1,10,2)
	par_pcs = np.arange(1,20)
	result_accu= np.matrix(np.zeros([len(par_knn),len(par_pcs)]))
	plot_colors = ['blue','green','red','black','cyan']

	for x in range(1,SPLITS  + 1):
		result = Q3Dot1(x, features, labels,par_knn,par_pcs)
		result_accu = result_accu + result
	result_accu /= SPLITS
	print "Average Accuracy : ", result_accu

	for i in range(len(par_knn)):
		knn_pcs_result = result_accu[i].T
		plot(par_pcs,knn_pcs_result[:],color=plot_colors[i],label=str(par_knn[i]) + 'NN')
	legend(loc='lower right')
	xlabel('Num. of LDA Features')
	xticks(par_pcs)
	ylabel('Average Accuracy over ' + str(SPLITS) + ' splits (%)')
	title('Q3 : PCA,LDA & KNN vs Accuracy (1-level classifier)')
	show()

#plot graph for accuracy at various parameters
# For method Q3.2
def evaluateQ3Dot2(features,labels,SPLITS):
	par_knn = np.arange(1,10,2)
	par_pcs = np.arange(1,10)
	result_accu= np.matrix(np.zeros([len(par_knn),len(par_pcs)]))
	plot_colors = ['blue','green','red','black','cyan']
	for x in range(1,SPLITS  + 1):
		result = Q3Dot2(x, features, labels,par_knn,par_pcs)
		result_accu = result_accu + result
	result_accu /= SPLITS
	print "Average Accuracy : ", result_accu

	for i in range(len(par_knn)):
		knn_pcs_result = result_accu[i].T
		plot(par_pcs,knn_pcs_result[:],color=plot_colors[i],label=str(par_knn[i]) + 'NN')
	legend(loc='lower right')
	xlabel('Num. of LDA Features')
	xticks(par_pcs)
	ylabel('Average Accuracy over ' + str(SPLITS) + ' splits (%)')
	title('Q3 : PCA,LDA & KNN vs Accuracy (2-level classifier)')
	show()

#plot graph for accuracy at various parameters
# For method Q2.1
def evaluateQ2Dot1(features,labels,SPLITS):
	par_knn = np.arange(1,10,2)

	result_accu= np.array(np.zeros([len(par_knn)]))
	for x in range(1,SPLITS  + 1):
		result = Q2Dot1(x, features, labels,par_knn)
		result_accu = result_accu + result
	result_accu /= SPLITS
	print "Average Accuracy : ", result_accu

	plot(par_knn,result_accu,color='blue')
	legend(loc='lower right')
	xlabel('K - Neighbors')
	xticks(par_knn)
	ylabel('Average Accuracy over ' + str(SPLITS) + ' splits')
	title('Q2 : KNN vs Accuracy')
	show()

#plot graph for accuracy at various parameters
# For method Q2.2
def evaluateQ2Dot2(features,labels,SPLITS):
	par_knn = np.arange(1,10,2)
	par_pcs = np.arange(1,30)
	result_accu= np.matrix(np.zeros([len(par_knn),len(par_pcs)]))
	plot_colors = ['blue','green','red','black','cyan']

	for x in range(1,SPLITS  + 1):
		result = Q2Dot2(x, features, labels,par_knn,par_pcs)
		result_accu = result_accu + result
	result_accu /= SPLITS
	print "Average Accuracy : ", result_accu

	for i in range(len(par_knn)):
		knn_pcs_result = result_accu[i].T
		plot(par_pcs,knn_pcs_result[:],color=plot_colors[i],label=str(par_knn[i]) + 'NN')
	legend(loc='lower right')
	xlabel('Num. of PCA Features')
	xticks([1,5,10,15,20,25,30])
	ylabel('Average Accuracy over ' + str(SPLITS) + ' splits (%)')
	title('Q2 : PCA & KNN vs Accuracy')
	show()

" ---------------------------"
""" MAIN """


def main():
	
	[features,labels] = getFeatures()
	print "Features : ", features.shape
	#print "Labels : ", labels
	SPLITS = 20

	plots = raw_input("Do you want to generate plots ? (y/n)")
	question = raw_input("Which question ? (1/2/3)")
	method = raw_input("Which method ? (1/2)")

	if str.lower(plots) == 'y' and question != "1":
		if question == "2":
			if method == "1":
				evaluateQ2Dot1(features, labels, SPLITS)
			else:
				evaluateQ2Dot2(features, labels, SPLITS)
		if question == "3":
			if method == "1":
				evaluateQ3Dot1(features, labels, SPLITS)
			else:
				evaluateQ3Dot2(features, labels, SPLITS)
	else:
		taccu = 0
		sqerr = 0
		tconf = np.matrix(np.zeros([10.10]))
		for x in range(1,SPLITS  + 1):
			[conf,accu] = train_classify(int(question), x, features, labels, int(method))
			tconf = tconf + conf
			taccu = taccu + accu
			sqerr += math.pow(1 - accu,2)
		taccu *= 100/SPLITS
		print "Average Accuracy : ", taccu,"%"
		print "Average Error Rate : ", 100 - taccu, "%"
		print "Standard Deviation : ", math.sqrt(sqerr/SPLITS)
		print "Average Confusion Matrix"
		print tconf

main()