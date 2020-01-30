"""
 Basic Implementation of Algorithm 2
"""
import math
import random 
import numpy as np
from scipy import optimize
from scipy.stats import beta
import sys
import pandas


def partialEstimate(pA, pB, bvResult, bvInput, sampleSize,query_num):

	def target_func_AB(z_AB):
		if (z_AB < 0 or z_AB> 1 or (pA-z_AB*pB)/(1-pB) < 0 or (pA-z_AB*pB)/(1-pB) > 1 ):
			return  sys.float_info.max

		cdf_AB = beta.cdf(z_AB, a1,b1)
		#substitute z_AnotB
		cdf_AnotB = beta.cdf((pA-z_AB*pB)/(1-pB), a2, b2)

		if(cdf_AB == 0 or cdf_AnotB == 0):
				return sys.float_info.max
		return max(cdf_AnotB/cdf_AB, cdf_AB/ cdf_AnotB)

	def target_func_AnotB(z_AnotB):
		if (z_AnotB < 0 or z_AnotB> 1 or (pA-z_AnotB*(1-pB))/pB < 0 or (pA-z_AnotB*(1-pB))/pB > 1 ):
			return sys.float_info.max

		cdf_AnotB = beta.cdf(z_AnotB,a2, b2)
		#substitute z_AB
		cdf_AB = beta.cdf((pA-z_AnotB*(1-pB))/pB, a1, b1)

		if(cdf_AB == 0 or cdf_AnotB == 0):
			return sys.float_info.max
		return max(cdf_AnotB/cdf_AB, cdf_AB/ cdf_AnotB)


	def getShapeParams(k,m):
		if k > 0: return k+1/3, m-k+1/3
		elif m > 0: return 0.634, m 
		return 1,1

	k_AB = np.count_nonzero(bvInput & bvResult)
	k_AnotB = np.count_nonzero(bvResult & np.invert(bvInput))
	m_B = np.count_nonzero(bvInput)
	m_notB = np.count_nonzero(np.invert(bvInput))

	#Uncomment to test C++ implementation!
	'''with open("parameters.txt","a+") as file:
		file.write("{} {} {} {} {} {} {}\n".format(query_num, pA, pB, k_AB, k_AnotB, m_B,m_notB))
	'''
	
	zAnotB_lower = max((k_AnotB-1)/m_notB,0)
	zAnotB_upper = min((k_AnotB+1)/m_notB,1)
	zAB_lower = max((k_AB-1)/m_B,0)
	zAB_upper = min((k_AB+1)/m_B,1)

	a1, b1 = getShapeParams(k_AB,m_B)
	a2, b2 = getShapeParams(k_AnotB, m_notB)

	try: 
		if zAB_upper - zAB_lower < zAnotB_upper - zAnotB_lower:
			res = optimize.minimize_scalar(target_func_AB, bounds=(zAB_lower,zAB_upper), method="bounded")
			if res.fun == sys.float_info.max:
				print("no solution found, fallback to AVI")
				return pA*pB
			else:
				return res.x*pB
		else:
			res = optimize.minimize_scalar(target_func_AnotB, bounds=(zAnotB_lower,zAnotB_upper), method="bounded")
			if res.fun == sys.float_info.max:
				print("no solution found, fallback to AVI")
				return pA*pB
			else:
				pAnotB = res.x
				return pA -pAnotB*(1-pB)
	except Exception as e:
		print(str(e))
		return pA * pB


def getSelectivityEstimate(enumeratedPreds, query_part, df_sample, query_num):
	sampleSize = len(df_sample)
	selectivityEstimate = 1
	bvResult = np.zeros(sampleSize,dtype = bool)
	bvResult = bvResult +1 #unit vector

	residual = np.zeros(sampleSize, dtype = bool)
	visitedPreds = []
	visitedBv = [] #BitVectors of visitedPreds

	for iter_num, predicate_num in enumerate(enumeratedPreds):
		# get bit vector of current predicate
		bvInput = np.zeros(sampleSize, dtype=bool)
		conj_part = (df_sample["column_"+ query_part[predicate_num*3]] >= float(query_part[predicate_num*3+1])) & \
				    (df_sample["column_"+ query_part[predicate_num*3]] <= float(query_part[predicate_num*3+2]))
		result_set = df_sample.loc[conj_part]

		for i in result_set.index:
			bvInput[i] = 1
	
		if np.count_nonzero(bvResult) == 0 and len(visitedPreds) > 0:
			theResidual = np.zeros(sampleSize)
			theResidual = theResidual + 1
			residual_test = df_sample
			for residual_num in range(7):
				if residual_num not in visitedPreds: 
					conj_part = (residual_test["column_"+ query_part[residual_num*3]] >= float(query_part[residual_num*3+1])) & \
					(residual_test["column_"+ query_part[residual_num*3]] <= float(query_part[residual_num*3+2]))
					residual_test = residual_test.loc[conj_part]
				if not len(residual_test): break
			
			if len(residual_test): #start second phase
				residual = np.zeros(sampleSize, dtype = bool)
				for i in residual_test.index:
					residual[i] = 1

				while(len(visitedBv)):
					testVec = np.zeros(sampleSize,dtype=bool)+1
					for vec in visitedBv:
						testVec = testVec & vec
					if np.count_nonzero(testVec &residual):
						return selectivityEstimate * np.count_nonzero(testVec &residual)/np.count_nonzero(testVec),0
					elif np.count_nonzero(np.invert(residual) & testVec):
						pB = np.count_nonzero(testVec)/sampleSize
						pA = np.count_nonzero(residual)/sampleSize
						return partialEstimate(pA, pB, residual, testVec,sampleSize, query_num)/pB * selectivityEstimate, 0
					visitedBv.pop()	
				
				return selectivityEstimate*len(residual_test)/sampleSize, 0 # fallback to AVI


		pA = selectivityEstimate
		pB = np.count_nonzero(bvInput)/sampleSize

		if (np.count_nonzero(bvInput & bvResult) != 0): # case qualifying sample
			selectivityEstimate = np.count_nonzero(bvInput&bvResult)/sampleSize
		elif (len(visitedPreds)== 0): # case first predicate gives null vector
			selectivityEstimate = math.log(2)/(2*sampleSize)
		else: # use estimator
			if(np.count_nonzero(bvInput) == 0): #take care of null vector (otherwise no solution)
				pB = math.log(2)/(2*sampleSize)
			elif(np.count_nonzero(bvInput) == len(bvInput)): # take care of unit vector (otherwise no solution)
				pB = 1 - math.log(2)/(2*sampleSize)
			if not np.count_nonzero(bvResult) or not np.count_nonzero(bvInput) or np.count_nonzero(bvInput) == sampleSize: # use AVI
				selectivityEstimate = selectivityEstimate*pB 
			else:
				selectivityEstimate = partialEstimate(pA,pB, bvResult, bvInput, sampleSize,query_num)

		visitedPreds.append(predicate_num)
		visitedBv.append(bvInput)
		bvResult = bvResult&bvInput

	if np.count_nonzero(bvResult) == 1: # 1 as strong median
		return selectivityEstimate*math.log(2), 1 

	return selectivityEstimate, np.count_nonzero(bvResult) # fallback to AVI
