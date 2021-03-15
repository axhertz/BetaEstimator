import math
import numpy as np
import pandas
from betaEstimator import partialEstimate
import random
import sys


sample_size = 1000
table_size = 581012


def getGreedyEnumeration(query_part,result_set):
	all_predicates = []
	best_candidate_position = 0
	visited_preds = np.zeros(7)
	greedy_plan = []

	for predicate_num in range(7):
			conj_part = (result_set["column_"+ query_part[predicate_num*3]] >= float(query_part[predicate_num*3+1])) \
			& (result_set["column_"+ query_part[predicate_num*3]] <= float(query_part[predicate_num*3+2]))
			all_predicates.append(conj_part)

	for i in range(7):
		best_cost = sys.maxsize
		for position in range(len(all_predicates)):
			if visited_preds[position] != 0 : continue
			current_cost = len(result_set.loc[all_predicates[position]])
			if current_cost <= best_cost:
				best_candidate_position = position
				best_cost = current_cost
		greedy_plan.append(best_candidate_position)
		result_set = result_set.loc[all_predicates[best_candidate_position]]
		visited_preds[best_candidate_position] = True

	return greedy_plan


def getAdaptiveGreedyEnumeration(query_part,n_sample,query_num):
	best_candidate_position = 0
	visited_preds = np.zeros(7)
	residual_preds = np.zeros(7)
	greedy_plan = []
	partial_plan = []
	bv_list = []
	qualTup = 0

	for predicate_num in range(7):
		col_num = int(query_part[predicate_num*3])
		lower_bound = float(query_part[predicate_num*3+1])
		upper_bound = float(query_part[predicate_num*3+2])
		res_col = n_sample[:,col_num]
		tid_list  = np.where(np.logical_and(res_col >= lower_bound, res_col <= upper_bound))
		bvInput = np.zeros(len(res_col), dtype=bool)
		for tid in tid_list:
			bvInput[tid] = True
		bv_list.append(bvInput.copy())

	bvResult = np.zeros(len(res_col),dtype=bool)
	bvResult = bvResult +1
	bvInput_res = np.zeros(len(res_col),dtype=bool)
	bvInput_res = bvInput_res + 1 
	residuals_started = False
	first_phase_sel = 1
	plan = []
	current_sel = 1
	early_break = True

	for p in range(7):
		best_sel = 1
		for predicate_num in range(7):
			if visited_preds[predicate_num] != 0 or residual_preds[predicate_num] != 0 : continue
			bvInput = bv_list[predicate_num]
			bvTest= bvResult&bvInput
			residual_test = np.zeros(len(bvResult),dtype=bool) +True
			for pred_num in range(7):
				if visited_preds[pred_num]:continue
				residual_test = residual_test & bv_list[pred_num]
			if np.count_nonzero(bvResult) == 0 and np.count_nonzero(visited_preds) > 0 and residuals_started \
			and np.count_nonzero(bvInput_res)==np.count_nonzero(bvInput_res & bvInput):
				sel = current_sel
				qualTup = 0
			elif np.count_nonzero(bvResult) == 0 and np.count_nonzero(visited_preds) > 0 and np.count_nonzero(residual_test):
				qualTup = 0
				residual_test = np.zeros(len(bvResult),dtype=bool) +True
				for pred_num in range(7):
					if visited_preds[pred_num]:continue
					residual_test = residual_test & bv_list[pred_num]
				bvInput = bvInput & bvInput_res
				if np.count_nonzero(residual_test): #start second phase
					residuals_started = True
					visitedBv = []
					for i in partial_plan:
						if visited_preds[i] != 0:
							visitedBv.append(bv_list[i])
					conditional_found = False
					while(len(visitedBv) and not conditional_found):
	
						testVec = np.zeros(sample_size,dtype=bool)
						testVec = testVec+True

						for vec in visitedBv:
							testVec = testVec & vec


						if np.count_nonzero(testVec & bvInput):
							sel = first_phase_sel * np.count_nonzero(testVec &bvInput)/np.count_nonzero(testVec)
							conditional_found = True
						elif np.count_nonzero(np.invert(bvInput) & testVec):
							pB = np.count_nonzero(testVec)/sample_size
							pA = np.count_nonzero(bvInput)/sample_size
							if early_break:
								sel_and_pred = []
								res_vec_complete = bvInput
								for res_p in range(7):
									if visited_preds[res_p] or residual_preds[res_p]: continue
									sel_and_pred.append((np.count_nonzero(bv_list[res_p]), res_p))
									res_vec_complete = res_vec_complete & bv_list[res_p]
								
								sel_and_pred.sort()
								res_plan = [x[1] for x in sel_and_pred]
								plan = plan + res_plan

								pB = np.count_nonzero(testVec)/sample_size
								pA = np.count_nonzero(res_vec_complete)/len(res_vec_complete)

								sel =  partialEstimate(pA, pB, res_vec_complete, testVec,sample_size, query_num)/pB * first_phase_sel
								return sel, 0, plan 

							pB = np.count_nonzero(testVec)/sample_size
							pA = np.count_nonzero(bvInput)/sample_size
							sel =  partialEstimate(pA, pB, bvInput, testVec,sample_size, query_num)/pB * first_phase_sel
							conditional_found = True
						visitedBv.pop()	

					if not conditional_found:
						sel = first_phase_sel*np.count_nonzero(bvInput)/sample_size# fallback to AVI

			if not residuals_started:
				pA = current_sel
				pB = np.count_nonzero(bvInput)/len(bvInput)
				if np.count_nonzero(bvTest):
					sel = np.count_nonzero(bvTest)/len(bvTest)
					qualTup = np.count_nonzero(bvTest)
				elif ( np.count_nonzero(visited_preds) == 0): # case first predicate gives null vector
					sel = math.log(2)/(2*len(bvResult))
				else: # use estimator
					qualTup = 0
					if(np.count_nonzero(bvInput) == 0): #take care of null vector (otherwise no solution)
						pB = math.log(2)/(2*sample_size)
					elif(np.count_nonzero(bvInput) == len(bvInput)): # take care of unit vector (otherwise no solution)
						pB = 1 - math.log(2)/(2*sample_size)
					if not np.count_nonzero(bvResult) or not np.count_nonzero(bvInput) or np.count_nonzero(bvInput) == sample_size: # use AVI
						sel = current_sel*pB
					else:
						sel = partialEstimate(pA, pB, bvResult, bvInput, sample_size, query_num)

			if sel <= best_sel:
				best_candidate_position = predicate_num
				best_sel = sel


		current_sel = best_sel
		if residuals_started:
			residual_preds[best_candidate_position] = 1
			bvInput_res = bvInput_res & bv_list[best_candidate_position]
			if current_sel*table_size < sample_size: early_break = True
		else:
			bvResult = bvResult&bv_list[best_candidate_position]
			visited_preds[best_candidate_position] = 1
			first_phase_sel = current_sel
			partial_plan.append(best_candidate_position)
			if current_sel*table_size < sample_size: early_break = True
		plan.append(best_candidate_position)
	return current_sel, qualTup, plan




if __name__ == "__main__":


	df_sample = pandas.read_csv('Data/forest_sample_1k.csv'.format(sample_size), index_col=0)
	df_sample.index=range(len(df_sample))
	n_sample = df_sample.to_numpy()
	query_file = open("Workload/forest_qu7.txt","r")

	for query_num,line in enumerate(query_file.readlines()):
		query_part = line.split(" ")
		selectivity, qualifying_tuples, plan = getAdaptiveGreedyEnumeration(query_part, n_sample,query_num)
		print('{:<15}  {:<30}  {:<25}  {:<0} '.format("query num: "+ str(query_num+1), "estimated cardinality: {0:.1f}".format(max(1,selectivity*table_size)),\
			 "qualifying tuples: "+str(qualifying_tuples), "greedy plan: "+str(plan)))

	query_file.close()


