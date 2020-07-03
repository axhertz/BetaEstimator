import math
import numpy as np
import sys


def getIaEnumeration(query_part,result_set):
	#result_set = df

	sel_pred_pairs = []
	for predicate_num in range(7):
			conj_part = (result_set["column_"+ query_part[predicate_num*3]] >= int(query_part[predicate_num*3+1])) & (result_set["column_"+ query_part[predicate_num*3]] <= int(query_part[predicate_num*3+2]))
			sel_pred_pairs.append((len(result_set.loc[conj_part]),predicate_num))
	#		all_predicates.append(conj_part)
	sel_pred_pairs.sort(key = lambda x: x[0])
	plan = [x[1] for x in sel_pred_pairs]
	return plan


def getFullEnumeration(query_part, result_set):
	#result_set = df
	all_predicates = []
	dpConj= []
	dpCost = np.zeros(2**7)
	dpTable = []

	for predicate_num in range(7):
		conj_part = (result_set["column_"+ query_part[predicate_num*3]] >= float(query_part[predicate_num*3+1])) & (result_set["column_"+ query_part[predicate_num*3]] <= float(query_part[predicate_num*3+2]))
		all_predicates.append(conj_part)
	
	for _ in range(2**7):
		dpTable.append([])
		dpConj.append([])

	for i in range(7):
		dpConj[2**i].append(all_predicates[i])
		dpCost[2**i]= len(result_set[all_predicates[i]])
		dpTable[2**i].append(i)
	

	for i in range(1,2**7):
		selected_preds_pos = np.zeros(7)
		selected_preds = []
		single_pred_pos = []

		conjunct_pred = True
		for j in range(1,8):
			if int(i/(2**(j-1)))%2 == 1:
				selected_preds.append(all_predicates[j-1])
				selected_preds_pos[j-1] = 1
				single_pred_pos.append(2**(j-1))
				conjunct_pred = conjunct_pred & all_predicates[j-1] 
		
		costForConj = len(result_set[conjunct_pred])

		for k in range(len(selected_preds)):
			currentPred = selected_preds[k]
			currentPredPos = single_pred_pos[k]

			if len(dpConj[i -currentPredPos]) == 0:
				continue
			currentPlan = dpConj[i- currentPredPos]
			currentCost = dpCost[i-currentPredPos] + costForConj
			#print("current cost", currentCost)
			if len(dpConj[i]) == 0 or currentCost < dpCost[i]:
				dpConj[i] = conjunct_pred
				dpCost[i] = currentCost
				dpTable[i] = dpTable[i-currentPredPos]+ dpTable[currentPredPos] 

	return dpTable[-1]


def getGreedyEnumeration(query_part,result_set):
	#result_set = df
	all_predicates = []
	best_candidate_position = 0
	visited_preds = np.zeros(7)
	greedy_plan = []

	for predicate_num in range(7):
			conj_part = (result_set["column_"+ query_part[predicate_num*3]] >= int(query_part[predicate_num*3+1])) & (result_set["column_"+ query_part[predicate_num*3]] <= int(query_part[predicate_num*3+2]))
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
		visited_preds[best_candidate_position] = 1

	return greedy_plan
