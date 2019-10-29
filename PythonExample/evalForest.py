import betaEstimator
import enumerator
import pandas
import numpy as np

sample_size = "1per" # 1k
enumeration_policy = "Greedy" # Dp, Desc
table_size = 581012

if __name__ == "__main__":

	# get pre-computed predicate enumeration
	plan_all_queries = []
	with open("./PredicateOrder/{}Plan.txt".format(enumeration_policy), "r") as plan_file:
		for line in plan_file:
			plan = [int(pred) for pred in line.rstrip().split(" ")]
			plan_all_queries.append(plan)
	# use pre-computed sample or generate new from Data/forest.csv 
	df_sample = pandas.read_csv('Data/forest_sample_{}.csv'.format(sample_size), index_col=0)
	#df_sample = pandas.read_csv('Data/forest.csv'.format(sample_size))
	
	query_file = open("Workload/forest_qu7.txt", "r") # column lower upper ... -> attribute BETWEEN lower AND upper AND ...
	for query_num, line in enumerate(query_file.readlines()):
		query_string = line.split(" ")
		# use pre-computed enumeration or call enumerator.strategy(query_string, df_sample)
		plan = plan_all_queries[query_num] 
		selectivity, qualifying_tuples =  betaEstimator.getSelectivityEstimate(plan, query_string, df_sample)
		print('{:<20}  {:<35}  {:<0}'.format("query num: "+ str(query_num+1), "estimated cardinality: {0:.3f}".format(selectivity*table_size),\
			 "qualifying tuples: "+str(qualifying_tuples)))

