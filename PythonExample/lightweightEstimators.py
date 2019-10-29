import re,sys
import math
import pandas
import random 
import numpy as np
from scipy import optimize
from scipy.stats import beta
import sys
import time



def getBasicEstimate(query_part,df_sample):
	m = len(df_sample)
	single_sel_list = []
	full_conj = True
	for predicate_num in range(7):
		conj_part = (df_sample["column_"+ query_part[predicate_num*3]] >= float(query_part[predicate_num*3+1])) & \
				    (df_sample["column_"+ query_part[predicate_num*3]] <= float(query_part[predicate_num*3+2]))
		full_conj = full_conj & conj_part
		single_sel_list.append(len(df_sample.loc[conj_part])/m)
	num_qualifying_tuples = len(df_sample.loc[full_conj])
	if num_qualifying_tuples:
		traditional_est = num_qualifying_tuples/m
		return [traditional_est*table_size]*5
	single_sel_list.sort()
	avi = 1
	ebo = 1
	for sel_num,sel in enumerate(single_sel_list):
		avi = avi*sel
		if sel_num < 4:
			ebo = ebo*sel**(1/(2**sel_num))
	minsel = min(single_sel_list)

	avi_est = avi*table_size
	ebo_est = ebo*table_size
	minsel_est = minsel*table_size
	sampleOpt_est =  table_size/m
	samplePess_est =  1

	return avi_est, ebo_est, minsel_est, samplePess_est, sampleOpt_est



if __name__ == "__main__":

	sample_size = "1k" #"1per"
	
	df_sample = pandas.read_csv('Data/forest_sample_{}.csv'.format(sample_size), index_col = 0)
	query_file = open("Workload/forest_qu7.txt", "r") # column lower upper ... -> attribute BETWEEN lower AND upper AND ...	
	table_size = 581012 #weather: 3475109		

	for num,line in enumerate(query_file.readlines()):
		query_string = line.split(" ")
		avi_est, ebo_est, minsel_est, samplePess_est, sampleOpt_est = getBasicEstimate(query_string, df_sample)
		print('{:<15}  {:<15}  {:<20}  {:<20}  {:<20}  {:<0}'.format("query num: "+ str(num), "AVI: "+ str(int(avi_est)), "ExpBackoff: "+ str(int(ebo_est)),\
			 "Min Sel: "+ str(int(minsel_est)), "Pessimistic: "+ str(int(samplePess_est)),"Optimistic: "+ str(int(sampleOpt_est))))

	query_file.close()
