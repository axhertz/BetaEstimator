# BetaEstimator

Repository of the upcoming paper [Small Selectivities Matter: Lifting the Burden of Empty Samples](https://dl.acm.org/doi/10.1145/3448016.3452805). Given a sample, the Beta Estimator derives more precise selectivity estimates
when no sample tuple matches the filter.

Please find the Python running example in "PythonExample". You may need to install additional pyhton-packages as can be found in the import-clauses. Simply use "python3 evalForest.py" to run 10k queries on the forest data set and get the estimated cardinality with regard to the sample. To see how our estimator integrates with predicate enumeration run "python3 run_BetaGH.py".  

The basic implementation of Algorithm 1 (partialEstimate) using Brent's method can be found in SolverC++. Please compile with the optimization flag "-O2" and then run the exectuable as usual. The procedure will read in a file with precomputed parameters (e.g. using betaEstimator.py) and outputs the estimated selectivity. 
