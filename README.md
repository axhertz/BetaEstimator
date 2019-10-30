# BetaEstimator

Please find the Python running example in "PythonExample". You may need to install additional pyhton-packages as can be found in the import-clauses. Simply use python3 evalForest.py to run 10k queries on the forest data set and get the estimated cardinality with regard to the sample.

The basic implementation of Algorithm 1 (partialEstimate) using Brent's method can be found in SolverC++. Please compile with the optimization flag "-O3" and then run the exectuable as usual. The procedure will read in a file with precomputed parameters (e.g. using betaEstimator.py) and outputs the estimated selectivity. 
