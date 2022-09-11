/**
 * Basic implementation of Algorithm 1
 */
#include "boost/math/distributions/beta.hpp"
#include "boost/math/tools/minima.hpp"
#include <iostream>
#include <chrono>
#include <limits>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <tuple>


struct targetFunc_AB{

	targetFunc_AB(double pA_, double pB_, double a1_, double b1_, double a2_, double b2_):
	pA(pA_), pB(pB_),a1(a1_), b1(b1_),a2(a2_), b2(b2_) {}

	double operator()(double const& z_AB){
		if (z_AB < 0 || z_AB> 1 || (pA-z_AB*pB)/(1-pB) < 0 || (pA-z_AB*pB)/(1-pB) > 1 ){
			return std::numeric_limits<double>::max();
		}

		cdf_AB = boost::math::ibetac(a1,b1, z_AB);
		cdf_AnotB = boost::math::ibetac(a2,b2, (pA-z_AB*pB)/(1-pB));

		if(cdf_AnotB == 0 || cdf_AB == 0){
			return std::numeric_limits<double>::max();
		}
		return std::max<double>(cdf_AB/cdf_AnotB,cdf_AnotB/cdf_AB);
	}

	double pA;
	double pB;
	double cdf_AB;
	double cdf_AnotB; 
	double a1,b1,a2,b2;
};

struct targetFunc_AnotB{

	targetFunc_AnotB(double pA_, double pB_, double a1_, double b1_, double a2_, double b2_):
	pA(pA_), pB(pB_),a1(a1_), b1(b1_),a2(a2_), b2(b2_) {}

	double operator()(double const& z_AnotB){
		if (z_AnotB < 0 || z_AnotB> 1 || (pA-z_AnotB*(1-pB))/pB < 0 || (pA-z_AnotB*(1-pB))/pB > 1 ){
			return std::numeric_limits<double>::max();
		}
		cdf_AB = boost::math::ibetac(a1,b1, (pA-z_AnotB*(1-pB))/pB);
		cdf_AnotB = boost::math::ibetac(a2,b2, z_AnotB);
		if(cdf_AnotB == 0 || cdf_AB == 0){
			return std::numeric_limits<double>::max();
		}
		return std::max<double>(cdf_AB/cdf_AnotB,cdf_AnotB/cdf_AB);
	}

	double pA;
	double pB;
	double cdf_AB;
	double cdf_AnotB;
	double a1,b1,a2,b2;
};

double partialEstimate(double pA, double pB,  double k_AB, double k_AnotB, double m_B, double m_notB ){

	double zAB_lower = std::max<double>((k_AB-1)/m_B,0);
	double zAB_upper = std::min<double>((k_AB+1)/m_B,1);
	double zAnotB_lower = std::max<double>((k_AnotB-1)/m_notB,0);
	double zAnotB_upper = std::min<double>((k_AnotB+1)/m_notB,1);
	

	auto getShapeParam = [](double k, double m) // Eqn. 9
	{ 
		if (k > 0){
			return std::make_tuple(k+1.0/3.0, m-k+1.0/3.0);
		}else if(m > 0){
			return std::make_tuple(0.634, m);
		}
		return std::make_tuple(1.0,1.0);
	};

	double a1, b1, a2, b2;
	std::tie(a1,b1) = getShapeParam(k_AB, m_B);
	std::tie(a2,b2) = getShapeParam(k_AnotB, m_notB);

	int bits = 24;
   	const boost::uintmax_t maxit = 50;
   	boost::uintmax_t it = maxit;
   	
	if(zAB_upper-zAB_lower < zAnotB_upper-zAnotB_lower){
		targetFunc_AB targetFunc = targetFunc_AB(pA, pB, a1, b1, a2, b2);
		std::pair<double,double> res = boost::math::tools::brent_find_minima(targetFunc, zAB_lower,zAB_upper, bits);
		return res.first*pB;
	}else{
		targetFunc_AnotB targetFunc = targetFunc_AnotB(pA, pB, a1, b1, a2, b2);
		std::pair<double, double> res = boost::math::tools::brent_find_minima(targetFunc, zAnotB_lower,zAnotB_upper, bits);
		return (pA-res.first*(1-pB));
	}
}

int main(){
	// read prepared parameters (e.g. using the betaEstimator.py)
	std::ifstream infile("parameters_for_1per.txt");
	// write out estimated selectivity
	std::ofstream outfile("results.txt");
	std::string line;
	int query_num;
	int query_num_before = -1;
	double pA, pB, k_AB, k_AnotB, m_B, m_notB, partialSel;
	double completion_time = 0;
	double counter = 0;

	while (std::getline(infile, line))
	{
	std::istringstream iss(line);
	if (!(iss >> query_num >> pA >> pB >> k_AB >> k_AnotB >> m_B >> m_notB)) { break; } // error
	auto t1 = std::chrono::high_resolution_clock::now();
	partialSel = partialEstimate(pA,pB,k_AB,k_AnotB,m_B,m_notB);
	auto t2 = std::chrono::high_resolution_clock::now();
	if(query_num == query_num_before){
		outfile<< "query_num: " << query_num <<" second phase result:  "<< partialSel << "\n"; 	
	}
	else{
		outfile<< "query_num: " << query_num <<" first  phase result:  "<< partialSel << "\n"; 
		counter++;
	}
	completion_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() ;
	query_num_before = query_num;
	}
	std::cout<<"average time: " << completion_time/counter<<" [µs]\n";
	std::cout<<"results have been written to [\"results.txt\"] \n";
	outfile.close();

	return 0;
}
