//#define markov 

//#define writedata 
//#define writedataforecast 2000
//#define use_sim_data 
//#define comphess 1.e-7 
//#define estsim 10
  

#define absf square // function to be used to ensure positivity of params
#define absfinv sqrt


#define covar square // space covariance function

#define obj innovation_normal_ll_cable // model

#define model_depth 1.5 // vertical depth of model 
  
const double breaksize = 1.e-5; // breaks the \| grad \|_2^2 < 1e-5. 
const int nit = 300; // total number of iteration for BFGS

const int nreset = 20, // number of iteration between Hessian approx reset 
		  nchange = 300; // after which impose curvature condition

#include "tronsholenskeiane.h" 
#include "testest.h"

#include <iomanip> // set precision



 
 
     
 
 


int main() { 

	//std::cout << std::fixed << std::setprecision(20);




	tronsholen_data(20); 





	return 0; 
}
