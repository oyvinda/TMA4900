#pragma once

#include "objective_cable.h"

#include "objective_cable_ref.h"

#include "simdata.h"

#include "BFGS.h"

#include <string>
#include <fstream>

/*
This is an auxillary file, which reads the data from Tronsholen Skeiane,
and calls routines to perform parameter estimation.
*/


void tronsholen_data(int dim) { 

	/*
	
	*/

	std::string path = "/home/oyvinda/Desktop/cabledata_test.txt",
					line;
#ifdef markov
	path = "/home/shomec/o/oyvinau/cabledata_test.txt";
#endif

	std::ifstream fin(path);

	// read first line
	getline(fin, line);
	// ignore newline char
	fin.ignore();

	// read the data in to a vector
	std::vector<double> data;
	double val;
	while (fin >> val) {
		data.push_back(val); 
	}
	

	// close file
	fin.close();


	// convert the data to matrices 

	int ncol = 21;
	int nrow = data.size()/ncol;

	Matrix<double,Dynamic,Dynamic> obsmat(16, nrow), // temperatures 1 - 16
								   exvarmat(5, nrow); // left, middle, right, rainAccumulation, temperatureAir
	for (int i = 0; i < nrow; i++) {

		// read the temperature entries to the observation matrix
		for (int j = 0; j < 16; j++) {
			obsmat(j,i) = data[i*ncol + j];
		}

		// read the external variables entries to the exvar matrix 
		for (int j = 16; j < 21; j++) {
			exvarmat(j-16,i) = data[i*ncol + j];
		}
	}








	int n = dim; 
	int m = 8;





	

	// initial parameters 
	Matrix<double,Dynamic,1> param(12); // coef, var, 

	param.setOnes(); 


	param[0] = 0.003; // beta, 
	param[1] = 3.e-3; // sigma D 
	param[2] = 3.e-3; // H phi
	param[3] = 5.e-5; // sigma R 
	param[4] = 0.001; // air soil interface 
	param[5] = 10.; // omega 
	
	param[6] = 5.; // bottom const
	param[7] = 5.; // bottom periodic 
	param[8] = 1.; // bottom offset 

	param[9] = 2.e-2; // alpha 

	param[10] = 1.e-3;//0.; // sigma top
	param[11] = 1.;//1.e4; // phi top

	auto param_(param);

	//param_[12] = 0.11480132701071939227; // mu1
	//param_[13] = 0.00000000000000137863; // mu2
	//param_[14] = 0.23058101411644213452; // mu3
	//param_[15] = 0.41418101851817534786; // mu1rain


	param=stan::math::absfinv(param);
	param_=stan::math::absfinv(param_);
	param[8] = -1000.;
	param_[8] = -8.; 



	// simulate data 
	simdata_cable1 sim(n, 8, obsmat.block(0,0,16,5814), exvarmat); 

#ifdef use_sim_data

	// use simluated data
	sim(param); 
#endif

	// define functor
	obj ll(n, 8, sim.y, exvarmat);






#ifdef estsim
	// estimate parameters on simulated data
	int nsim = estsim;

	path = "/home/oyvinda/Desktop/paramests.txt";
#ifdef markov
	path = "/home/shomec/o/oyvinau/paramests.txt";
#endif
	std::ofstream fout(path, std::ios::app);

	for (int i = 0; i < nsim; i++) {
		sim(param);
		innovation_normal_ll_cable1 ll(n, 8, sim.y, exvarmat);

		param_ = BFGS(ll, param, breaksize, nit, true, false);

		// using square 
		double delta = param_[8];
		param_ = stan::math::absf(param_);
		param_[8] = delta;

		//

		for (int j = 0; j < param_.size(); j++) {
			fout << param_[j] << ' ';
		}
		fout << '\n';

		std::cout << "itnr. " << i << std::endl;
	}

	fout.close();
	return;
#endif


#ifdef comphess

	// compute the Hessian by AD/FD

	Matrix<double,Dynamic,1> eps = comphess*stan::math::fabs(param_);

	Matrix<double,Dynamic,Dynamic> hess(param.size(),param.size());
	hess.setZero();
	for (int i = 0; i < param_.size(); i++) {
		auto delta(eps);
		delta.setZero();
		delta(i) = eps(i);


		auto forw = compute_gradient(ll, param_ + delta),
			backw = compute_gradient(ll, param_ - delta);
		auto col = (forw-backw) / (2*eps(i));

		hess.block(0, i, param_.size(), 1) = col;
	}
	auto mat_ = 2.*hess.inverse(); 


	std::ofstream fout("/home/oyvinda/Desktop/invHess.txt");
	for (int i_ = 0; i_ < param.size(); i_++) {
		for (int j_ = 0; j_ < param.size(); j_++) {
			fout << mat_(i_,j_) << ' ';
		}
		fout << '\n';
	}
	// close the file
	fout.close(); 
	return; 
#endif

#ifdef writedata
	std::cout << '\n' << "AIC: " << ll(param_) + 2.*param_.size() + 5814.*8.*stan::math::log(2.*3.14159265359) << ' ';
	return; 
#endif


	// estimate parameters
 	param = BFGS(ll, param_, breaksize, nit); 


	

	// compute AIC 
	std::cout << "AIC: " << 2*param.size() + ll(param) + 5814*m*stan::math::log(2* 3.14159265359) << std::endl;

}
