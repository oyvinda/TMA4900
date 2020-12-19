#pragma once 

#include <cmath>
#include <stan/math.hpp>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <random>
#include <fstream>

using Eigen::Matrix;
using Eigen::Dynamic;


/*
This file contains an implementation 
of the quasi-Newton method BFGS.
*/


template<typename T>
Matrix<double,Dynamic,1> compute_gradient(const T& f, const Matrix<double,Dynamic,1>& x, bool ad = true) {

	/*
	Return the computed gradient of functor f. 
	*/

	double fx;
	Matrix<double,Dynamic,1> grad;
	if (ad) {
		stan::math::gradient(f, x, fx, grad);
	} else {
		stan::math::finite_diff_gradient(f, x, fx, grad, 1.e-3);
	}

	return grad; 
}



template<typename T> 
Matrix<double,Dynamic,1> BFGS(const T& f,
							  const Matrix<double,Dynamic,1>& init_param,
							  double breaksize,
							  int maxit,
							  bool ad = true,
							  bool verbose = true) {

	/*
	Minimize f using BFGS, 
	and return the minima. 
	*/


	// number of parameters
	int nparam = init_param.size();

	// parameters
	Matrix<double,Dynamic,1> param = init_param, // parameters
									p(nparam), // search direciton
									g(nparam), gprev(nparam), gstep(nparam), // gradient
									s(nparam), u(nparam); // auxillary variables 

	// inverse Hessian approximation
	Matrix<double,Dynamic,Dynamic> Binv(nparam,nparam); 
	Binv.setIdentity();

	double alpha = 1.,
		   curr, prop, su=0., gradsize;



	// compute initial gradient
	g = compute_gradient(f, param, ad);
	// 
	if (g.transpose()*g <= breaksize) {
		std::cout << "Converged!\n" << "\nGradient size: " << g.transpose()*g << std::endl;
		return param;
	}





	// BFGS

	int maxitlinesearch = 200; 
	double c1 = 1.e-5, // Armijo
		   c2 = 0.9; // Goldstein

	for (int it = 0; it < maxit; it++) {

		// compute search direction
		p = -Binv * g;
		
		// compute alpha 
		alpha = 1.;
		curr = f(param);

		// attempt Newton step first
		param += p;

		for (int itlinesearch = 0; itlinesearch < maxitlinesearch; itlinesearch++) {

			try {
				// try to compute linesearch value
				prop = f(param); 

				
				// if nan 
				if (isnan(prop))
					throw std::exception();


			} catch (std::exception& e) {
				//std::cout << "Error in linesearch" << std::endl;

				alpha /= 2.;
				param -= alpha * p;
				continue;
			}
			
			// check i) Armijo and ii) Goldstein (strong) condition 

			if (prop <= curr + c1 * alpha * p.transpose() * g) { // i)


				// compute gradient to check ii)
				gstep = compute_gradient(f, param, ad); 

				if (abs(p.transpose() * gstep) <= abs(c2 * p.transpose() * g) || (it<nchange) ) // ii)
					break;

			}


			alpha /= 2.;
			param -= alpha * p;
		}

		// update
		s = alpha * p;


		// compute new gradient
		gprev = g;
		g = gstep; //compute_gradient(f, param, ad);


		// see if we can break
		gradsize = g.transpose() * g;
		if ((gradsize < breaksize) && true ) { // (s.transpose() * s < 1.e-10) ) { // 
			if (verbose)
				std::cout << "Converged!\n" << "Iterations: " << it << "\nGradient size: " << gradsize << std::endl;

			/*
			// read the approximate inverse Hessian to file
			std::ofstream fout("/home/oyvinda/Desktop/invHess.txt");
			for (int i_ = 0; i_ < nparam; i_++) {
				for (int j_ = 0; j_ < nparam; j_++) {
					fout << 2 * Binv(i_,j_) << ' ';
				}
				fout << '\n';
			}
			// close the file
			fout.close(); */


			break;
		}


		// auxillary variable 
		u = g - gprev;
		su = s.transpose() * u;



		// update inverse Hessian approximation
		// update only if close to solution (and reset regularily)
		if (su > 0. && (it % nreset > 0) ) { 
			/*Binv = (id - s*u.transpose()/su) * Binv * (id - u*s.transpose()/su) 
					+ s*s.transpose()/su; */
			Binv = Binv + ( su + u.transpose() * Binv * u ) * ( s * s.transpose() ) / (su*su) 
				   - ( Binv * u * s.transpose() + s * u.transpose() * Binv ) / su;
		} else {
			// reset the Hessian approximation
			Binv.setIdentity();
		}



		
		if (verbose && (it % 10 == 0)) {
			std::cout << "Objective value: " << prop 
					  << "\nGradient size: " << gradsize 
					  << std::endl;

			std::cout << "Current est:\n" << param << "\n\n" << std::endl; 
			//std::cout << "Current grad:\n" << g << "\n\n" << std::endl;
		}

	}

	return param;
	
}

