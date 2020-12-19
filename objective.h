#pragma once

#include <cmath>
#include <stan/math.hpp>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <random>

#include <ctime>
std::default_random_engine generator(3); // time(NULL) defined globally for simluations 

#include "BFGS.h"

// for matrix sqrt
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::Matrix;
using Eigen::Dynamic;

using stan::math::var;


/*
This file contains an implementation of direct maximization of the innovations likelihood,
and maximization by EM. Both model have completely parameterized sytem, observation and 
noise covariance matrices. 

It was used in the thesis to compare the two methods of obtaining ML esitmates. 
*/


struct innovation_normal_ll_discrete {

	/*
	Functor returning -2 * log(L) + m * T * log(2 * pi),
	where L is the likelihood of the Gaussian observations, 
	produced by the innovations and their variance. 
	*/


	const int n, m;

	
	// observations
	const Matrix<double,Dynamic,Dynamic> y;
	// external variables
	const Matrix<double,Dynamic,Dynamic> exvar;






	//constructor
	innovation_normal_ll_discrete(int n_, int m_, const Matrix<double,Dynamic,Dynamic>& y_,
						 const Matrix<double,Dynamic,Dynamic>& exvar_) : n(n_), m(m_), y(y_), exvar(exvar_) { }




	template<typename T>
	T operator ()(const Matrix<T,Dynamic,1>& param) const {

		using stan::math::multiply;
		using stan::math::log_determinant;
		using stan::math::sum;
		using stan::math::matrix_exp;



		// insert initial parameters

		Matrix<T,Dynamic,1> x0(n), xu(n), xc(n), e(m);
		Matrix<T,Dynamic,Dynamic> S0(n,n), Su(n,n), Sc(n,n), A(n,n), B(m,n), Q(n,n), R(m,m), evar(m,m), evarinv(m,m), M(n,m);


		x0.setOnes(); 
		S0.setZero();
		A.setZero();
		B.setIdentity();
		Q.setZero();
		R.setIdentity();
		R *= 1e-4;

		
		int idx = 0;

		// A matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A(i,j) = param[idx++];
			}
		}

		// Q matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i;j++) {
				Q(i,j) = param[idx++];
				Q(j,i) = Q(i,j);
			}
		}
		// square
		Q = Q * Q.transpose();

		// compute gaussian ll

		std::vector<T> acc; 

		xu = x0;
		Su = S0;

		for (int t = 0; t < y.cols(); t++) {


			// perform filtering computations


			// compute the innovation variance
			evar = multiply(B, multiply(Su, B.transpose())) + R;

			// compute its inverse 
			evarinv = evar.inverse();

			// compute the projection matrix
			M = multiply(Su, multiply(B.transpose(), evarinv)); 

			// compute the innovation
			e = y.col(t) - multiply(B, xu);

			// condition on observation
			xc = xu + multiply(M, e);

			// conditioned variance
			Sc = Su - multiply(M, multiply(B, Su));

			// forward
			xu = multiply(A, xc);
			Su = multiply(A, multiply(Sc, A.transpose())) + Q;


			// add to the log likelihood
			// the log determinant should be computed stabily
			acc.push_back(log_determinant(evar) + multiply(e.transpose(), multiply(evarinv, e)));


		}

		return sum(acc);
	}

};




struct em_normal_ll_discrete {

	/*
	Data structure to perfom ML estimation 
	using the EM algorithm. 
	*/
	

	const int n, m,
			nobs;

	const Matrix<double,Dynamic,Dynamic> y, exvar;

	Matrix<double,Dynamic,Dynamic>* Sus;
	Matrix<double,Dynamic,Dynamic>* Scs;
	Matrix<double,Dynamic,1>* xus;
	Matrix<double,Dynamic,1>* xcs;



	em_normal_ll_discrete(int n_, int m_, const Matrix<double,Dynamic,Dynamic>& y_, const Matrix<double,Dynamic,Dynamic>& exvar_)
		: n(n_), m(m_), y(y_), exvar(exvar_), nobs(y_.cols()) 
	{ 
		
		// 

		xus = new Matrix<double,Dynamic,1>[nobs];
		xcs = new Matrix<double,Dynamic,1>[nobs];

		Sus = new Matrix<double,Dynamic,Dynamic>[nobs];
		Scs = new Matrix<double,Dynamic,Dynamic>[nobs];

	}


	Matrix<double,Dynamic,1> estimate(const Matrix<double,Dynamic,1>& init_param, double breaksize, int maxit) {
		


		// insert initial parameters

		Matrix<double,Dynamic,1> param = init_param,
								 param_prev = init_param;

		Matrix<double,Dynamic,1> x0(n), xu(n), xc(n), e(m);
		Matrix<double,Dynamic,Dynamic> S0(n,n), Su(n,n), Sc(n,n), A(n,n), B(m,n), Q(n,n), R(m,m), evar(m,m), evarinv(m,m), M(n,m);

		
		x0.setOnes(); 
		S0.setZero();
		
		A.setZero(); 
		B.setIdentity();

		Q.setZero();

		R.setIdentity();
		R *= 1e-4;

		int idx = 0;


		// A matrix
		
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A(i,j) = param[idx++];
			}
		}

		// Q matrix

		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				Q(i,j) = param[idx++];
				Q(j,i) = Q(i,j);
			}
		}
		// square
		Q = Q * Q.transpose();


		
		// perform EM update scheme

		for (int it = 0; it < maxit; it++) {


			// perform filtering

			// set initial values

			xu = x0;
			Su = S0;

			for (int t = 0; t < y.cols(); t++) {


				// store the predictions and variances 
				
				xus[t] = xu;
				Sus[t] = Su;
				


				// compute the innovation variance
				evar = B * Su * B.transpose() + R;

				// compute its inverse 
				evarinv = evar.inverse();

				// compute the projection matrix
				M = Su * B.transpose() * evarinv; 

				// compute the innovation
				e = y.col(t) - B * xu;

				// condition on observation
				xc = xu + M * e;

				// conditioned variance
				Sc = Su - M * B * Su;

				// forward
				xu = A * xc;
				Su = A * Sc * A.transpose() + Q;


				// store 

				Scs[t] = Sc; 
				xcs[t] = xc;
				
			}

			// perform smoothing

			// use Sc and xc as St xt

			for (int t = y.cols()-2; t >= 0; t--) {

				// compute the projection matrix
				M = Scs[t] * A.transpose() * Sus[t+1].inverse();

				// compute the smoothed mean
				xcs[t] = xcs[t] + M * (xcs[t+1] - xus[t+1]);

				// compute the smoothed variance
				Scs[t] = Scs[t] + M * (Scs[t+1] - Sus[t+1]) * M.transpose();

			}

			
			// em estimates

			Matrix<double,Dynamic,Dynamic> eyxt(n,n), exxt(n,n), yyt(n,n), exxpastt(n,n);
			eyxt.setZero();
			exxt.setZero();
			yyt.setZero();
			exxpastt.setZero();



			for (int i = 0; i < nobs; i++) {
				eyxt += y.col(i) * xcs[i].transpose();
				yyt += y.col(i) * y.col(i).transpose();
				exxt += Scs[i] + xcs[i] * xcs[i].transpose();

				if (i == 0) 
					continue;
				exxpastt += A * Scs[i-1] + xcs[i] * xcs[i-1].transpose();

			}

			//x0 = xcs[0]; // initial mean
			//S0 = Scs[0]; // initial variance

			A = exxpastt * (exxt - xcs[nobs-1] * xcs[nobs-1].transpose()).inverse(); // system
			//B = eyxt * exxt.inverse(); // measurment

			Q = ( (exxt - xcs[0] * xcs[0].transpose()) - A * exxpastt.transpose() ) / (nobs - 1.); // system noise
			//R = (yyt - B * eyxt.transpose()) / nobs; // measurement noise


			// insert parameters to check if we can break...
			int idx = 0;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					param[idx++] = A(i,j);
				}
			}
			for (int i = 0; i < n; i++) {
				for (int j = 0; j <= i; j++) {
					param[idx++] = Q(i,j);
				}
			}
			double delta_param_size = (param - param_prev).transpose() * (param - param_prev);
			if (delta_param_size < breaksize) {
				std::cout << "Converged!\n" << "Iterations: " << it << std::endl;
				break;
			}
			param_prev = param;
		}

		return param;
	}



	~em_normal_ll_discrete() 
	{
	
		delete[] xus;
		delete[] xcs;
		delete[] Sus;
		delete[] Scs;	

	}
};

