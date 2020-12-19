#pragma once

#include "objective.h"
#include <fstream>

/*
This file contains an implementation of the different 
objective functions (c++ functors) used in the thesis.
*/


struct innovation_normal_ll_cable {

	/*
	Functor returning -2 * log(L) + m * T * log(2 * pi),
	produced by the innovations and their variance, for
	the Trondsholen-Skeiane dataset.
	*/


	const int n, m;

	
	// observations
	const Matrix<double,Dynamic,Dynamic> y;
	// external variables
	const Matrix<double,Dynamic,Dynamic> exvar;


	const double cableradi = 0.05, // radius of cables (equal for all cables)
			 maxdepth = model_depth, // maximum depth (below which 10C)
			 maxradi = 0.5; // distance to layer furthest away from source centers 


	const double depth = 1.; // depth at which cables are buried (center of trefoil intallation)

	const double d,
				drad;

	// location of cables 
	const int ncables = 6;
	const Matrix<double,2,1> cable_loc[6] = { {-cableradi, -depth - 0.58 * cableradi}, {-cableradi, depth + 0.58 * cableradi}, // source with opposing source 
											  {0., -depth + 1.15 * cableradi}, {0., depth - 1.15 * cableradi},
											  {cableradi, -depth - 0.58 * cableradi}, {cableradi, depth + 0.58 * cableradi} }; 

	// location of measurement devices
	const Matrix<double,2,1> measurement_loc[16] = { //{0.7, -1 + 2.15 * cableradi}, {0.6, -1 + 2.15 * cableradi}, {0.5, -1 + 2.15 * cableradi}, {0.4, -1 + 2.15 * cableradi},
													 //{0.3, -1 + 2.15 * cableradi}, {0.2, -1 + 2.15 * cableradi}, {0.1, -1 + 2.15 * cableradi}, {0., -1 + 2.15 * cableradi},
										   {-2.*cableradi, 0.0865+0.7 -depth-1.58*cableradi}, {-2*cableradi, 0.0865+0.6 -depth-1.58*cableradi}, {-2.*cableradi, 0.0865+0.5 -depth-1.58*cableradi},
										   																	 {-2.*cableradi, 0.0865+0.4 -depth-1.58*cableradi},
										   {-2.*cableradi, 0.0865+0.3 -depth-1.58*cableradi}, {-2.*cableradi, 0.0865+0.2 -depth-1.58*cableradi}, {-2.*cableradi, 0.0865+0.1 -depth-1.58*cableradi},
										   																	 {-2. * cableradi, 0.0865+0. -depth-1.58*cableradi} }; 


	Matrix<double,Dynamic,Dynamic> obsmat;


	//constructor
	innovation_normal_ll_cable1(int n_, int m_, const Matrix<double,Dynamic,Dynamic>& y_,
						 const Matrix<double,Dynamic,Dynamic>& exvar_) : n(n_), m(m_), y(y_), exvar(exvar_), obsmat(m_, 2*n_),
						 												 d(maxdepth/(n_-1)), drad(maxradi / (n_-1)) {

		/*
		s2 coordinate of distance of layer i is d * i
		*/


		// create observation matrix
		obsmat.setZero();

		// for every measurement device
		for (int j = 0; j < m; j++) {
			
			// add the entries of the heat from the air temperature 
			double dist = measurement_loc[j][1];  // depth of measurement device

			// every vertical location in soil
			// make sure that all devices are in the soil model


			int k = -dist / d;
			

			double rkk = (-dist - k * d) / d;
			double rk = 1. - rkk;

			obsmat(j,k) = rk;
			obsmat(j,k+1) = rkk;
			


			// for every source 
			for (int i = 0; i < ncables; i++) {

				auto diff = measurement_loc[j] - cable_loc[i];
				
				// distance between cable center to measurement device
				double dist = sqrt(diff.transpose() * diff);

				// corresponds to radial node 
				int k = dist / drad;
				// assert that the measurement location is within radial solution
				// if not then we cannot add it to observation matrix
				if (k >= n - 1)
					continue;

				double rkk = (dist - k * drad) / drad;
				double rk = 1. - rkk;

				// check if opposing cable (all cable locations with odd idx)
				if (i % 2 != 0) {
					rk = -rk;
					rkk = -rkk;
				}

				obsmat(j,n+k) += rk;
				obsmat(j,n+k+1) += rkk;

			}
		}


		// 

		//std::cout << obsmat.block(0,0,m,n) << "\n\n\n\n\n" << std::endl;

	}




	template<typename T>
	T operator ()(const Matrix<T,Dynamic,1>& param) const {  

		using stan::math::multiply;
		using stan::math::add;
		using stan::math::subtract;
		using stan::math::sum;

		using stan::math::log_determinant_spd;
		using stan::math::matrix_exp;

		using stan::math::abs;
		using stan::math::square;
		using stan::math::exp;


		// insert initial parameters

		Matrix<T,Dynamic,1> x0(2*n+1), xu(2*n+1), xc(2*n+1), e(m);
		Matrix<T,Dynamic,Dynamic> S0(2*n+1,2*n+1), Su(2*n+1,2*n+1), Sc(2*n+1,2*n+1), A(2*n+1,2*n+1),
								  Avert(n,n), Avertinv(n,n), Arad(n,n), Aradinv(n,n),
								  Q(2*n+1,2*n+1),
								  R(m,m), evar(m,m), evarinv(m,m), M(2*n+1,m),
							      F(2*n+1,2*n+1), Fvert(n,n), Gvert(n,n), Frad(n,n), Grad(n,n);
		// auxillary variables
		Matrix<T,Dynamic,1> q(n), q_marked(n), s(n), s_marked(n);



		Matrix<double,Dynamic,Dynamic> B(m,2*n+1), Bvert(m,n), Brad(m,n);
		Bvert.setZero();
		Bvert = obsmat.block(0,0,m,n); // vertical 
		Brad.setZero();
		Brad = obsmat.block(0,n,m,n); 

		B.setZero();
		B.block(0,0,m,n) = Bvert;


		R.setIdentity(); // observation noise
		R *= 1e-5*absf(param[3]); 



		x0.setZero(); // initial mean 


		// 
		S0.setIdentity(); 
		S0 *= 4.;


		// insert parameters 
		
		// A matrix
		Avert.setZero();

		// vertical solution
		Avert(0,0) = -2.; 
		Avert(0,1) = 1.;
		for (int i = 1; i < n-1; i++) {
			Avert(i,i) = -2.;
			Avert(i,i-1) = 1.;
			Avert(i,i+1) = 1.;
		}
		Avert(n-1,n-2) = 1.;
		Avert(n-1,n-1) = -2.; 


		// radial solution
		Arad.setZero();

		Arad(0,0) = -2.;
		Arad(0,1) = 2.;
		for (int i = 1; i < n-1; i++) {
			Arad(i,i) = -2;
			Arad(i,i-1) = (i*drad - 0.5*drad) / (i*drad);
			Arad(i,i+1) = (i*drad + 0.5*drad) / (i*drad);
		}
		Arad(n-1,n-2) = ((n-1)*drad - 0.5*drad) / ((n-1)*drad);
		Arad(n-1,n-1) = -2.; // ambient bc 
		
		



		// insert parameters

		A.setZero();

		Avert *= 1e-3*absf(param[0]) / (d*d); // thermal diff vertical
		Avert(0,0) = (-1e-3*absf(param[0]) - absf(param[4]))/(d*d); // param[4] 
		
		A.block(0,0,n,n) = Avert;

		A.block(0,n,n,n) = Avert; // noise block vertical



		A(0,n) = (-1e-3*absf(param[0]))/(d*d); // noise top vertical
		A(n-1,2*n-1) = -1e-3*absf(param[0])/(d*d); // bottom

		A.block(n,n,n,n).setIdentity(); // noise  
		A.block(n,n,n,n) *= -absf(param[2]); // -phi
		

		Arad *= 1e-3*absf(param[0]) / (drad*drad); // thermal diff radial



		// independent ou process at top soil layer

		A(0,2*n) = 1./d; 
		A(2*n,2*n) = -absf(param[11]);



		// state matrix 
		F.setZero();
		F = matrix_exp(A); // multiply by delta t



	 	// vertical solution
		Avertinv = stan::math::inverse(Avert);

		Fvert = matrix_exp(Avert);
		
		// convolution matrix 
		Gvert = subtract(multiply(Fvert, Avertinv), Avertinv);

		q.setOnes(); // vertical solution
		q *= 14.;

		// radial solution
		Aradinv = stan::math::inverse(Arad); 

		Frad = matrix_exp(Arad);

		Grad = subtract(multiply(Frad, Aradinv), Aradinv);

		s.setZero(); // radial solution




		// system noise
		Q.setZero();

		for (int i = 0; i < n; i++) { // 
			for (int j = 0; j <= i; j++) {
				Q(n+i,n+j) = absf(param[1]) * stan::math::exp(-1e-4*absf(param[5])*covar((i-j)*d));
							 //+ absf(param[18]) * stan::math::exp(-absf(param[19])*stan::math::square((i-j)*d)); // omega 
				Q(n+j,n+i) = Q(n+i,n+j);
			}
		} 

		Q(2*n,2*n) = absf(param[10]); // vertical top noise

		// Lyapunov approx
		Q = add(Q, multiply(F, multiply(Q, F.transpose())))*0.5; 


		// compute gaussian ll

		std::vector<T> acc; 

		xu = x0;
		Su = S0;
			
#ifdef writedata
		// write the values 
		std::ofstream fout("/home/oyvinda/Desktop/est.txt"),
					  fout_state("/home/oyvinda/Desktop/states.txt"),
					  fout_var("/home/oyvinda/Desktop/vars.txt"),
					  fout_diagnostics("/home/oyvinda/Desktop/scaledinnov.txt"); 
#endif
		for (int t = 0; t < y.cols(); t++) {

#ifdef writedata
			// write data 
			auto val = add(multiply(B, xu), add(multiply(Bvert,q), multiply(Brad,s)));
			for (int l = 0; l < m; l++) 
				fout << val(l,0) << ' '; 
			auto states = add(xu.block(0,0,n,1), q); 
			for (int l = 0; l < n; l++)
				fout_state << states(l,0) << ' '; 
			auto vars = multiply(B, multiply(Su, B.transpose()));
			for (int l = 0; l < m; l++)
				fout_var << vars(l,l) << ' ';
			fout << '\n';
			fout_state << '\n';
			fout_var << '\n'; 
#endif


			// external variables
			double current = exvar(1,t), // current in cable 
				   rainacc = exvar(3,t), // rainAccumulation
				   temperatureAir = exvar(4,t); // temperatureAir


			// perform filtering computations


			// compute the innovation variance
			evar = add(multiply(B, multiply(Su, B.transpose())), R);

			// compute its inverse 
			evarinv = evar.inverse(); // stan::math::inverse_spd


			// compute the projection matrix
			M = multiply(Su, multiply(B.transpose(), evarinv)); 


			// compute the innovation
			e = subtract(y.block(8,t,8,1), add(multiply(B, xu), add(multiply(Bvert, q), multiply(Brad, s)))); 


			// condition on observation
			xc = add(xu, multiply(M, e));

			// conditioned variance
			Sc = subtract(Su, multiply(M, multiply(B, Su)));



#ifdef writedataforecast 
			// forecast last half
			if (t > writedataforecast) {
				xc =xu;
				Sc =Su;
			} 
#endif 


			// compute the constant term

			// source term 
			q_marked.setZero();

			// vertical
			q_marked[0] = absf(param[4]) * temperatureAir / (d*d); // param[4] first BC for vertical solution
			//q_marked[0] += -absf(param[12]) / d; // for radiation out
			
			q_marked[n-1] = 1e-3*absf(param[0]) / (d*d) * (absf(param[6]) 
			+ absf(param[7])*stan::math::square(stan::math::cos(1.*3.14 * (t + (1e2*param[8])) / (24.*365.)) )); // second BC for vertical solution

			/*if (rainacc < 1.e-9 ) { // not significant
				q_marked[0] += (absf(param[13]) + absf(param[14]) * stan::math::square(stan::math::cos(1.*3.14 * (t+ 10.*24.) / (24.*365.)) ))
						*stan::math::square(stan::math::cos(1.*3.14 * (t+ 4.5) / 24.)) / d;
			} else {
				// cloudy 
				q_marked[0] += absf(param[15])*(absf(param[13]) + absf(param[14]) * stan::math::square(stan::math::cos(1.*3.14 * (t+ 10.*24) / (24.*365.)))) 
						* stan::math::square(stan::math::cos(1.*3.14 * (t+ 4.5) / 24.)) / d;
				
			} */

			// radial
			s_marked.setZero();
			s_marked[0] = 1.e-6*absf(param[9])*current*current / (drad*drad); // source 
			// BC is 0 



			q = add(multiply(Fvert, q), multiply(Gvert, q_marked)); // compute constant term
			s = add(multiply(Frad, s), multiply(Grad, s_marked)); // radial solution 

			// forward
			xu = multiply(F, xc); // add the constant term
			Su = add(multiply(F, multiply(Sc, F.transpose())), Q);

			


			// add to the log likelihood
			if (t > 0) 
				acc.push_back(stan::math::log_determinant(evar) + multiply(e.transpose(), multiply(evarinv, e)));


#ifdef writedata
			// write diagnostics data

			auto scaled_innovation = multiply( Matrix<double,8,8>(stan::math::value_of(evarinv)).sqrt(), e); // matrix sqrt is in unsupported Eigen
			for (int l = 0; l < m; l++) {
				fout_diagnostics << scaled_innovation(l,0) << '\n';
			}

#endif

		}

#ifdef writedata
		fout.close();
		fout_state.close();
		fout_var.close();
		fout_diagnostics.close(); 
#endif
		
		return sum(acc);
	}

};
