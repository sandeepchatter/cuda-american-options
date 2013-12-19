#ifndef _REGRESS_CPU_H_
#define _REGRESS_CPU_H_

#include <stdlib.h>


#include "./regress_CPU.h"
#include "./LinearAlgebraSubroutines.h"

class Linear_Regression
{
	LinearAlgebraSubroutines las;		// linear algebra subroutines
	
	public:
	
	//int num_terms;					// number of terms in regression
	//vector<float> x;
	//vector<float> y;
	
	// function to find coefficients of linear regression of y on x
	// with Laguerre polynomials as the basis functions
	// nlp: number of Laguerre polynomials to be used, max 5
	void find_linear_fit( vector<float> &x, vector <float> &y, vector<float> &g, int nlp);
	
	// uses a constant and 'nlp' Laguerre polynomials as the basis functions
	// the basis then has a size [x.size()x(nlp+1)], where first column is all ones   
	void get_basis_lg( vector<float> &x, int num_terms, vector< vector<float> > *x_base);
	
	void get_basis_ch( vector<float> &x, int num_terms, vector< vector<float> > *x_base);
	
	// returns the Laguerre polynomial L(index). 
	float get_Laguerre_Polynomial(float x, int index);
	
	// returns the Chebyshev polynomial L(index). 
	float get_Chebyshev_Polynomial(float x, int index);
	
	void test_function( float val){ printf("In Linear_Regression::test_function() with value %g\n", val); }
};


#endif
