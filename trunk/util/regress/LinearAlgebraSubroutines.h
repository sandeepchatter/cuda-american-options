#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <math.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

struct solution_table
{
	float norm_rk;
	vector<float> x;
};

class LinearAlgebraSubroutines
{
	vector<solution_table> solution;
	
	int max_CG_iterations;
	
	public:
	LinearAlgebraSubroutines();
	
	bool matrixVectorMutiply( vector< vector<float> > &mat, vector<float> &vec, vector<float> *prod );
	
	bool matrixTransposeVectorMutiply( vector< vector<float> > &mat, vector<float> &vec, vector<float> *prod );
	
	bool matrixTransposeMatrixMult( vector< vector<float> > &mat, vector< vector<float> > *mTm );
	
	bool dotProduct( vector<float> &u, vector<float> &V, float &prod);
	
	//template<class T>
	void print_vector(vector<float> &V, string name);
	
	//template<class T>
	void print_matrix(vector< vector <float> > &V, string name);
	
	bool preconditioned_CG( vector< vector <float> > &A, vector<float> &b, vector<float> &x);
	
	float norm( vector<float> &v, int which_norm );
	
	void test_function( float val){ printf("In LinearAlgebraSubroutines::test_function() with value %g\n", val); }
};

#endif
