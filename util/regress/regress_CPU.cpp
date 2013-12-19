#include "./regress_CPU.h"

void Linear_Regression::find_linear_fit( vector<float> &x, vector <float> &y, vector<float> &g, int nlp)
{
	// maximum Laguerre_Polynomial used is L4
	nlp = (nlp > 5)? 5 : nlp;
	
	vector< vector<float> > X;
	//get_basis_lg( x, nlp, &X);
	get_basis_ch( x, nlp, &X);
	//las.print_matrix(X, "base_matrix");
	
	vector< vector<float> > A;
	las.matrixTransposeMatrixMult( X, &A );	// A = X'*X	
	//las.print_matrix(A, "A = X'*X is");
	
	vector <float> b;
	las.matrixTransposeVectorMutiply( X, y, &b );	// b = X'*y
	//las.print_vector(b, "b = X'*y is");
	
	// solve A*g = b for g
	las.preconditioned_CG( A, b, g);
}

/*
sample basis functions for nlp = 3 (i.e. we include L0, L1 and L2):
y[i] = g_c + exp(-x[i]/2)*(g_0*1 + g_1*(1-x[i]) + g_2*(1 - 2*x[i] - pow(x[i],2)/2))
where [g_c, g_0, g_1, g_2] are to be determined
*/
void Linear_Regression::get_basis_lg( vector<float> &x, int nlp, vector< vector<float> > *x_base)
{
	(*x_base).resize( x.size() );
	for ( int i = 0; i < (*x_base).size(); i++ )
	{
		(*x_base)[i].resize(nlp+1);
		(*x_base)[i][0] = 1;
	}
	
	float weight = 0;
	
	for ( int row = 0; row < (*x_base).size(); row++ )
	{
		weight = exp( -x[row]/2.0);
		for (int col = 1; col < (*x_base)[row].size(); col++ )
		{
			(*x_base)[row][col] = weight*get_Laguerre_Polynomial( x[row], col-1 );
			// implement error checking here
		}
	}
}

// nlp here means number of terms after the first constant T_0, i.e we use chebyshev polynomials
// T_0 = 1, T_1 = x .. upto T_nlp. For ex, nlp = 3 means we use base vector of size 4.
void Linear_Regression::get_basis_ch( vector<float> &x, int nlp, vector< vector<float> > *x_base)
{
	(*x_base).resize( x.size() );
	for ( int i = 0; i < (*x_base).size(); i++ )
	{
		(*x_base)[i].resize(nlp+1);
	}
	
	for ( int row = 0; row < (*x_base).size(); row++ )
	{
		for (int col = 0; col < (*x_base)[row].size(); col++ )
		{
			(*x_base)[row][col] = get_Chebyshev_Polynomial( x[row], col );
			// implement error checking here
		}
	}
}

float Linear_Regression::get_Laguerre_Polynomial(float x, int index)
{
	switch(index)
	{
		case 0: return 1; 
		case 1: return (-x + 1);
		case 2: return (1 - 2*x + x*x/2);
		case 3: return (-1*pow(x,3) + 9*pow(x,2) - 18*x + 6)/6;
		case 4: return (pow(x,4) - 16*pow(x,3) + 72*pow(x,2) - 96*x + 24)/24; 
	}
	return -1;
}

float Linear_Regression::get_Chebyshev_Polynomial(float x, int index)
{
	switch(index)
	{
		case 0: return 1; 
		case 1: return x;
		default: return 2*x*get_Chebyshev_Polynomial(x, index-1)-get_Chebyshev_Polynomial(x, index-2);
	}
	return -1;
}

/*
int main()
{
	Linear_Regression least_squares;
	
	//vector<float> x(7);
	//x[0] = 0.821925; x[1] = 1.0202; x[2] = 0.870332; x[3] = 0.806952; x[4] = 0.654273; x[5] = 0.747412; x[6] = 0.806047;
	//vector<float> x(8);
	//x[0] = 1.02687; x[1] = 0.911153; x[2] = 0.943797; x[3] = 0.813402; x[4] = 0.676303; x[5] = 0.816441; x[6] = 0.944745; x[7] = 0.924859;
	vector<float> x(6);
	x[0] = 0.834108; x[1] = 0.87468; x[2] = 0.986261; x[3] = 0.715828; x[4] = 0.982273; x[5] = 0.983835; 

	//vector<float> y(7);
	//y[0] = 0.333415; y[1] = 0; y[2] = 0.170238; y[3] = 0.270117; y[4] = 0.446454; y[5] = 0.293145; y[6] = 0.195586;
	//vector<float> y(8);
	//y[0] = 0.22528; y[1] = 0.0294352; y[2] = 0.177466; y[3] = 0.24007; y[4] = 0.390878; y[5] = 0.298881; y[6] = 0.240964; y[7] = 0;
	vector<float> y(6);
	y[0] = 0.137146; y[1] = 0.104902; y[2] = 0.233699; y[3] = 0.369118; y[4] = 0.230697; y[5] = 0.103966; 
	
	vector<float> g;
	least_squares.find_linear_fit( x, y, g, 3);
	
	printf( "\n* Linear least Squares fit with [X,Y]= [S(t),Discounted]:\n");
	//printf("* y[i] = %.6f*1 + %.6f*x[i] + %.6f*(2*x[i]*x[i]-1) + %.6f*(4*pow(x[i],3) - 3*x[i])",
	//	   g[0], g[1], g[2], g[3]); y - (20.182110*1 + -30.693182.*x + 13.208112*(2.*x.*x-1) + -2.659580*(4.*x.*x.*x - 3.*x))
	printf("* y = %.6f*1 + %.6f.*x + %.6f*(2.*x.*x-1) + %.6f*(4.*x.*x.*x - 3.*x)",
	g[0], g[1], g[2], g[3]);
}//*/
