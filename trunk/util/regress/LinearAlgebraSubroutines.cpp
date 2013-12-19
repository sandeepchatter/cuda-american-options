#include <stdlib.h>
#include <string>
#include <iostream>
#include "LinearAlgebraSubroutines.h"

LinearAlgebraSubroutines::LinearAlgebraSubroutines()
{
	max_CG_iterations = 100;
	solution.resize(max_CG_iterations+1);
}

bool LinearAlgebraSubroutines::matrixVectorMutiply( vector< vector<float> > &mat, vector<float> &vec, vector<float> *prod )
{
	if ( mat[0].size() != vec.size() )
	{
		fprintf(stderr, "number of cols in matrix != number of rows in vector, Error!\n");
		return 0; 
	}
	
	(*prod).resize( mat.size() );
	
	for (int row = 0; row < mat.size(); row++)
	{
		(*prod)[row] = 0;
		for (int col = 0; col < vec.size(); col++)
		{
			(*prod)[row] += mat[row][col]*vec[col]; 
		}
	}
	
	return 1;
}

bool LinearAlgebraSubroutines::matrixTransposeVectorMutiply( vector< vector<float> > &mat, vector<float> &vec, vector<float> *prod )
{
	if ( mat.size() != vec.size() )
	{
		fprintf(stderr, "number of rows in matrix != number of rows in vector, Error!\n");
		return 0; 
	}
	
	(*prod).resize( mat[0].size() );
	
	for (int col = 0; col < mat[0].size(); col++)
	{
		(*prod)[col] = 0;
		for (int row = 0; row < vec.size(); row++)
		{
			//printf("(*prod)[%d] += mat[%d][%d]*vec[%d]\n", row, row, col, row);
			(*prod)[col] += mat[row][col]*vec[row]; 
		}
	}
	
	return 1;
}

bool LinearAlgebraSubroutines::matrixTransposeMatrixMult( vector< vector<float> > &mat, vector< vector<float> > *mTm )
{
	(*mTm).resize( mat[0].size() );
	for ( int i = 0; i < (*mTm).size(); i++ )
	{
		(*mTm)[i].resize( mat[0].size() );
		for ( int j = 0; j < mat[0].size(); j++ )
		{
			(*mTm)[i][j] = 0;
		}
	}
	
	for (int rowT = 0; rowT < mat[0].size(); rowT++)
	{
        for (int col = 0; col < mat[0].size(); col++)
        {
            // Multiply the row of A by the column of B to get the row, column of product.
            for (int colT = 0; colT < mat.size(); colT++)
            {
            	//printf("(*mTm)[%d][%d] += mat[%d][%d] * mat[%d][%d]\n", rowT, col, rowT, colT, colT, col);
                (*mTm)[rowT][col] += mat[colT][rowT] * mat[colT][col];
            }
            //printf("----------\n");
        }
    }	
	return 1;
}

bool LinearAlgebraSubroutines::dotProduct( vector<float> &u, vector<float> &v, float &prod)
{
	if ( u.size() != v.size() )
	{
		fprintf(stderr, "number of cols in 1st vector != number of rows in 2nd vector, Error!\n");
		return 0; 
	}
	
	prod = 0;
	
	for (int i = 0; i < u.size(); i++)
	{
		prod += u[i]*v[i];
	}
	
	return 1;
}


float LinearAlgebraSubroutines::norm( vector<float> &v, int which_norm )// 0- infinity, 1-sum-norm, 2-eucledian
{
	if ( which_norm == 0 )
	{
		float max = -1*pow(10, 8);
		for( int u = 0; u < v.size(); u++ )
		{
			if ( max < fabs( v[u] ) )
				max = fabs( v[u] );
		}
		return max;
	}
	else if ( which_norm == 1 )
	{
		float sum = 0;
		for( int u = 0; u < v.size(); u++ )
		{
			sum = sum + fabs( v[u] );
		}
		return sum;
	}
	else if ( which_norm == 2 )
	{
		float sum = 0;
		for( int u = 0; u < v.size(); u++ )
		{
			sum = sum + v[u]*v[u];
		}
		return sqrt(sum);
	}
}

bool LinearAlgebraSubroutines::preconditioned_CG( vector< vector <float> > &A, vector<float> &b, vector<float> &x)
{
	if ( A.size() != A[0].size() || A.size() != b.size() )
	{
		fprintf(stderr, "Cannot solve the linear system, it is inconsistent!!!\n");
		return 0;
	}
	
	int n = A.size();
	x.resize( n );
	
	for ( int i = 0; i < solution.size(); i++ )
	{
		solution[i].x.resize(n);
	}	
	
	vector<float> M( n );
	for(int q = 0; q < n; q++)
		M[q] = A[q][q];
	
	vector<float> rk(n);
	vector<float> yk(n);
	vector<float> pk(n);
	
	matrixVectorMutiply(A, x, &rk);
	
	for (int q = 0; q<rk.size(); q++)
	{
		rk[q] = rk[q] - b[q];
		yk[q] = rk[q]/M[q];
		pk[q] = -yk[q];
	}
	
	float alpha_k = 0;
	float beta_k = 0;				
			
	//Helper values			
	float rk_yk = 0; 
	float rk_yk_new = 0;
	vector<float> Apk(n);
	
	//finding rk'*yk
	dotProduct( rk, yk, rk_yk);
	
	// LOOP	
	int idx = 0;
	while (true)
	{
		//finding A*pk
		matrixVectorMutiply(A, pk, &Apk);
		float pkApk = 0;
		
		//finding pkApk
		dotProduct( pk, Apk, pkApk);	
		
		alpha_k = rk_yk/pkApk;
		
		//Next guess
		for (int q = 0; q< x.size(); q++)
			x[q] = x[q] + alpha_k*pk[q];
		
		//New residual
		for (int q = 0; q< n; q++)
			rk[q] = rk[q] + alpha_k*Apk[q];
		
		//Check stopping criteria
		float norm_rk = norm(rk, 0);
		if ( norm_rk <= pow(10, -8))
		{
			//printf("\nSuccessful in finding solution at iteration  %d with norm: %g",
			//idx+1, norm_rk);
			break;
		}
		// we do not clear solution table vector because it is used only when its capacity is fulfilled, 
		// which means that all its entries have been refilled 
		solution[idx].norm_rk = norm_rk;
		for( int fg = 0; fg < n; fg++ )
			solution[idx].x[fg] = x[fg];
		
		if ( idx == max_CG_iterations )
		{
			int min_index = 0;
			float min_norm = 1000;
			for ( int p = 0; p <= idx; p++  )
			{
				if ( solution[p].norm_rk < min_norm )
				{
					min_norm = solution[p].norm_rk;
					min_index = p;
				}
			}
			for( int fg = 0; fg < n; fg++ )
				x[fg] = solution[min_index].x[fg];
			
			printf("\nWARNING: Exceeded max iteration count of %d. Returning best solution found, with norm: %g",
			max_CG_iterations, solution[min_index].norm_rk);
			break;
		}
		
		// solve for new y
		for (int q = 0; q < n; q++)
			yk[q] = rk[q]/M[q];
		
		//finding rk+1'*yk+1
		dotProduct( rk, yk, rk_yk_new);
		
		//finding beta_k
		beta_k = rk_yk_new/rk_yk;
		
		//updating rk_yk
		rk_yk = rk_yk_new;
		
		//updating pk
		for (int q = 0; q < pk.size(); q++)
			pk[q] = -yk[q] + beta_k*pk[q];
	
		idx++;
	}
}

//template<class T>
void LinearAlgebraSubroutines::print_vector(vector<float> &V, string name)
{
	cout<<name<<":- ";
	for (int i = 0; i<V.size(); i++)
		cout<<V[i]<<", ";
	if ( V.size() == 0 )
		cout<<"[NULL]";
	cout<<endl;
}

//template<class T>
void LinearAlgebraSubroutines::print_matrix(vector< vector <float> > &V, string name)
{
	cout<<endl<<name<<":- "<<endl;
	for (int i = 0; i<V.size(); i++)
	{
		for (int j = 0; j < V[i].size(); j++)
			printf("%10g, ", V[i][j]);
		cout<<endl;
	}	
	cout<<endl;
}

/*
For matlab testing:

 Set 1:
A = [ -0.2050    0.6715    1.0347
   -0.1241   -1.2075    0.7269
    1.4897    0.7172   -0.3034
    1.4090    1.6302    0.2939
    1.4172    0.4889   -0.7873];
    
v = [0.8884
   -1.1471
   -1.0689
   -0.8095
   -2.9443];
   
 Set 2:
A = [1.4384   -0.2414    0.6277   -1.1135
    0.3252    0.3192    1.0933   -0.0068
   -0.7549    0.3129    1.1093    1.5326
    1.3703   -0.8649   -0.8637   -0.7697
   -1.7115   -0.0301    0.0774    0.3714
   -0.1022   -0.1649   -1.2141   -0.2256];

v = [1.1174   -1.0891    0.0326    0.5525    1.1006    1.5442]';

u = [0.0859   -1.4916   -0.7423   -1.0616    2.3505   -0.6156]';

w = [0.4882   -0.1774   -0.1961    1.4193]';
*/

/* testing main function()*----
int main()
{
	vector< vector<float> > A(6);
	for (int i = 0; i < A.size(); i++)
	{
		//A[i].push_back();
		A[i].resize(4);
	}
	A[0][0]= 1.4384;  A[0][1]=-0.2414;  A[0][2]= 0.6277;  A[0][3]=-1.1135;
	A[1][0]= 0.3252;  A[1][1]= 0.3192;  A[1][2]= 1.0933;  A[1][3]=-0.0068; 
	A[2][0]=-0.7549;  A[2][1]= 0.3129;  A[2][2]= 1.1093;  A[2][3]= 1.5326; 
	A[3][0]= 1.3703;  A[3][1]=-0.8649;  A[3][2]=-0.8637;  A[3][3]=-0.7697;
	A[4][0]=-1.7115;  A[4][1]=-0.0301;  A[4][2]= 0.0774;  A[4][3]= 0.3714;
	A[5][0]=-0.1022;  A[5][1]=-0.1649;  A[5][2]=-1.2141;  A[5][3]=-0.2256;

	vector<float> v(6);
	v[0]=1.1174;  v[1]=-1.0891;  v[2]=0.0326;  v[3]=0.5525; v[4]=1.1006; v[5]=1.5442;

	vector<float> u(6);
	u[0]=0.0859;  u[1]=-1.4916;  u[2]=-0.7423;  u[3]=-1.0616; u[4]=2.3505; u[5]=-0.6156;
	
	vector<float> w(4);
	w[0]=0.4882;  w[1]=-0.1774;  w[2]=-0.1961;  w[3]=1.4193;

	vector<float> prod;
	
	LinearAlgebraSubroutines linalg;
	
	printf("Using the following matrices:\n--------------\n");
	linalg.print_matrix(A, "6x4 matrix A");
	linalg.print_vector(v, "6x1 Vector v");
	linalg.print_vector(u, "6x1 Vector u");
	linalg.print_vector(w, "4x1 Vector w");
	printf("----------\n");
	
	linalg.matrixVectorMutiply( A, w, &prod );
	linalg.print_vector(prod, "\nproduct A*w");
	
	linalg.matrixTransposeVectorMutiply( A, v, &prod );
	linalg.print_vector(prod, "\nproduct A(T)*v");
	
	vector< vector<float> > mTm;
	linalg.matrixTransposeMatrixMult( A, &mTm );
	linalg.print_matrix(mTm, "product A(T)*A");
	
	float dot_prod;
	linalg.dotProduct( u, v, dot_prod);
	printf("Dot product (u*v): %g\n", dot_prod);
	
	vector<float> x(4);
	//x[0]=1;  x[1]=1;  x[2]=1;  x[3]=1;
	
	linalg.preconditioned_CG( mTm, w, x);
	linalg.print_vector(x, "\nSolving linear system inv(mTm)*w");
	
}*/
