#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include <curand.h>
#include "option_kernel.h"
#include "../cuPrintf.cu"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>


template<typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const{
            return x*x;
        }
};


__device__ float phi(float x) {
    return 0.5*(1 + erf(x/sqrtf(2)));
}

__device__ float get_black_scholes_continuation_value_gpu(float x, float time, int height, InputData indata ) {
    float del_t = indata.expiry_time/(height-1)/365;
    float t = time*del_t;
//    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float d1, d2, den;
    float ttm = (indata.expiry_time - t)/365;
    d1 = log(x/indata.strike_price) + ( indata.discount_rate + 0.5*indata.volatility*indata.volatility )*ttm;
    d2 = log(x/indata.strike_price) + ( indata.discount_rate - 0.5*indata.volatility*indata.volatility )*ttm;
    den = indata.volatility*sqrtf( ttm );
    d1 = d1/den;
    d2 = d2/den;

    return indata.strike_price*exp(-1*indata.discount_rate*ttm)*phi(-1*d2) - x*phi(-1*d1);
    // cuPrintf("htid[%d] = %f\n", tid, h[tid]);
    //printf("d1: %g, d2: %g, den: %g, phi(-1*d2): %g, phi(-1*d1): %g, h[i]: %g, x[i]: %g\n", d1, d2, den, phi(-1*d2), phi(-1*d1), h[i], x[i]);
}

static __global__ void generate_asset_price_paths_and_cash_flow_reg(float *option_value_eu, float *option_value_am, int width, int height, InputData indata, float *norm_sample) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int pathid = blockDim.x * blockIdx.x + threadIdx.x;

	
	int ts_am = height-1;
	int oeb = ts_am;		// optimal_exercise_boundary for this path
	float cf_am = 0;		// cash flow of american option for this path
	
	// the following should be single read and broadcast for all threads I hope
	float spot_price = indata.S_0;
	float strike_price = indata.strike_price;
	float expiry_time = indata.expiry_time;
	float discount_rate = indata.discount_rate;
	float volatility = indata.volatility;
	
	//if ( pathid == 256  )
	//		cuPrintf ("\nAt t=0, spot_price = %g, strike_price = %g\n", spot_price, strike_price);
			
	// for decicing optimal exercise boundary in american options
	float put_value = 0;
	float h = 0;
	float time = 0;
	float d1, d2, den, ttm;
	
	// assuming uniformaly distributed option exercise times
	float drift = discount_rate - indata.dividend - 0.5*pow(volatility,2);
    float del_t = expiry_time/ts_am/365;
    float sigma = sqrt(del_t)*volatility; 
	
	int k = 0;
	int nt = blockDim.x;
	int start_index = bid*nt*ts_am;

	// 19 float/int register variables so far
	
	while ( k < ts_am ) 
	{
		// NOTE: (k < oeb) should be a good stopping criteria, stop computation as soon as
		// you exercise, but can lead to highly divergent code paths.  
		
		int index = start_index + k*nt + tid;
		spot_price = spot_price*exp(drift*del_t + sigma*norm_sample[index]);
		
		//if (index > width*height)
		//	cuPrintf ("index = %d for start_index = %d, k = %d, nt = %d, tid= %d\n", index, start_index, k, nt, tid );
		
		put_value = fmaxf( strike_price - spot_price, 0.0); //put
		
		//=======================Black-scholes continuation value========================//
		int kt = k+1;
		time = kt*del_t; 			// is the current time
		
		//if ( pathid == 256 )
		//	cuPrintf ("At t = %g, put value = %g; spot_price = %g, strike_price = %g, index = %d and normrand = %g\n",
		//	time, put_value, spot_price, strike_price, index, norm_sample[index]);
		
		ttm = (expiry_time - time)/365;
		d1 = log(spot_price/strike_price) + ( discount_rate + 0.5*volatility*volatility )*ttm;
		d2 = log(spot_price/strike_price) + ( discount_rate - 0.5*volatility*volatility )*ttm;
		den = volatility*sqrtf( ttm );
		d1 = d1/den;
		d2 = d2/den;

		h = strike_price*exp(-1*discount_rate*ttm)*phi(-1*d2) - spot_price*phi(-1*d1);
		//===============================================================================//
		if ( oeb > kt & put_value > h )
		{
		    oeb = kt;
		    cf_am = fmaxf( strike_price - spot_price, 0.0);
		}
		
		k++;
		//if ( pathid == 0 )
		//	cuPrintf ("----------------------\n");
	}
	
	
    option_value_eu[ pathid ] = put_value*exp(-1*discount_rate*expiry_time/365 );
	option_value_am[ pathid ] = cf_am*exp(-1*discount_rate*oeb*del_t ); 
	
}

// incomplete
static __global__ void generate_asset_price_paths_and_cash_flow_reg_mp(float *option_value_eu, float *option_value_am, int width, int height, InputData indata, float *norm_sample)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int ts_am = height-1;
	
	// the following should be single read and broadcast for all threads I hope
	float spot_price = indata.S_0;
	float strike_price = indata.strike_price;
	float expiry_time = indata.expiry_time;
	float discount_rate = indata.discount_rate;
	float volatility = indata.volatility;
	
	//if ( pathid == 256  )
	//		cuPrintf ("\nAt t=0, spot_price = %g, strike_price = %g\n", spot_price, strike_price);
			
	// for decicing optimal exercise boundary in american options
	float put_value = 0;
	float h = 0;
	float time = 0;
	float d1, d2, den, ttm;
	
	// assuming uniformaly distributed option exercise times
	float drift = discount_rate - indata.dividend - 0.5*pow(volatility,2);
    float del_t = expiry_time/ts_am/365;
    float sigma = sqrt(del_t)*volatility; 
	
	int m = 1;
	int nt = blockDim.x;
	int start_index = 4*bid*nt*ts_am;
	int pathid = 4*blockDim.x * blockIdx.x + threadIdx.x;
	
	
	// 19 float/int register variables so far
	while( m <= 4)
	{
		int oeb = ts_am;		// optimal_exercise_boundary for this path
		float cf_am = 0;		// cash flow of american option for this path
		
		int k = 0;

		while ( k < ts_am ) 
		{
			// NOTE: (k < oeb) should be a good stopping criteria, stop computation as soon as
			// you exercise, but can lead to highly divergent code paths.  
		
			int index = start_index + k*nt + tid;
			spot_price = spot_price*exp(drift*del_t + sigma*norm_sample[index]);
		
			//if (pathid == 100000)
			//	cuPrintf ("pathid = %d for start_index = %d, k = %d, nt = %d, tid= %d, m = %d\n", pathid, start_index, k, nt, tid, m );
		
			put_value = fmaxf( strike_price - spot_price, 0.0); //put
		
			//=======================Black-scholes continuation value========================//
			int kt = k+1;
			time = kt*del_t; 			// is the current time
		
			//if ( tid == 0 || tid == 255)
			//	cuPrintf ("At t = %g, put value = %g; spot_price = %g, strike_price = %g, index = %d and normrand = %g\n",
			//	time, put_value, spot_price, strike_price, index, norm_sample[index]);
		
			ttm = (expiry_time - time)/365;
			d1 = log(spot_price/strike_price) + ( discount_rate + 0.5*volatility*volatility )*ttm;
			d2 = log(spot_price/strike_price) + ( discount_rate - 0.5*volatility*volatility )*ttm;
			den = volatility*sqrtf( ttm );
			d1 = d1/den;
			d2 = d2/den;

			h = strike_price*exp(-1*discount_rate*ttm)*phi(-1*d2) - spot_price*phi(-1*d1);
			//===============================================================================//
			if ( oeb > kt & put_value > h )
			{
				oeb = kt;
				cf_am = fmaxf( strike_price - spot_price, 0.0);
			}
		
			k++;
		}
		
		
		//if ( tid == 0 )
		//	cuPrintf ("----------------------\n");
		m++;
		
		option_value_eu[ pathid ] = put_value*exp(-1*discount_rate*expiry_time/365 );
		option_value_am[ pathid ] = cf_am*exp(-1*discount_rate*oeb*del_t ); 
		
		start_index = start_index + nt*ts_am;
		pathid = pathid + nt;
	}
	
    
	
}

static __global__ void generate_asset_price_paths_and_cash_flow(float *S, float *cash_flow, float *option_value, int width, int height, InputData indata, float *norm_sample) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //InputData indata = inputdata;
    // shared memory to make sure accesses are fast (not sure if no use of tid affects things)

    //curandState state;

    //curand_init(indata.random_seed, tid, 0, &state);

    float drift = indata.discount_rate - indata.dividend - 0.5*pow(indata.volatility,2);
    float del_t = indata.expiry_time/(height-1)/365;
    float sigma = sqrt(del_t)*indata.volatility;
    S[tid*height] = indata.S_0;
    float temp = indata.S_0;
    for (int j = 1; j < height; j++ )
    {
	    S[tid*height+j] = temp = temp*exp(drift*del_t + sigma*norm_sample[tid*height+j]);
    }
    
    int expiry_index = height-1;
    // at the expiry time, the only choice is to exercise the option
    float discount_eu = exp(-1*indata.discount_rate*indata.expiry_time/365 );
	float cash_temp;
    cash_flow[tid] = cash_temp = fmaxf(indata.strike_price - S[tid*height+expiry_index], 0.0); //put
    option_value[tid] = cash_temp*discount_eu;

    //__syncthreads();

}

static __global__ void find_optimal_exercise_boundary_gpu(float *S, float *cash_flow, float *option_value, int width, int height,
														  InputData indata, float *x, float *h, int *optimal_exercise_boundary, float *cash_flow_am) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int expiry_index = height-1;

    //InputData indata = inputdata;

    float del_t = indata.expiry_time/(height-1)/365;
    // discount for merican counterpart
    float discount = exp(-1*indata.discount_rate*del_t );

    float put_value = 0;
	
	optimal_exercise_boundary[tid] = expiry_index;
	float xtemp = 0;
	float htemp = 0;
    // for all other times when the option can be exercised, we comapre the
    // value of exercising and continuation value to find optimal exercise boundary  
    for ( int time = expiry_index-1; time >= 1; time-- ) // move back in time
    {
        put_value = fmaxf( indata.strike_price - S[tid*height+time], 0.0); //put

        xtemp = S[tid*height+time];
        cash_flow[tid] = put_value;

        htemp = get_black_scholes_continuation_value_gpu(xtemp, time, height, indata);

        if ( cash_flow[tid] > htemp )
        {
            optimal_exercise_boundary[tid] = time;
            cash_flow_am[tid] = fmaxf(indata.strike_price - S[tid*height+time], 0.0);
        }
    }
	
	
    //cash_flow_am[tid] = fmaxf(indata.strike_price - S[tid*height+optimal_exercise_boundary[tid]], 0.0); 
    //cash_flow_am[tid] = 0;
    discount = exp(-1*indata.discount_rate*optimal_exercise_boundary[tid]*del_t );
    option_value[tid] = cash_flow_am[tid]*discount;//*/ 

}

static __global__ void generate_asset_price_paths_and_cash_flow_multiple_paths(float *S, float *cash_flow, float *option_value, int width, int height, InputData indata, float *norm_sample) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float drift = indata.discount_rate - indata.dividend - 0.5*pow(indata.volatility,2);
    float del_t = indata.expiry_time/(height-1)/365;
    float sigma = sqrt(del_t)*indata.volatility;
    S[tid*height] = indata.S_0;
    float temp  = indata.S_0;
    float temp1 = temp;
    float temp2 = temp;
    float temp3 = temp;
    for (int j = 1; j < height; j++ )
    {
	    S[(4*tid  )*height+j] = temp  = temp *exp(drift*del_t + sigma*norm_sample[(4*tid  )*height+j]);
	    S[(4*tid+1)*height+j] = temp1 = temp1*exp(drift*del_t + sigma*norm_sample[(4*tid+1)*height+j]);
	    S[(4*tid+2)*height+j] = temp2 = temp2*exp(drift*del_t + sigma*norm_sample[(4*tid+2)*height+j]);
	    S[(4*tid+3)*height+j] = temp3 = temp3*exp(drift*del_t + sigma*norm_sample[(4*tid+3)*height+j]);
    }
    
    int expiry_index = height-1;
  
    // at the expiry time, the only choice is to exercise the option
    float discount_eu = exp(-1*indata.discount_rate*indata.expiry_time/365 );
	float cash_temp;
    cash_flow[(4*tid)] = cash_temp = fmaxf(indata.strike_price - S[(4*tid  )*height+expiry_index], 0.0); //put
    option_value[(4*tid)] = cash_temp*discount_eu;
    
    cash_flow[(4*tid+1)] = cash_temp = fmaxf(indata.strike_price - S[(4*tid+1)*height+expiry_index], 0.0); //put
    option_value[(4*tid+1)] = cash_temp*discount_eu;
    
    cash_flow[(4*tid+2)] = cash_temp = fmaxf(indata.strike_price - S[(4*tid+2)*height+expiry_index], 0.0); //put
    option_value[(4*tid+2)] = cash_temp*discount_eu;
    
    cash_flow[(4*tid+3)] = cash_temp = fmaxf(indata.strike_price - S[(4*tid+3)*height+expiry_index], 0.0); //put
    option_value[(4*tid+3)] = cash_temp*discount_eu;

}

void checkError(cudaError_t err) {

    if (err != cudaSuccess) {
        fprintf(stderr, "cuda function failed (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

//Main computations
extern "C" void generate_and_find_exercise_boundary()
{
	printf( "\nGPU COMPUTATION\n=============================\n");
	
    InputData h_indata;
    // read the input file for options relating to the number of paths, number
    // of discrete time-steps etc. 
    FileIO fileIO;
    fileIO.readInputFile((char*)"./input/options.txt", h_indata);

    float GPU_t = 0;
    // allocate memory to store all Monte Carlo paths, and intialize
    // the initial value of the asset at t=0.
    int num_paths = (h_indata.num_paths%2 == 0)?h_indata.num_paths:h_indata.num_paths+1;  

    float *d_S = NULL;
    float *d_x = NULL;
    float *d_h = NULL;
    float *d_cash_flow = NULL;
    float *d_option_value = NULL;
    float *d_option_value_am = NULL;
    float *d_cash_flow_am = NULL;
    
    int *d_optimal_exercise_boundary = NULL;
    
    int width = num_paths;
    int height = h_indata.num_time_steps+1;
    //size_t size = num_paths*(h_indata.num_time_steps+1)*sizeof(float);
    //printf("width=%d\n", width);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
	
	cudaEvent_t startt, stopt;
    cudaEventCreate(&startt);
    cudaEventCreate(&stopt);
    cudaEventRecord(startt,0);
    
    checkError(cudaMalloc((void**)&d_S, width*sizeof(float)*height));
    checkError(cudaMalloc((void**)&d_x, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_h, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_cash_flow, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_option_value, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_option_value_am, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_cash_flow_am, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_optimal_exercise_boundary, width*sizeof(int)));
	
	cudaEventRecord(stopt,0);
    cudaEventSynchronize(stopt);
    cudaEventElapsedTime(&GPU_t, startt, stopt);
	printf("\n### GPU: Time to cudamalloc: %fs\n", GPU_t/1000);
	
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)ceil( 0.25*width/threadsPerBlock);

    printf("	- Blocks per Grid = %d\n", blocksPerGrid);
    printf("	- Threads per Block = %d\n", threadsPerBlock);
    
    random_normal normrnd;
    normrnd.zigset( 78542121 );

	size_t size_norm = width*height*sizeof(float);
    float *h_norm_sample = (float *) malloc(size_norm);

    /*for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {    
            h_norm_sample[i*height+j] = normrnd.RNOR();
            //printf("h = %f\n", h_norm_sample[i*height+j]);
        }
    }*/
    
    for (int j = 0; j < height; j++) {
    	for (int i = 0; i < width; i++) {    
            h_norm_sample[j*width + i] = normrnd.RNOR();
            //printf("h = %f\n", h_norm_sample[i*height+j]);
        }
    }
	printf("	- size of d_norm_sample: %d\n", size_norm/4);
	
    float *d_norm_sample = NULL;

    checkError(cudaMalloc((void**)&d_norm_sample, size_norm));

    checkError(cudaMemcpy(d_norm_sample, h_norm_sample, size_norm, cudaMemcpyHostToDevice));
	
    cudaPrintfInit();
	
	cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2,0);
    //generate_asset_price_paths_and_cash_flow_reg<<<blocksPerGrid,threadsPerBlock>>>(d_option_value, d_option_value_am, width, height, h_indata, d_norm_sample);
    generate_asset_price_paths_and_cash_flow_reg_mp<<<blocksPerGrid,threadsPerBlock>>>(d_option_value, d_option_value_am, width, height, h_indata, d_norm_sample);
    //generate_asset_price_paths_and_cash_flow<<<blocksPerGrid,threadsPerBlock>>>(d_S, d_cash_flow, d_option_value, width, height, h_indata, d_norm_sample);
    //generate_asset_price_paths_and_cash_flow_multiple_paths<<<blocksPerGrid/4,threadsPerBlock>>>(d_S, d_cash_flow, d_option_value, width, height, h_indata, d_norm_sample);     
	
	cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&GPU_t, start2, stop2);
	printf("\n### GPU: Time to generate price paths: %fs\n", GPU_t/1000);
	
    thrust::device_ptr<float> dev_option_value_b(d_option_value);
    thrust::device_ptr<float> dev_option_value_e = dev_option_value_b + width;
    float sum = thrust::reduce(dev_option_value_b, dev_option_value_e, (float)0, thrust::plus<float>());
    float var_eu = thrust::transform_reduce(dev_option_value_b, dev_option_value_e, square<float>(), (float)0, thrust::plus<float>());


    float european_option_value  = sum/width;
    var_eu = (var_eu - pow(european_option_value, 2) )/width;

	cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3,0);
    
    //find_optimal_exercise_boundary_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_cash_flow, d_option_value, width, height, h_indata, d_x, d_h, d_optimal_exercise_boundary, d_cash_flow_am);

	cudaEventRecord(stop3,0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&GPU_t, start3, stop3);
	printf("\n### Time to generate optimal exercise boundary: %fs\n", GPU_t/1000);
	
	thrust::device_ptr<float> dev_option_value_am_b(d_option_value_am);
    thrust::device_ptr<float> dev_option_value_am_e = dev_option_value_am_b + width;
    float sum_a = thrust::reduce(dev_option_value_am_b, dev_option_value_am_e, (float)0, thrust::plus<float>());
    float var_am = thrust::transform_reduce(dev_option_value_am_b, dev_option_value_am_e, square<float>(), (float)0, thrust::plus<float>());
    float american_option_value  = sum_a/width;
    var_am = (var_am - pow(american_option_value, 2) )/width;


    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&GPU_t, start, stop);

	// show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte );

    if ( cudaSuccess != cuda_status )
    {
	    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
     	//exit(1);
     }

     //double free_db = (double)free_byte ;
     //double total_db = (double)total_byte ;
     //double used_db = total_db - free_db ;

	printf("\n\nSUMMARY RESULTS FOR GPU\n------------------------\n");
	printf(" i) American Option:\n");
	printf("%40s:   %.6f \n", "Valuation at t=0", american_option_value);
	printf("%40s:   %.6f \n", "Std dev of the samples", sqrt(var_am) );
	float delta_am = 1.96*sqrt(var_am/width)/american_option_value;
	printf("%40s:   %.3g %% (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", 100*delta_am/(1-delta_am) );
	printf("\nii) European Option:\n");
	printf("%40s:   %.6f \n", "Valuation at t=0", european_option_value);
	printf("%40s:   %.6f \n", "Std dev of the samples", sqrt(var_eu) );
	float delta_eu = 1.96*sqrt(var_eu/width)/european_option_value;
	printf("%40s:   %.3g %% (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", 100*delta_eu/(1-delta_eu) );
	printf("\niii) Early Exercise Value: %g\n", american_option_value - european_option_value);
	
	printf("\n\nRESOURCE USAGE FOR GPU\n------------------------\n");
    printf("%40s: %.3fs\n", "Time in GPU",GPU_t/1000);
    printf("%40s: %.2f megabyte\n", "GPU memory estimate", (total_byte - free_byte)*9.53674e-7);

    cudaPrintfDisplay(stdout,true);
    cudaPrintfEnd();
    /*
       for (int i = 0; i < width; i++) {
    //for (int j = 0; j < height; j++) {    
    //printf("s[%d][%d] = %f\n", i,j,h_S[i*height+j]);
    printf("cashflow[%d] = %f\n", i,h_cash_flow[i]);
    printf("option_value[%d] = %f\n", i,h_option_value[i]);
    printf("var_eu[%d] = %f\n", i,h_var_eu[i]);
    //}
    }
     */

    checkError(cudaFree(d_S));
    checkError(cudaFree(d_x));
    checkError(cudaFree(d_h));
    checkError(cudaFree(d_cash_flow));
    checkError(cudaFree(d_option_value));
    checkError(cudaFree(d_cash_flow_am));
    checkError(cudaFree(d_optimal_exercise_boundary));
}
