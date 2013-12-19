
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include <curand.h>
#include "option_kernel.h"
#include "MonteCarlo_reduction.cuh"
#include "cuPrintf.cu"

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

__device__ void get_black_scholes_continuation_value_gpu(float *x, float time, float *h, int width, InputData indata ) {
    float del_t = indata.expiry_time/(width-1)/365;
    float t = time*del_t;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float d1, d2, den;
    float ttm = (indata.expiry_time - t)/365;
    d1 = log(x[tid]/indata.strike_price) + ( indata.discount_rate + 0.5*indata.volatility*indata.volatility )*ttm;
    d2 = log(x[tid]/indata.strike_price) + ( indata.discount_rate - 0.5*indata.volatility*indata.volatility )*ttm;
    den = indata.volatility*sqrtf( ttm );
    d1 = d1/den;
    d2 = d2/den;

    h[tid] = indata.strike_price*exp(-1*indata.discount_rate*ttm)*phi(-1*d2) - x[tid]*phi(-1*d1);
    // cuPrintf("htid[%d] = %f\n", tid, h[tid]);
    //printf("d1: %g, d2: %g, den: %g, phi(-1*d2): %g, phi(-1*d1): %g, h[i]: %g, x[i]: %g\n", d1, d2, den, phi(-1*d2), phi(-1*d1), h[i], x[i]);
}

static __global__ void generate_asset_price_paths_and_cash_flow(float *S, float *cash_flow, float *option_value, int width, int height, float *norm_sample, InputData indata) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // shared memory to make sure accesses are fast (not sure if no use of tid affects things)

    float drift = indata.discount_rate - indata.dividend - 0.5*pow(indata.volatility,2);
    float del_t = indata.expiry_time/(height-1)/365;
    float sigma = sqrt(del_t)*indata.volatility;
    S[tid*height] = indata.S_0;
    for (int j = 1; j < height; j++ )
    {
        if (tid%2 == 0) {
            S[tid*height+j] = S[tid*height+j-1]*exp( drift*del_t + sigma*norm_sample[tid*height+j] );
        } else {
            S[tid*height+j] = S[tid*height+j-1]*exp( drift*del_t + sigma*-1*norm_sample[tid*height+j] );
        }
        /*
           if (tid == 0 && j == 1) {
           printf("s[%d][%d] = %f\n", tid,j,S[tid*height+j]);
           printf("norm = %f, drift = %f, del_t = %f, sigma = %f\n", norm_sample[tid*height+j], drift, del_t, sigma);
           printf("discrate = %f, dvidi = %f, vol = %f, exptime = %f\n", indata.discount_rate, indata.dividend, indata.volatility, indata.expiry_time);
           }
         */
    }
    int expiry_index = width-1;
    // at the expiry time, the only choice is to exercise the option
    float discount_eu = exp(-1*indata.discount_rate*indata.expiry_time/365 );

    cash_flow[tid] = fmaxf(indata.strike_price - S[tid*height+expiry_index], 0.0); //put
    option_value[tid] = cash_flow[tid]*discount_eu;

    __syncthreads();

}

static __global__ void find_optimal_exercise_boundary_gpu(float *S, float *cash_flow, float *option_value, int width, int height, InputData indata, float *x, float *h, int *optimal_exercise_boundary, float *cash_flow_am) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int expiry_index = width-1;


    float del_t = indata.expiry_time/(width-1)/365;
    // discount for merican counterpart
    float discount = exp(-1*indata.discount_rate*del_t );

    float put_value = 0;

    // for all other times when the option can be exercised, we comapre the
    // value of exercising and continuation value to find optimal exercise boundary  
    for ( int time = expiry_index-1; time >= 1; time-- ) // move back in time
    {

        // find in the money paths
        put_value = fmaxf( indata.strike_price - S[tid*height+time], 0.0); //put

        if ( put_value > 0 )
        {
            x[tid] = S[tid*height+time];
            //y[tid] =  cash_flow_am[tid]*exp(-1*discount_rate*del_t*(optimal_exercise_boundary[tid]-time) ) ;
            //imp_indices[tid] = path;
        } else {
            x[tid] = -1;
            //y[tid] = -1;
        }
        cash_flow[tid] = put_value;

        __syncthreads();

        /********* USING LSM as boundary ************/
        //vector<float> g;
        //least_squares.find_linear_fit( x, y, g, num_laguerre_poly);
        //get_continuation_value_ch( g, h, x );

        /*float norm = 0;
          for( int z = 0; z < h.size(); z++ )
          {
          norm = norm + fabs(y[z] - h[z]);
          }
          printf("\ntime: %gd, error-norm: %.3g, avg-error: %.3g", time*del_t*365, norm, norm/h.size());*/

        /********* USING BSE as boundary ************/

        get_black_scholes_continuation_value_gpu( x, time, h, width, indata);

        if ( cash_flow[tid] > h[tid] )
        {
            optimal_exercise_boundary[tid] = time;
            cash_flow_am[tid] = fmaxf(indata.strike_price - S[tid*height+time], 0.0);
        }

        __syncthreads();
    }

    cash_flow_am[tid] = fmaxf(indata.strike_price - S[tid*height+optimal_exercise_boundary[tid]], 0.0); 
    discount = exp(-1*indata.discount_rate*optimal_exercise_boundary[tid]*del_t );
    option_value[tid] = cash_flow_am[tid]*discount; 

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
    InputData h_indata;
    cudaError_t err = cudaSuccess;
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
    float *d_cash_flow_am = NULL;
    int *d_optimal_exercise_boundary = NULL;
    float *h_S = NULL;
    float *h_x = NULL;
    float *h_h = NULL;
    float *h_cash_flow = NULL;
    float *h_option_value = NULL;
    float *h_cash_flow_am = NULL;
    int *h_optimal_exercise_boundary = NULL;
    int width = num_paths;
    int height = h_indata.num_time_steps+1;
    //size_t size = num_paths*(h_indata.num_time_steps+1)*sizeof(float);
    printf("width=%d\n", width);

    h_S = (float*) malloc(sizeof(float)*width*height);
    h_x = (float*) malloc(sizeof(float)*width);
    h_h = (float*) malloc(sizeof(float)*width);
    h_cash_flow = (float*) malloc(sizeof(float)*width);
    h_option_value = (float*) malloc(sizeof(float)*width);
    h_cash_flow_am = (float*) malloc(sizeof(float)*width);
    h_optimal_exercise_boundary = (int*) malloc(sizeof(int)*width);

    checkError(cudaMalloc((void**)&d_S, width*sizeof(float)*height));

    checkError(cudaMalloc((void**)&d_x, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_h, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_cash_flow, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_option_value, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_cash_flow_am, width*sizeof(float)));
    checkError(cudaMalloc((void**)&d_optimal_exercise_boundary, width*sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = width/threadsPerBlock;

    printf("blocksergrdi=%d\n", blocksPerGrid);
    printf("threadsperblock=%d\n", threadsPerBlock);
    random_normal normrnd;
    normrnd.zigset(h_indata.random_seed);

    size_t size_norm = width*height*sizeof(float);
    float *h_norm_sample = (float *) malloc(size_norm);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {    
            h_norm_sample[i*height+j] = normrnd.RNOR();
            //printf("h = %f\n", h_norm_sample[i*height+j]);
        }
    }

    float *d_norm_sample = NULL;

    checkError(cudaMalloc((void**)&d_norm_sample, size_norm));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    checkError(cudaMemcpy(d_norm_sample, h_norm_sample, size_norm, cudaMemcpyHostToDevice));


    cudaPrintfInit();

    generate_asset_price_paths_and_cash_flow<<<blocksPerGrid,threadsPerBlock>>>(d_S, d_cash_flow, d_option_value, width, height, d_norm_sample, h_indata);     


    thrust::device_ptr<float> dev_option_value_b(d_option_value);
    thrust::device_ptr<float> dev_option_value_e = dev_option_value_b + width;
    float sum = thrust::reduce(dev_option_value_b, dev_option_value_e, (float)0, thrust::plus<float>());
    float var_eu = thrust::transform_reduce(dev_option_value_b, dev_option_value_e, square<float>(), (float)0, thrust::plus<float>());


    printf("sum = %f, vareu = %f\n", sum, var_eu);
    float european_option_value  = sum/width;
    var_eu = (var_eu - pow(european_option_value, 2) )/width;
    printf("european option value = %f, vareu = %f\n", european_option_value, var_eu);



    find_optimal_exercise_boundary_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_cash_flow, d_option_value, width, height, h_indata, d_x, d_h, d_optimal_exercise_boundary, d_cash_flow_am);

    float sum_a = thrust::reduce(dev_option_value_b, dev_option_value_e, (float)0, thrust::plus<float>());
    float var_am = thrust::transform_reduce(dev_option_value_b, dev_option_value_e, square<float>(), (float)0, thrust::plus<float>());
    printf("sum = %f, vareu = %f\n", sum_a, var_am);
    float american_option_value  = sum_a/width;
    var_am = (var_am - pow(american_option_value, 2) )/width;

    printf("american option value = %f, varam = %f\n", american_option_value, var_am);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&GPU_t, start, stop);

    printf("Time for GPU: %f\n", GPU_t);

    cudaPrintfDisplay(stdout,true);
    cudaPrintfEnd();
    //err = cudaMemcpy2D(h_S, pitch, d_S, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyDeviceToHost);
    /*err = cudaMemcpy(h_S, d_S, width*sizeof(float)*height, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
      fprintf(stderr, "Failed to get device vector S (error code %s)\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
      }
      err = cudaMemcpy(h_cash_flow, d_cash_flow, width*sizeof(float), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
      fprintf(stderr, "Failed to get device vector cash_flow (error code %s)\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
      }
      err = cudaMemcpy(h_option_value, d_option_value, width*sizeof(float), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
      fprintf(stderr, "Failed to get device vector option_value (error code %s)\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
      }
     */
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


}



