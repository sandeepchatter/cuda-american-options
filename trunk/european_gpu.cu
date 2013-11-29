
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include <curand_kernel.h>
#include "MonteCarlo_common.h"
#include "MonteCarlo_reduction.cuh"

#define THREAD_N 256


//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real MuByT;
    real VBySqrtT;
} __TOptionData;
static __device__ __constant__ __TOptionData *d_OptionData;

static __device__ __TOptionValue *d_CallValue;


__device__ inline float endCallValue(float S, float X, float r, float MuByT, float VBySqrtT)
{
    float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}



////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
static __global__ void MonteCarloOneBlockPerOption(
    curandState *rngStates,
    int pathN)
{
    const int SUM_N = THREAD_N;
    __shared__ real s_SumCall[SUM_N];
    __shared__ real s_Sum2Call[SUM_N];

    //const int optionIndex = blockIdx.x;
    const real        S = d_OptionData->S;
    const real        X = d_OptionData->X;
    const real    MuByT = d_OptionData->MuByT;
    const real VBySqrtT = d_OptionData->VBySqrtT;

    // determine global thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy random number state to local memory for efficiency
    curandState localState = rngStates[tid];

    //Cycle through the entire samples array:
    //derive end stock price for each path
    //accumulate partial integrals into intermediate shared memory buffer
    for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x)
    {
        __TOptionValue sumCall = {0, 0};

        for (int i = iSum; i < pathN; i += SUM_N)
        {
            real              r = curand_normal(&localState);
            real      callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
            sumCall.Expected   += callValue;
            sumCall.Confidence += callValue * callValue;
        }

        s_SumCall[iSum]  = sumCall.Expected;
        s_Sum2Call[iSum] = sumCall.Confidence;
    }

    // store random number state back to global memory
    rngStates[tid] = localState;

    //Reduce shared memory accumulators
    //and write final result to global memory
    sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call);

    if (threadIdx.x == 0)
    {
        __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]};
        *d_CallValue = t;
    }
}

//Main computations
extern "C" void MonteCarloGPU(TOptionPlan *plan)
{
    __TOptionData *h_OptionData;
    __TOptionValue *h_CallValue = plan->h_CallValue;


        const double           T = plan->optionData->T;
        const double           R = plan->optionData->R;
        const double           V = plan->optionData->V;
        const double       MuByT = (R - 0.5 * V * V) * T;
        const double    VBySqrtT = V * sqrt(T);
        &(h_OptionData)->S        = (real)plan->optionData->S;
        *h_OptionData->X        = (real)plan->optionData->X;
        *h_OptionData->MuByT    = (real)MuByT;
        *h_OptionData->VBySqrtT = (real)VBySqrtT;

    cudaMemcpyToSymbol(
                        d_OptionData,
                        h_OptionData,
                        sizeof(__TOptionData),
                        (size_t)0, cudaMemcpyHostToDevice
                    );

    MonteCarloOneBlockPerOption<<<plan->optionCount, THREAD_N, 0>>>(
        plan->rngStates,
        plan->pathN
    );


    cudaMemcpyFromSymbol(
                        h_CallValue,
                        d_CallValue,
                        sizeof(__TOptionValue), (size_t)0, cudaMemcpyDeviceToHost
                    );

    //cudaDeviceSynchronize();

}
