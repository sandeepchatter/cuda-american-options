
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include <curand.h>
#include "MonteCarlo_common.h"



//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real MuByT;
    real VBySqrtT;
} __TOptionData;
static __device__ __constant__ __TOptionData d_OptionData[MAX_OPTIONS];

static __device__ __TOptionValue d_CallValue[MAX_OPTIONS];


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

    const int optionIndex = blockIdx.x;
    const real        S = d_OptionData[optionIndex].S;
    const real        X = d_OptionData[optionIndex].X;
    const real    MuByT = d_OptionData[optionIndex].MuByT;
    const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

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
        d_CallValue[optionIndex] = t;
    }
}







float randFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream);

///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(
    TOptionValue   &callValue,
    TOptionData optionData,
    float *h_Random,
    int pathN
);

int main(void) {

    TOptionData optionData;
    TOptionValue callValueGPU;
    TOptionValue callValueCPU;
    int PATH_N = 262133;
    optionData.S = randFloat(5.0f, 50.0f);
    optionData.X = randFloat(10.0f, 25.0f);
    optionData.T = randFloat(1.0f, 5.0f);
    optionData.R = 0.06f;
    optionData.V = 0.10f;
    callValueGPU.Expected   = -1.0f;
    callValueGPU.Confidence = -1.0f;

    MonteCarloCPU(
            callValueCPU,
            optionData,
            NULL,
            PATH_N
            );

    printf("Expected: %f\n", callValueCPU.Expected);
    printf("Confidence: %f\n", callValueCPU.Confidence);


    return 0;
}


//Main computations
void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{
    __TOptionData h_OptionData[MAX_OPTIONS];
    __TOptionValue *h_CallValue = plan->h_CallValue;

    if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS)
    {
        printf("MonteCarloGPU(): bad option count.\n");
        return;
    }

    for (int i = 0; i < plan->optionCount; i++)
    {
        const double           T = plan->optionData[i].T;
        const double           R = plan->optionData[i].R;
        const double           V = plan->optionData[i].V;
        const double       MuByT = (R - 0.5 * V * V) * T;
        const double    VBySqrtT = V * sqrt(T);
        h_OptionData[i].S        = (real)plan->optionData[i].S;
        h_OptionData[i].X        = (real)plan->optionData[i].X;
        h_OptionData[i].MuByT    = (real)MuByT;
        h_OptionData[i].VBySqrtT = (real)VBySqrtT;
    }

    checkCudaErrors(cudaMemcpyToSymbolAsync(
                        d_OptionData,
                        h_OptionData,
                        plan->optionCount * sizeof(__TOptionData),
                        0, cudaMemcpyHostToDevice, stream
                    ));

    MonteCarloOneBlockPerOption<<<plan->optionCount, THREAD_N, 0, stream>>>(
        plan->rngStates,
        plan->pathN
    );
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");


    checkCudaErrors(cudaMemcpyFromSymbolAsync(
                        h_CallValue,
                        d_CallValue,
                        plan->optionCount * sizeof(__TOptionValue), (size_t)0, cudaMemcpyDeviceToHost, stream
                    ));

    //cudaDeviceSynchronize();

}
