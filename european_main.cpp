#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include <curand.h>
#include "MonteCarlo_common.h"


float randFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}




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



