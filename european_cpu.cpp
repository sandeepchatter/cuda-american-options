
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include <curand.h>
#include "MonteCarlo_common.h"

static double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

extern "C" void MonteCarloCPU(
        TOptionValue    &callValue,
        TOptionData optionData,
        float *h_Samples,
        int pathN
        )
{
    const double        S = optionData.S;
    const double        X = optionData.X;
    const double        T = optionData.T;
    const double        R = optionData.R;
    const double        V = optionData.V;
    const double    MuByT = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);

    float *samples;
    curandGenerator_t gen;

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    unsigned long long seed = 1234ULL;
    curandSetPseudoRandomGeneratorSeed(gen,  seed);

    if (h_Samples != NULL)
    {
        samples = h_Samples;
    }
    else
    {
        samples = (float *) malloc(pathN * sizeof(float));
        curandGenerateNormal(gen, samples, pathN, 0.0, 1.0);
    }

    // for(int i=0; i<10; i++) printf("CPU sample = %f\n", samples[i]);

    double sum = 0, sum2 = 0;

    for (int pos = 0; pos < pathN; pos++)
    {

        double    sample = samples[pos];
        double callValue = endCallValue(S, X, sample, MuByT, VBySqrtT);
        sum  += callValue;
        sum2 += callValue * callValue;
    }

    if (h_Samples == NULL) free(samples);

    curandDestroyGenerator(gen);

    //Derive average from the total sum and discount by riskfree rate
    callValue.Expected = (float)(exp(-R * T) * sum / (double)pathN);
    //Standart deviation
    double stdDev = sqrt(((double)pathN * sum2 - sum * sum)/ ((double)pathN * (double)(pathN - 1)));
    //Confidence width; in 95% of all cases theoretical value lies within these borders
    callValue.Confidence = (float)(exp(-R * T) * 1.96 * stdDev / sqrt((double)pathN));
}

