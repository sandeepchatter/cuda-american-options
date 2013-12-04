// #ifdef __cplusplus
// extern "C" {
// #endif


//	INCLUDE/DEFINE

//	LIBRARIES

#include <stdio.h>									// (in path known to compiler)	needed by printf
#include <stdlib.h>									// (in path known to compiler)	needed by malloc, free
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include <curand.h>


//	HEADER

#include "./main.h"									// (in current path)

//	UTILITIES
#include "./util/timer/timer.h"						// (in specified path)
#include "./util/regress/regress_CPU.h"				// (in specified path)
#include "./util/MonteCarlo_structs.h"
#include "./util/FileIO/FileIO.h"

//  PACKAGES
#include "./stock_simulation/stock_simulation.h"

//	KERNEL

#include "./kernel/kernel_gpu_cuda_wrapper.h"

//	End

//	MAIN FUNCTION


float randFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}


///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
/*extern "C" void MonteCarloCPU(
    TOptionValue   &callValue,
    TOptionData optionData,
    float *h_Random,
    int pathN
);*/

int main(void) {

    /*TOptionData optionData;
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
	
	stock_simulation sim; 
    sim.EuropeanOptionsMonteCarloCPU(
            callValueCPU,
            optionData,
            NULL,
            PATH_N
            );

    printf("Expected: %f\n", callValueCPU.Expected);
    printf("Confidence: %f\n", callValueCPU.Confidence);*/
	
	InputData indata;
	FileIO fileIO;
	
	fileIO.readInputFile((char*)"./input/options.txt", indata);
	printf( "num_paths: %d, num_stamps: %d \n", indata.num_paths, indata.num_time_stamps); 
    return 0;
}

//	END
