//	C++ LIBRARIES
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA LIBRARIES
#include <cuda_runtime.h>
#include <curand.h>


// HEADERS (not needed, but just in case)
#include "./main.h"

// PACKAGES (the main class that simulates mutiple
// stock paths and evaluates options)
#include "./stock_simulation/stock_simulation.h"

// UTILITIES
#include "./util/timer/timer.h"			// profiler for the code
#include "./util/regress/regress_CPU.h"	// A small math library that implements regression
#include "./util/MonteCarlo_structs.h"	// A header file to declare structures common to all classes
#include "./util/FileIO/FileIO.h"		// A utility to read and write Disk files

//	CUSTOM KERNEL
#include "./kernel/kernel_gpu_cuda_wrapper.h"
#include "./kernel/option_kernel.h"
#include <time.h>

using namespace std;

float randFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

//	MAIN FUNCTION
// just a wrappper around the main stock simulation class.
int main(void)
{
    /* NVIDIAs CPU implementation for Europeans Options
     
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
	
	stock_simulation sim; 
    sim.EuropeanOptionsMonteCarloCPU(
            callValueCPU,
            optionData,
            NULL,
            PATH_N
            );

    printf("Expected: %f\n", callValueCPU.Expected);
    printf("Confidence: %f\n", callValueCPU.Confidence);
    */
	
    clock_t begin, end;
    float CPU_t = 0;

    begin = clock();

	// Create an instance of stock_simulation class, which in this case
	// is a put option. 
	stock_simulation option;
	
	// For the given option, generate the price paths for the underlying asset.
	option.generate_asset_price_paths();
	
	// Find the optimal exercise boundary for the American Option based on
	// previously generated Price paths
	option.find_optimal_exercise_boundary();

    end = clock();
    CPU_t = (float) (end - begin) / CLOCKS_PER_SEC;
    printf("CPU time = %fs\n", CPU_t);

	option.get_resource_usage(stdout);
	
   // stock_gpu_simulation option_gpu;

    //option_gpu.generate_and_find_exercise_boundary();
    generate_and_find_exercise_boundary();
	
    return 0;
}

