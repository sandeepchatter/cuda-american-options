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
    //_gpu_find_option_values_using_normrand();
	//_gpu_find_option_values_using_curand();
	//_gpu_find_option_values_using_normrand_v2();
	_gpu_find_option_values_using_curand_v2();
	
	// Create an instance of stock_simulation class, which in this case
	// is a put option. 
	stock_simulation option;
	
	// For the given option, generate the price paths for the underlying asset.
	clock_t begin, end;
    float CPU_t = 0;
	begin = clock();
	
	option.generate_asset_price_paths();
	
	end = clock();
    CPU_t = (float) (end - begin) / CLOCKS_PER_SEC;
    printf("\n### CPU: Time to generate price paths = %f\n", CPU_t);
    
	// Find the optimal exercise boundary for the American Option based on
	// previously generated Price paths
	CPU_t = 0;
	begin = clock();
	
	option.find_optimal_exercise_boundary();
	
	end = clock();
    CPU_t = (float) (end - begin) / CLOCKS_PER_SEC;
    printf("\n### CPU: Time to generate optimal exercise boundary = %f\n", CPU_t);
	option.get_resource_usage(stdout);
	
   
	
    return 0;
}

