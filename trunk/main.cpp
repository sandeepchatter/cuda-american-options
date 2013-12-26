//	C++ LIBRARIES
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<vector>

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

vector<result_set*> r_sets;

float randFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void analyze_results(FILE* out)
{
	if ( r_sets.size() == 0 )
	{
		fprintf(stderr, "\nNo results in Result-Set!!!\n");
		return;
	}
	
	
	//always assume that the last reuslt set belongs to the CPU
	int cpu_index = r_sets.size()-1;
	
	for ( int i = 0; i < cpu_index; i++ )
	{
		fprintf(out, "\n\nComparsion between %s and %s\n---------------------------------------------------------\n\n",
		(char*)r_sets[cpu_index]->desc.c_str(), (char*)r_sets[i]->desc.c_str());
		
		fprintf(out, "%35s    %25s | %25s | %6s \n", " ", (char*)r_sets[cpu_index]->desc.c_str(), (char*)r_sets[i]->desc.c_str(), "Error");
		fprintf(out, " i) American Option:\n");
		
		fprintf(out, "%35s:   %25g | %25g | %6.2f\% \n", "Valuation at t=0", r_sets[cpu_index]->american_option_value, r_sets[i]->american_option_value,
		100.0*( r_sets[cpu_index]->american_option_value - r_sets[i]->american_option_value)/r_sets[cpu_index]->american_option_value );
		
		fprintf(out, "%35s:   %25g | %25g | %6.2f\% \n", "Std dev of the samples", r_sets[cpu_index]->std_dev_am , r_sets[i]->std_dev_am ,
		100.0*( r_sets[cpu_index]->std_dev_am  - r_sets[i]->std_dev_am )/r_sets[cpu_index]->std_dev_am );
		
		fprintf(out, "%35s:   %25g | %25g | %6s \n", "Max rel error (95% confidence)", r_sets[cpu_index]->max_rel_error_am , r_sets[i]->max_rel_error_am ,
		"-N.A-" );
		
		fprintf(out, "\nii) European Option:\n");
		
		fprintf(out, "%35s:   %25g | %25g | %6.2f\% \n", "Valuation at t=0", r_sets[cpu_index]->european_option_value, r_sets[i]->european_option_value,
		100.0*( r_sets[cpu_index]->european_option_value - r_sets[i]->european_option_value)/r_sets[cpu_index]->european_option_value );
		
		fprintf(out, "%35s:   %25g | %25g | %6.2f\% \n", "Std dev of the samples", r_sets[cpu_index]->std_dev_eu, r_sets[i]->std_dev_eu ,
		100.0*( r_sets[cpu_index]->std_dev_eu - r_sets[i]->std_dev_eu )/r_sets[cpu_index]->std_dev_eu );
		
		fprintf(out, "%35s:   %25g | %25g | %6s \n", "Max rel error (95% confidence)", r_sets[cpu_index]->max_rel_error_eu , r_sets[i]->max_rel_error_eu ,
		"-N.A-" );
		
		float exv_cpu = r_sets[cpu_index]->american_option_value - r_sets[cpu_index]->european_option_value;
		float exv_gpu = r_sets[i]->american_option_value - r_sets[i]->european_option_value;
		fprintf(out, "\niii)%31s:   %25g | %25g | %6.2f\% \n", "Early Exercise Value", exv_cpu, exv_gpu,100.0*( exv_cpu - exv_gpu )/exv_cpu );
		
		fprintf(out, "\niv)Resource Usage:\n");
		
		fprintf(out, "%35s:   %25g | %25g | %6.2fx \n", "Total time taken (sec)", r_sets[cpu_index]->net_clock_time, r_sets[i]->net_clock_time ,
		r_sets[cpu_index]->net_clock_time/r_sets[i]->net_clock_time);
		
		fprintf(out, "%35s:   %25g | %25g | %6.2fx \n", "Peak memory estimate (Mb)", r_sets[cpu_index]->memory_usage, r_sets[i]->memory_usage ,
		r_sets[cpu_index]->memory_usage/r_sets[i]->memory_usage );
	} 
}

//	MAIN FUNCTION
// just a wrappper around the main stock simulation class.
int main(int argc, char** argv )
{
	bool nv1 = 0, cv1 = 0, nv2 = 0, cv2 = 0;
	
	for( int i = 1; i < argc; i++)
	{
		if (strncmp(argv[i],"--nv1", 5) == 0)
			nv1 = 1;
		else if (strncmp(argv[i],"--cv1", 5) == 0)
			cv1 = 1;
		else if (strncmp(argv[i],"--nv2", 5) == 0)
			nv2 = 1;
		else if (strncmp(argv[i],"--cv2", 5) == 0)
			cv2 = 1;
	}
	
    if ( nv1 )
    {
    	result_set* r_set = new result_set("GPU:normrand_v1 results");
    	_gpu_find_option_values_using_normrand( r_set );
    	r_sets.push_back( r_set);
    }
    
    if ( cv1 )
    {
    	result_set* r_set = new result_set("GPU:curand_v1 results");
		_gpu_find_option_values_using_curand( r_set );
		r_sets.push_back( r_set);
	}
	
	if ( nv2 )
	{
		result_set* r_set = new result_set("GPU:normrand_v2 results");
		_gpu_find_option_values_using_normrand_v2( r_set );
		r_sets.push_back( r_set);
	}
	
	if ( cv2 )
	{
		result_set* r_set = new result_set("GPU:curand_v2 results");
		_gpu_find_option_values_using_curand_v2( r_set );
		r_sets.push_back( r_set);
	}
	
	// Create an instance of stock_simulation class, which in this case
	// is a put option.
	result_set* r_set = new result_set("CPU results");
	stock_simulation option (r_set);
	r_sets.push_back( r_set);
	
	// For the given option, generate the price paths for the underlying asset.
	#ifdef VERBOSE
	clock_t begin, end;
    float CPU_t = 0;
	begin = clock();
	#endif
	
	option.generate_asset_price_paths();
	
	#ifdef VERBOSE
	end = clock();
    CPU_t = (float) (end - begin) / CLOCKS_PER_SEC;
    printf("\n### CPU: Time to generate price paths = %f\n", CPU_t);
    #endif
	
	#ifdef VERBOSE
	CPU_t = 0;
	begin = clock();
	#endif
	
	// Find the optimal exercise boundary for the American Option based on
	// previously generated Price paths
	option.find_optimal_exercise_boundary();
	
	#ifdef VERBOSE
	end = clock();
    CPU_t = (float) (end - begin) / CLOCKS_PER_SEC;
    printf("\n### CPU: Time to generate optimal exercise boundary = %f\n", CPU_t);
	#endif
	option.get_resource_usage(stdout);
	
	
	analyze_results(stdout);
	
    return 0;
}

