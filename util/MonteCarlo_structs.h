#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H

#include "realtype.h"
#include "curand_kernel.h"

//#define VERBOSE

typedef struct
{
    float S;
    float X;
    float T;
    float R;
    float V;
} TOptionData;

typedef struct
{
    float Expected;
    float Confidence;
} TOptionValue;

//GPU outputs before CPU postprocessing
typedef struct
{
    real Expected;
    real Confidence;
} __TOptionValue;

typedef struct
{
    //Device ID for multi-GPU version
    int device;
    //Option count for this plan
    int optionCount;

    //Host-side data source and result destination
    TOptionData  *optionData;
    TOptionValue *callValue;

    //Temporary Host-side pinned memory for async + faster data transfers
    __TOptionValue *h_CallValue;


    //Intermediate device-side buffers
    void *d_Buffer;

    //random number generator states
    curandState *rngStates;

    //Pseudorandom samples count
    int pathN;

    //Time stamp
    float time;

    //random number generator seed.
    unsigned long long seed;
} TOptionPlan;

extern "C" void MonteCarloGPU(TOptionPlan *plan);

//=================================//=================================//
/*! \struct InputData
    
    A structure to store all the variable values read from a input file
    provided by the user. These variables specify the i) settings to be used
    while evaluating american options using Monte Carlo and ii) statistical 
    properties of the underlying asset.     
*/
struct InputData
{
	// settings to be used
	int num_paths;			/*!< Number of Monte Carlo paths to be generated
								 for the underlying asset*/
	int num_time_steps;		/*!< NUmber of time steps between t=0 and expiry time
								 at which the American option can be exercised. */
	int random_seed;		/*!< Seed for the random number generator */
	int num_chebyshev_poly;	/*!< Number of Laguerre polynomials to be used as
								 basis functions*/
	
	// statistical properties of the underlying asset 
	float discount_rate;	/*!< The risk free rate of return  */
	float dividend;			/*!< Dividend on the underlying asset */
	float expiry_time;		/*!< Expiry time of the option (in days)*/
	float S_0;				/*!< The price of the asset at t=0 */
	float volatility;		/*!< The volatility of the underlying asset */
	float strike_price;		/*!< The agreed upon strike price of the underlying asset */
	
	// GPU configurations
	int num_paths_per_thread;
	
	void print_details( FILE* out )
	{
		fprintf(out, "\nSETTINGS FOR SIMULATION\n-------------------------------\n\n");
		fprintf(out, "%40s:   %d \n", "Number of Monte Carlo Paths", num_paths);
		fprintf(out, "%40s:   %d \n", "Number of time-steps for each path", num_time_steps );
		fprintf(out, "%40s:   %d \n", "Random seed used", random_seed );
		fprintf(out, "%40s:   %s \n", "Continuation Function used", (num_chebyshev_poly==0)?"Black-Scholes":"Least-Squares" );
		//fprintf(out, "%40s:   %d \n", "Number of Chebyshev polynomials used", num_chebyshev_poly );
		
		fprintf(out, "\nPROPERTIES OF THE OPTION/ASSET\n-------------------------------\n\n");
		fprintf(out, "%40s:   %.3f \n", "The risk free rate of return", discount_rate);
		fprintf(out, "%40s:   %.3f \n", "Dividend on the underlying asset", num_time_steps );
		fprintf(out, "%40s:   %.3f \n", "Expiry time of the option (in days)", expiry_time );
		fprintf(out, "%40s:   %.3f \n", "Strike Price", strike_price );
		fprintf(out, "%40s:   %.3f \n", "Volatility on the underlying asset", volatility);
		fprintf(out, "%40s:   %.3f \n", "The price of the asset at t=0", S_0 );
		
		fprintf(out, "\nGPU CONFIGURATION\n-------------------------------\n\n");
		fprintf(out, "%40s:   %d \n", "Number of paths per thread", num_paths_per_thread);
	}
};

struct result_set
{
	std::string desc;
	float american_option_value;
	float european_option_value;
	float std_dev_am;
	float std_dev_eu;
	float max_rel_error_am;	// Maximum rel error w.r.t. true mean for 95% confidence
	float max_rel_error_eu;	// Maximum rel error w.r.t. true mean for 95% confidence
	
	float net_clock_time;	// using cpu (user+system) time on CPU and times using events on GPU, seconds
	float memory_usage;		// the peak resident set size, megabytes
	
	result_set( std::string _desc )
	{
		desc = _desc;
	}
	
	void print_details( FILE* out )
	{
		fprintf(out, "\n\nSummary for %s\n------------------------\n", (char*)desc.c_str());
		fprintf(out, " i) American Option:\n");
		fprintf(out, "%40s:   %.6f \n", "Valuation at t=0", american_option_value);
		fprintf(out, "%40s:   %.6f \n", "Std dev of the samples", std_dev_am );
		fprintf(out, "%40s:   %.3g %% (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", max_rel_error_am );
		fprintf(out, "\nii) European Option:\n");
		fprintf(out, "%40s:   %.6f \n", "Valuation at t=0", european_option_value);
		fprintf(out, "%40s:   %.6f \n", "Std dev of the samples", std_dev_eu );
		fprintf(out, "%40s:   %.3g %% (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", max_rel_error_eu );
		fprintf(out, "\niii) Early Exercise Value: %g\n", american_option_value - european_option_value);
	
		fprintf(out, "\n\nRESOURCE USAGE FOR GPU\n------------------------\n");
		fprintf(out, "%40s: %.3fs\n", "Time taken by GPU", net_clock_time);
		fprintf(out, "%40s: %.2f megabyte\n", "GPU memory estimate", memory_usage);	
	}
};

#endif
