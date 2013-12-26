#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H

#include "realtype.h"
#include "curand_kernel.h"


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
	int num_laguerre_poly;	/*!< Number of Laguerre polynomials to be used as
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
};


#endif
