#ifndef _STOCK_SIMULATION_H_
#define _STOCK_SIMULATION_H_

#include <vector>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "../util/random/random_normal.h"
#include "../util/timer/timer.h"
#include "../util/FileIO/FileIO.h"
#include "../util/MonteCarlo_structs.h"
#include "../util/regress/regress_CPU.h"

using namespace std;

/*! \class stock_simulation
    \brief The main class that finds the value of American options
           based on Monte-carlo Simulations.

    This class uses the LSM algorithm proposed by Longstaff and Schwartz (2001)
    in their paper titled "Valuing American Options by Simulation: A Simple
    Least-Squares Approach" to evaluate American Options based on user provided
    parameters for discount rate, volatility, strike price and dividend. 
*/
class stock_simulation
{
	// A 2D matrix, where each row represents a particular price path for
	// the underlying asset and each column represents a discrete exercise date: 
	// a time when the option can be execised. The exercise times are assumed to
	// be evenly distributed between start date and expiry date.  
	vector < vector <float> > S;	
	
	// Dividend on the underlying asset, which is constant and assumed
	//  to be discretely payable in between exercise dates. 
	float dividend;
	
	// annualized risk free rate of return, stored as a fraction					
	float discount_rate;
	
	// The expiry time of the option, in days. 
	float expiry_time;
	
	// Volatility of the underlying asset, used to derive the variance
	// for the Brownian Motion
	float volatility;
	
	// The agreed upon strike price for the underlying asset 
	float strike_price;
	
	// a random number genrator based on the zigguart method
	// (http://en.wikipedia.org/wiki/Ziggurat_algorithm)
	random_normal normrnd;
	
	// number of Laguerre polynomials to be used as basis functions
	int num_laguerre_poly;
	
	// variables used in functions, representing respectively the drift, the 
	// time between exercise dates and standard deviation of the underlying 
	// brownian motion.  
	float drift, del_t, sigma;
	
	// A vector that stores the optimal exercise boundary for the American
	// option, i.e. it stores, for each price path, the optimal time at which
	// option should be exercised to get the maximum return  
	vector <float> optimal_exercise_boundary;
	
	// An instance of the class that implements the linear regression using
	// linear least squares method. 
	Linear_Regression least_squares;
	
	// The calculated value of the American option, and the maxium relative error
	// for 95% confidence
	float american_option_value, am_epsilon, var_am;
	
	// The calculated value of the corresponding European option, and the
	// maxium relative error for 95% confidence
	float european_option_value, eu_epsilon, var_eu;
	
	// a pointer fileIO object to write to log file
	FileIO fileIO;
	
	// the start time fot the stock simulation class
	double start_time;
	public:
	
	/*! \brief Constructor for the stock_simulation class.
	 *
	 *  The constructor initializes the private class varibles
	 *  based on input read from the input file. 
	 */
	stock_simulation( );
	
	/*! \brief Generates price paths for underlying assets.
	 *
	 *  This function generates the price paths for an underlying asset based
	 *  on risk-neutral no-arbritage setting using parameters as read from
	 *  input file. 
	 */
	void generate_asset_price_paths();
	
	/*! \brief Finds the optimal stopping times for each price path.
	 *
	 *  Uses the LSM algorithm proposed by Longstaff and Schwartz (2001)
	 *  to find the optimal stopping times so that return on American Option 
	 *  is maximized.
	 */
	void find_optimal_exercise_boundary();
	
	/*! \brief Finds the continuation cash flow.
	 *
	 *  Calculates the expected payoff if the option is not exercised at
	 *  given time instant. 
	 */
	void get_continuation_value_lg( vector<float> &g, vector<float> &h, vector<float>& x);
	
	/*! \brief Finds the continuation cash flow.
	 *
	 *  Calculates the expected payoff if the option is not exercised at
	 *  given time instant. 
	 */
	void get_continuation_value_ch( vector<float> &g, vector<float> &h, vector<float>& x);
	
	
	void get_black_scholes_continuation_value( vector<float>& x, float time, vector<float> &h); 
	
	float phi( float x);
	
	void get_resource_usage( FILE* out);
	void line2arr (char* str, vector<string>* arr, char *tokenizer);
	double get_wall_time();
	//=================================
	/* For European options */
	double EuropeanOptionsEndCallValue(
		double S,
		double X,
		double r,
		double MuByT,
		double VBySqrtT);

	void EuropeanOptionsMonteCarloCPU(
        TOptionValue    &callValue,
        TOptionData optionData,
        float *h_Samples,
        int pathN );
	/* End European Options */
};

//	END

#endif
