#ifndef _STOCK_SIMULATION_H_
#define _STOCK_SIMULATION_H_

#include <vector>

#include "../util/random/random_normal.h"
#include "../util/timer/timer.h"
#include "../util/MonteCarlo_structs.h"

using namespace std;

class stock_simulation
{
	vector < vector <float> > S;	// asset price paths 2D matrix
	float dividend;					// dividend on the underlying asset
	float expiry_time;
	random_normal normrnd;

	public:
	
	stock_simulation( InputData &indata );
	
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
