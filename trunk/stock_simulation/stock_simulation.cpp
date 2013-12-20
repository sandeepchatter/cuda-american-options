#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <curand.h>

#include "stock_simulation.h"
#include "../util/regress/regress_CPU.h"

stock_simulation::stock_simulation( )
{
	InputData indata;
	
	// read the input file for options relating to the number of paths, number
	// of discrete time-steps etc. 
	fileIO.readInputFile((char*)"./input/options.txt", indata);
	printf( "Num Monte Carlo paths: %d\nNum time-steps: %d \n", indata.num_paths, indata.num_time_steps);
	
	start_time = get_wall_time();
	
	// store dividend, volatility and expiry time for the option 
	dividend = indata.dividend;
	expiry_time = indata.expiry_time;
	volatility = indata.volatility;
	discount_rate = indata.discount_rate;
	strike_price = indata.strike_price;
	num_laguerre_poly = indata.num_laguerre_poly;
	
	// allocate memory to store all Monte Carlo paths, and intialize
	// the initial value of the asset at t=0.
	int num_paths = (indata.num_paths%2 == 0)?indata.num_paths:indata.num_paths+1;  
	S.resize(num_paths);
	
	for (int i = 0; i < S.size(); i++)
	{
		S[i].resize(indata.num_time_steps+1);
		S[i][0] = indata.S_0;
	}
	
	// initilize the random generator
	normrnd.zigset(indata.random_seed);
	
	#ifdef SAVE_DATA_TO_LOG
		fileIO.prepare_log_buffer();
		
		// write the input parameters
		fprintf(fileIO.log_file, "\nSETTINGS FOR SIMULATION\n-------------------------------\n\n");
		fprintf(fileIO.log_file, "%40s:   %d \n", "Number of Monte Carlo Paths", indata.num_paths);
		fprintf(fileIO.log_file, "%40s:   %d \n", "Number of time-steps for each path", indata.num_time_steps );
		fprintf(fileIO.log_file, "%40s:   %d \n", "Random seed used", indata.random_seed );
		fprintf(fileIO.log_file, "%40s:   %d \n", "Number of Laguerre polynomials used", indata.num_laguerre_poly );
		
		fprintf(fileIO.log_file, "\nPROPERTIES OF THE OPTION\n-------------------------------\n\n");
		fprintf(fileIO.log_file, "%40s:   %.3f \n", "The risk free rate of return", indata.discount_rate);
		fprintf(fileIO.log_file, "%40s:   %.3f \n", "Dividend on the underlying asset", indata.num_time_steps );
		fprintf(fileIO.log_file, "%40s:   %.3f \n", "Expiry time of the option (in days)", indata.expiry_time );
		fprintf(fileIO.log_file, "%40s:   %.3f \n", "Strike Price", indata.strike_price );
		fprintf(fileIO.log_file, "%40s:   %.3f \n", "Volatility on the underlying asset", indata.volatility);
		fprintf(fileIO.log_file, "%40s:   %.3f \n", "The price of the asset at t=0", indata.S_0 );
	#endif
}

void stock_simulation::generate_asset_price_paths()
{
	drift = discount_rate - dividend - 0.5*pow(volatility,2);
	del_t = expiry_time/(S[0].size()-1)/365;
	sigma = sqrt(del_t)*volatility;
	
	float norm_sample;
	for ( int i = 0; i < S.size(); i=i+2 )
	{
		for (int j = 1; j < S[i].size(); j++ )
		{
			norm_sample = normrnd.RNOR();
			S[i][j] = S[i][j-1]*exp( drift*del_t + sigma*norm_sample );
			S[i+1][j] = S[i+1][j-1]*exp( drift*del_t + sigma*-1*norm_sample );
		}
	}
	#ifdef SAVE_DATA_TO_LOG
	// In printing to log file, it is checked if S.size() <= 100
	// and number of time-steps <= 10. Otherwise, nothing is printed.
	if ( S.size() <= 100 && S[0].size() <= 15 )
	{
		fprintf(fileIO.log_file, "\nSTOCK PRICE PATHS (time in days)\n----------------------------------\n\n");
		fprintf(fileIO.log_file, "%-7s ", "Path No.");
		for ( int i = 0; i < S[0].size(); i++ )
		{
			fprintf(fileIO.log_file, "t = %-6g  ", i*del_t*365 );
		}
		fprintf(fileIO.log_file, "\n----------------------------------------------------------\n");
		for ( int i = 0; i < S.size(); i++ )
		{
			fprintf(fileIO.log_file, "%-7d ", i+1);
			for (int j = 0; j < S[i].size(); j++ )
			{
				fprintf(fileIO.log_file, "%s%-10.4f ", (strike_price>S[i][j] && j!=0)?"*":" ", S[i][j] );
			}
			fprintf(fileIO.log_file, "\n");
		}
	}
	else
	{
		fprintf(fileIO.log_file, "******************************************************************\n");	
		fprintf(fileIO.log_file, "* Stock Price paths are NOT printed because there a lot of them. *\n");
		fprintf(fileIO.log_file, "******************************************************************\n");	
	}
	#endif
}

// assuming a put option as of now
void stock_simulation::find_optimal_exercise_boundary()
{	
	vector<float> cash_flow( S.size() );
	
	// copy varibales
	vector<float> cash_flow_eu( S.size() );
	vector<float> cash_flow_am( S.size() );

	int expiry_index = S[0].size()-1;
	optimal_exercise_boundary.resize( S.size(), expiry_index );
	
	// at the expiry time, the only choice is to exercise the option
	float sum = 0;
	float discount_eu = exp(-1*discount_rate*expiry_time/365 );
	
	var_eu = 0;
	float option_value = 0;
	for ( int path = 0; path < S.size(); path++ )
	{
		cash_flow[path] = fmaxf( strike_price - S[path][expiry_index], 0.0); //put
		cash_flow_eu[path] = cash_flow[path];
		cash_flow_am[path] = cash_flow[path];
		
		option_value = cash_flow[path]*discount_eu;
		sum = sum + option_value;
		var_eu += pow(option_value,2);
	}
	
	// find the value of the european option
	european_option_value  = sum/S.size();
	var_eu = (var_eu - pow(european_option_value, 2) )/S.size();
	
	#ifdef SAVE_DATA_TO_LOG
		fprintf(fileIO.log_file, "\nCASH FLOW AT t = %g days (expiry date)\n----------------------------------\n\n",
		expiry_time );
		
		fprintf(fileIO.log_file,"%-7s  %-10s  %-10s  %-10s  %-10s\n", "Path No", "S(t)", "Cash Flow", "EU_option", "OXB");
		fprintf(fileIO.log_file,"%-7s  %-10s  %-10s  %-10s  %-10s\n", "-------", "----", "---------", "---------", "---");
		for ( int path = 0; path < cash_flow.size(); path++ )
		{
			fprintf(fileIO.log_file, "%-7d  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n",
			path+1, S[path][expiry_index], cash_flow[path], cash_flow[path]*discount_eu, optimal_exercise_boundary[path]*del_t*365);
		}
		fprintf(fileIO.log_file, "\n\n******************************************************\n");
		fprintf(fileIO.log_file, "* Value of Corresponding European Option: %.4f *\n", european_option_value);
		fprintf(fileIO.log_file, "******************************************************\n");
	#endif
	
	vector<float> x;
	vector<float> y;
	vector<int> imp_indices; // In-the-Money-Path indices list
	vector<float> h;
	
	// discount for merican counterpart
	float discount = exp(-1*discount_rate*del_t );
	
	float put_value = 0;
	
	// for all other times when the option can be exercised, we comapre the
	// value of exercising and continuation value to find optimal exercise boundary  
	for ( int time = expiry_index-1; time >= 1; time-- ) // move back in time
	{
		x.clear(); y.clear();
		imp_indices.clear();
		
		// find in the money paths
		for ( int path = 0; path < S.size(); path++ )
		{
			put_value = fmaxf( strike_price - S[path][time], 0.0); //put
			
			if ( put_value > 0 )
			{
				x.push_back( S[path][time] );
				y.push_back( cash_flow_am[path]*exp(-1*discount_rate*del_t*(optimal_exercise_boundary[path]-time) ) );
				imp_indices.push_back( path );	
			}
			cash_flow[path] = put_value;
		}
		
		/********* USING LSM as boundary ***********
		vector<float> g;
		least_squares.find_linear_fit( x, y, g, num_laguerre_poly);
		get_continuation_value_ch( g, h, x );*/
		
		/*float norm = 0;
		for( int z = 0; z < h.size(); z++ )
		{
			norm = norm + fabs(y[z] - h[z]);
		}
		printf("\ntime: %gd, error-norm: %.3g, avg-error: %.3g", time*del_t*365, norm, norm/h.size());*/
		
		/********* USING BSE as boundary ************/
		get_black_scholes_continuation_value( x, time, h);
		
		for ( int i = 0; i < imp_indices.size(); i++ )
		{
			if ( cash_flow[imp_indices[i]] > h[i] )
			{
				optimal_exercise_boundary[ imp_indices[i] ] = time;
				cash_flow_am[imp_indices[i]] = fmaxf(strike_price - S[imp_indices[i]][ time ], 0.0);
			}
		}

		#ifdef SAVE_DATA_TO_LOG
			fprintf(fileIO.log_file, "\n\nCASH FLOW AT t = %.2f days \n----------------------------------\n\n",
			del_t*time*365 );
			
			int index_inception = 0;
			bool flag = 0;
			fprintf(fileIO.log_file,"%-7s  %-10s  %-10s  %-10s  %-10s  %-10s\n", "Path No", "S(t)", "Discounted", "Cash Flow", "Continue", "OXB");
			fprintf(fileIO.log_file,"%-7s  %-10s  %-10s  %-10s  %-10s  %-10s\n", "-------", "----", "----------", "---------", "--------", "---");
			for ( int path = 0; path < cash_flow.size(); path++ )
			{
				flag = (imp_indices[index_inception] == path);
				fprintf(fileIO.log_file, "%-7d  %s%-9.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n",
				path+1, (strike_price > S[path][time])?"*":" ",
				S[path][time], (flag)? y[index_inception]: -1.0, cash_flow[path],
				( flag )? h[index_inception]: -1.0, optimal_exercise_boundary[path]*del_t*365);
				
				if (flag) index_inception++;
			}
			//fprintf(fileIO.log_file, "\n\n* Linear least Squares fit with [X,Y]= [S(t),Discounted]:\n");
			//fprintf(fileIO.log_file, "* y = %.4f*1 + %.4f*x + %.4f*(2*x^2 - 1) + %.4f*(4*x^3 - 3*x)",
			//g[0], g[1], g[2], g[3]);
			
		#endif
	}
	
	sum = 0;
	var_am = 0;
	for ( int i = 0; i < optimal_exercise_boundary.size(); i++ )
	{
		cash_flow_am[i] = fmaxf(strike_price - S[i][optimal_exercise_boundary[i]], 0.0); 
		
		discount = exp(-1*discount_rate*optimal_exercise_boundary[i]*del_t );
		option_value = cash_flow_am[i]*discount; 
		sum = sum + option_value;
		var_am += pow(option_value, 2);
	}
	american_option_value  = sum/S.size();
	var_am = (var_am - pow(american_option_value, 2) )/S.size();
	
	/*printf("\n\nFINAL RESULTS\n------------------------\n");
	printf("%7s  %12s  %12s  %12s  %12s  %12s\n", "Path no", "Eu_Cash", "Eu_discount", "Am_Cash", "Excercise_at", "Am_discount");
	printf("%7s  %12s  %12s  %12s  %12s  %12s\n", "-------", "-------", "-----------", "-------", "------------", "-----------");
	for ( int i = 0; i < optimal_exercise_boundary.size(); i++ )
	{
		printf("%7d  %12.4f  %12.4f  %12.4f  %12.4f  %12.4f\n", i+1, cash_flow_eu[i], discount_eu, cash_flow_am[i],
		optimal_exercise_boundary[i]*del_t*365, exp(-1*discount_rate*optimal_exercise_boundary[i]*del_t ) );
	}*/
	
	vector<float> sc; sc.push_back(S[0][0]);
	vector<float> hc;
	get_black_scholes_continuation_value( sc, 0, hc);
	printf("\n\nSUMMARY RESULTS FOR CPU\n--------------------------\n");
	printf("  i) American Option:\n");
	printf("%40s:   %.6f \n", "Valuation at t=0", american_option_value);
	printf("%40s:   %.6f \n", "Std dev of the samples", sqrt(var_am) );
	float delta_am = 1.96*sqrt(var_am/S.size())/american_option_value;
	printf("%40s:   %.3g \% (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", 100*delta_am/(1-delta_am) );
	printf("\n ii) European Option:\n");
	printf("%40s:   %.6f \n", "Valuation at t=0 by Black-Scholes", hc[0]);
	printf("%40s:   %.6f \n", "Valuation at t=0 by Monte-Carlo", european_option_value);
	printf("%40s:   %.6f \n", "Std dev of the samples", sqrt(var_eu) );
	float delta_eu = 1.96*sqrt(var_eu/S.size())/european_option_value;
	printf("%40s:   %.3g \% (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", 100*delta_eu/(1-delta_eu) );
	printf("\niii) Early Exercise Value: %g\n", american_option_value - european_option_value);
	#ifdef SAVE_DATA_TO_LOG
			fprintf(fileIO.log_file, "\n\nFINAL RESULTS\n------------------------\n");
			fprintf(fileIO.log_file, "  i) American Option:\n");
			fprintf(fileIO.log_file, "%40s:   %g \n", "Valuation at t=0", american_option_value);
			fprintf(fileIO.log_file, "%40s:   %g \n", "Std dev of the samples", sqrt(var_am) );
			fprintf(fileIO.log_file, "%40s:   %g (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", delta_am/(1-delta_am) );
			fprintf(fileIO.log_file, "\n ii) European Option:\n");
			fprintf(fileIO.log_file, "%40s:   %g \n", "Valuation at t=0", european_option_value);
			fprintf(fileIO.log_file, "%40s:   %g \n", "Std dev of the samples", sqrt(var_eu) );
			fprintf(fileIO.log_file, "%40s:   %g (w.r.t. true mean)\n", "Maximum rel error (95% confidence)", delta_eu/(1-delta_eu) );
			fprintf(fileIO.log_file, "\niii) Early Exercise Value: %g\n", american_option_value - european_option_value);
			//fprintf(fileIO.log_file, "******************************************************\n");
	#endif
}

void stock_simulation::get_continuation_value_lg( vector<float>& g, vector<float>& h, vector<float>& x )
{
	h.clear();
	h.resize( x.size(), g[0] );
	float weight = 0;
	
	for ( int i = 0; i < x.size(); i++ )
	{
		weight = exp( -x[i]/2.0);
		for ( int j = 1; j < g.size(); j++ )
		{
			h[i] += weight*least_squares.get_Laguerre_Polynomial(x[i], j-1);
		}
	}
}

void stock_simulation::get_continuation_value_ch( vector<float>& g, vector<float>& h, vector<float>& x )
{
	h.clear();
	h.resize( x.size(), 0 );
	
	for ( int i = 0; i < x.size(); i++ )
	{
		for ( int j = 0; j < g.size(); j++ )
		{
			h[i] += g[j]*least_squares.get_Chebyshev_Polynomial(x[i], j);
		}
	}
}

void stock_simulation::get_black_scholes_continuation_value( vector<float>& x, float time, vector<float> &h)
{
	h.clear();
	h.resize( x.size(), 0 );
	float t = time*del_t;
	
	float d1, d2, den;
	float ttm = (expiry_time - t)/365;
	for ( int i = 0; i < x.size(); i++ )
	{
		d1 = log(x[i]/strike_price) + ( discount_rate + 0.5*pow(volatility, 2) )*ttm;
		d2 = log(x[i]/strike_price) + ( discount_rate - 0.5*pow(volatility, 2) )*ttm;
		den = volatility*sqrt( ttm );
		d1 = d1/den;
		d2 = d2/den;

		h[i] = strike_price*exp(-1*discount_rate*ttm)*phi(-1*d2) - x[i]*phi(-1*d1);
		//printf("d1: %g, d2: %g, den: %g, phi(-1*d2): %g, phi(-1*d1): %g, h[i]: %g, x[i]: %g\n",
		//d1, d2, den, phi(-1*d2), phi(-1*d1), h[i], x[i]);
	}
} 

float stock_simulation::phi( float x)
{
	return 0.5*(1 + erf(x/sqrt(2)));
}

// data about resource usage

void stock_simulation::get_resource_usage( FILE* out)
{
	double current_time = get_wall_time();
	double diff_time = (current_time - start_time);

	struct rusage usage;
	int who = RUSAGE_SELF;
	int result = getrusage(who, &usage);
	vector<string> arr;
	
	int VmPeak, VmSize, VmLck, VmHWM, VmRSS, VmData, VmStk, VmExe, VmLib, VmPTE, Threads;

	//process_status = "";
	bool status_available = true;
	ifstream status;
	status.open("/proc/self/status");
	if (!status)
	{
		printf("%s\n", "Could not open file for reading status\n");
		status_available = false;
	}
	if( status_available )
	{
		char str[255];
		while (!status.eof())
		{
			status.getline(str, 255);
			//printf("%s\n", str);
			arr.clear();
			line2arr(str, &arr, (char*)"\t: ");
			//printf("arr[0] = %s, arr[1] = %s\n", arr[0].c_str(), arr[1].c_str());
			if ( arr[0] == "VmPeak" )
				VmPeak = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmSize" )
				VmSize = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmLck" )
				VmLck = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmHWM" )
				VmHWM = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmRSS" )
				VmRSS = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmData" )
				VmData = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmStk" )
				VmStk = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmExe" )
				VmExe = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmLib" )
				VmLib = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "VmPTE" )
				VmPTE = atoi((char*)arr[1].c_str());
			else if ( arr[0] == "Threads" )
				Threads = atoi((char*)arr[1].c_str());
		}
	}
	status.close();
	
	double total_user_cpu_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec*pow(10,-6),
	       total_syst_cpu_time = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec*pow(10,-6);
	double total_cpu_process_time = total_user_cpu_time + total_syst_cpu_time;
	double num_cores = sysconf( _SC_NPROCESSORS_ONLN );
	
	if ( result != -1 )
	{
		fprintf(out,"\nRESOURCE USGAE DETAILS\n-------------------------------\n");	
		fprintf(out, "\n%50s:  %.4f sec\n%50s:  %.4f sec\n%50s:  %.2f\%\n%50s:  %.2f Sec\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %.2f megabyte\n%50s:  %d\n%50s:  %d\n%50s:  %d\n%50s:  %d\n%50s:  %d\n%50s:  %d\n%50s:  %d\n%50s:  %d\n",
		"Total user CPU-time used", total_user_cpu_time, 
		"Total system CPU-time used", total_syst_cpu_time,
		"Percent CPU used by this process", 100*total_cpu_process_time/(num_cores*diff_time),   
		"This application has been running for", diff_time,
		"Peak virtual memory size", 0.000976562*VmPeak,
		"Current virtual memory size", 0.000976562*VmSize,
		"Current locked memory size", 0.000976562*VmLck,
		"Peak resident set size", 0.000976562*VmHWM,
		"Current resident set size", 0.000976562*VmRSS,
		"Size of Data", 0.000976562*VmData,
		"Size of stack", 0.000976562*VmStk,
		"Size of text segments", 0.000976562*VmExe,
		"Shared library code size", 0.000976562*VmLib,
		"Page table entries size (since Linux 2.6.10)", 0.000976562*VmPTE,
		"Number of Threads", Threads,
		"page reclaims (soft page faults) no I/O",usage.ru_minflt,   
		"page faults (hard page faults) required I/O", usage.ru_majflt,
		"Number of Swaps", usage.ru_nswap,
		"Num-times the file system had to perform input", usage.ru_inblock,
		"Num-times the file system had to perform output", usage.ru_oublock,
		"Number of voluntary context switches", usage.ru_nvcsw,
		"Number of involuntary context switches", usage.ru_nivcsw );
	}
	else
	 	fprintf(out,"### There was an error retreiving the stats!!!!!\n");
}

void stock_simulation::line2arr (char* str, vector<string>* arr, char *tokenizer)
{	
	string ts;
	char* tok;
	(*arr).clear();
	tok = strtok(str,tokenizer);
	while ( tok != NULL )
	{
		//printf("%s", tok);
		ts.assign(tok);
		(*arr).push_back(ts);
		tok = strtok(NULL,tokenizer);
	}
}

double stock_simulation::get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

//=================================//=================================//=================================//
/* For European options */
double stock_simulation::EuropeanOptionsEndCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

void stock_simulation::EuropeanOptionsMonteCarloCPU(
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
        double callValue = EuropeanOptionsEndCallValue(S, X, sample, MuByT, VBySqrtT);
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
/* End European Options */
