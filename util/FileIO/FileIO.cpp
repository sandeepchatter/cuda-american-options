#include <stdlib.h>
#include <stdio.h>
#include "./FileIO.h"

using namespace std;

FileIO::FileIO( )
{
}

void FileIO::readInputFile(char *inputFileName, InputData &indata)
{
	ifstream in;
	in.open(inputFileName);
	if (!in)
	{
		printf("Could not open file %s for reading options\n", inputFileName);
		exit(1);
	}
	while (!in.eof())
	{
		char str[255];
		in.getline(str, 255);
		string line( str );
		
		if (strcmp(str, ".end")  == 0 )
			break;
		if ( !line.empty() )
		{			
			char * param = strtok(str, " =");
			char * str_option = strtok(NULL, " =");
		
			if ( strcmp(param, "*") == 0 )
				continue;
			
			if (strcmp(param, "num_paths") == 0)
			{
				indata.num_paths = atoi(str_option);
			}
			else if (strcmp(param, "num_time_steps") == 0)
			{
				indata.num_time_steps = atoi(str_option);
			}
			else if (strcmp(param, "discount_rate") == 0)
			{
				indata.discount_rate = atof(str_option);
			}
			else if (strcmp(param, "random_seed") == 0)
			{
				indata.random_seed = atoi(str_option);
			}
			else if (strcmp(param, "dividend") == 0)
			{
				indata.dividend = atof(str_option);
			}
			else if (strcmp(param, "expiry_time") == 0)
			{
				indata.expiry_time = atof(str_option);
			}
			else if (strcmp(param, "S_0") == 0)
			{
				indata.S_0 = atof(str_option);
			}
			else if (strcmp(param, "volatility") == 0)
			{
				indata.volatility = atof(str_option);
			}
			else if (strcmp(param, "strike_price") == 0)
			{
				indata.strike_price = atof(str_option);
			}
			else if (strcmp(param, "num_laguerre_poly") == 0)
			{
				indata.num_laguerre_poly = atoi(str_option);
			}
			else if (strcmp(param, "num_paths_per_thread") == 0)
			{
				indata.num_paths_per_thread = atoi(str_option);
			}
		}
	}
}

void FileIO::prepare_log_buffer()
{
	time_t rawtime;
  	struct tm * timeinfo;
  	time ( &rawtime );
  	timeinfo = localtime ( &rawtime );
  	char* timeinfo_str = asctime(timeinfo);
  	char filename[50];
  	sprintf(filename, "./output/%u.log", rawtime);
  	//printf ( "Current local time and date: %s and epoch: %u", str, rawtime);
  	
  	log_file = fopen(filename,"a");
	if ( log_file == NULL )
	{
		perror ("Error opening file\n");
		exit(1);
	}
	
	setvbuf ( log_file , NULL , _IOFBF , 10240 );
	fprintf(log_file, "TIMESTAMP: %s\n", timeinfo_str );
}


