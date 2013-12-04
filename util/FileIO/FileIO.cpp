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
			else if (strcmp(param, "seed") == 0)
			{
				indata.seed = atoi(str_option);
			}
			else if (strcmp(param, "dividend") == 0)
			{
				indata.dividend = atof(str_option);
			}
			else if (strcmp(param, "expiry_date") == 0)
			{
				indata.expiry_date = atof(str_option);
			}
			else if (strcmp(param, "S_0") == 0)
			{
				indata.S_0 = atof(str_option);
			}
		}
	}
}

void writeOutputFile()
{
	
}


