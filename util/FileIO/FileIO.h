#ifndef _FILEIO_H_
#define _FILEIO_H_

#include <fstream>
#include "../timer/timer.h"
#include "../MonteCarlo_structs.h"

//#define SAVE_DATA_TO_LOG

#define PRINT_DATA_TO_CONSOLE

class FileIO
{
	char *outputFileName;
	
	public:
	FILE* log_file;
	
	FileIO( ); 
	
	void readInputFile(char *inputFilename, InputData &indata);

	void prepare_log_buffer();
};


#endif
