#ifndef _STOCK_SIMULATION_H_
#define _STOCK_SIMULATION_H_

#ifdef __cplusplus
//extern "C" {
#endif

#include <fstream>
#include "../timer/timer.h"
#include "../MonteCarlo_structs.h"

class FileIO
{
	char *outputFileName;
	public:
	FileIO( ); 
	
	void readInputFile(char *inputFilename, InputData &indata);

	void writeOutputFile();
};

//	END

#ifdef __cplusplus
//}
#endif

#endif
