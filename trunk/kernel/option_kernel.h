#ifndef _OPTION_KERNEL_H_
#define _OPTION_KERNEL_H_

#include <vector>
#include <math.h>

#include "../util/random/random_normal.h"
#include "../util/timer/timer.h"
#include "../util/FileIO/FileIO.h"
#include "../util/MonteCarlo_structs.h"
//#include "../util/regress/regress_CPU.h"

using namespace std;

extern "C" void _gpu_find_option_values_using_normrand();
extern "C" void _gpu_find_option_values_using_curand();
extern "C" void _gpu_find_option_values_using_normrand_v2();
extern "C" void _gpu_find_option_values_using_curand_v2();


#endif
