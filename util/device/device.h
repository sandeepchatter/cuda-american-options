#ifndef _DEVICE_H_
#define _DEVICE_H_

#include <stdio.h>					// (in library path known to compiler)		needed by printf

//	SET DEVICE PROTOTYPE

void 
setdevice(void);

//	GET LAST PROTOTYPE

void 
checkCUDAError(const char *msg);

#endif
