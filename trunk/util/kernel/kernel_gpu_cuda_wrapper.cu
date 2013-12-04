//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"									// (in main directory)

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"					// (in current directory)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/device/device.h"					// (in specified directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./prepare_kernel.cu"							// (in the current directory)
#include "./extract_kernel.cu"							// (in the current directory)
#include "./reduce_kernel.cu"							// (in the current directory)
#include "./srad_kernel.cu"								// (in the current directory)
#include "./srad2_kernel.cu"							// (in the current directory)
#include "./compress_kernel.cu"							// (in the current directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_cuda_wrapper(fp* image,											// input image
						int Nr,												// IMAGE nbr of rows
						int Nc,												// IMAGE nbr of cols
						long Ne,											// IMAGE nbr of elem
						int niter,											// nbr of iterations
						fp lambda,											// update step size
						long NeROI,											// ROI nbr of elements
						int* iN,
						int* iS,
						int* jE,
						int* jW,
						int iter,											// primary loop
						int mem_size_i,
						int mem_size_j,
						int mem_size_single)
{

	//======================================================================================================================================================150
	// 	GPU VARIABLES
	//======================================================================================================================================================150

	// CUDA kernel execution parameters
	dim3 threads;
	int blocks_x;
	dim3 blocks;
	dim3 blocks2;
	dim3 blocks3;

	// memory sizes
	int mem_size;															// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* d_sums;															// partial sum
	fp* d_sums2;
	int* d_iN;
	int* d_iS;
	int* d_jE;
	int* d_jW;
	fp* d_dN; 
	fp* d_dS; 
	fp* d_dW; 
	fp* d_dE;
	fp* d_I;																// input IMAGE on DEVICE
	fp* d_c;

	//======================================================================================================================================================150
	// 	ALLOCATE AND COPY DATA TO GPU
	//======================================================================================================================================================150

	// allocate memory for entire IMAGE on DEVICE
	mem_size = sizeof(fp) * Ne;												// get the size of float representation of input IMAGE
	cudaMalloc(	(void **)&d_I, 
				mem_size);													//

	// allocate memory for coordinates on DEVICE
	cudaMalloc(	(void **)&d_iN, 
				mem_size_i);												//
	cudaMemcpy(	d_iN, 
				iN, 
				mem_size_i, 
				cudaMemcpyHostToDevice);									//
	cudaMalloc(	(void **)&d_iS, 
				mem_size_i);												// 
	cudaMemcpy(	d_iS, 
				iS, 
				mem_size_i, 
				cudaMemcpyHostToDevice);									//
	cudaMalloc(	(void **)&d_jE, 
				mem_size_j);												//
	cudaMemcpy(	d_jE, 
				jE, 
				mem_size_j, 
				cudaMemcpyHostToDevice);									//
	cudaMalloc(	(void **)&d_jW, 
				mem_size_j);												// 
	cudaMemcpy(	d_jW, 
				jW, 
				mem_size_j, 
				cudaMemcpyHostToDevice);									//

	// allocate memory for partial sums on DEVICE
	cudaMalloc(	(void **)&d_sums, 
				mem_size);													//
	cudaMalloc(	(void **)&d_sums2, 
				mem_size);													//

	// allocate memory for derivatives
	cudaMalloc(	(void **)&d_dN, 
				mem_size);													// 
	cudaMalloc(	(void **)&d_dS, 
				mem_size);													// 
	cudaMalloc(	(void **)&d_dW, 
				mem_size);													// 
	cudaMalloc(	(void **)&d_dE, 
				mem_size);													// 

	// allocate memory for coefficient on DEVICE
	cudaMalloc(	(void **)&d_c, 
				mem_size);													// 

	checkCUDAError("setup");

	//======================================================================================================================================================150
	// 	KERNEL EXECUTION PARAMETERS
	//======================================================================================================================================================150

	// all kernels operating on entire matrix
	threads.x = NUMBER_THREADS;												// define the number of threads in the block
	threads.y = 1;
	blocks_x = Ne/threads.x;
	if (Ne % threads.x != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks.x = blocks_x;													// define the number of blocks in the grid
	blocks.y = 1;

	printf("max # of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)blocks.x, (int)threads.x);

	//======================================================================================================================================================150
	// 	COPY INPUT TO CPU
	//======================================================================================================================================================150

	cudaMemcpy(d_I, image, mem_size, cudaMemcpyHostToDevice);

	//======================================================================================================================================================150
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//======================================================================================================================================================150

	extract<<<blocks, threads>>>(	Ne,
									d_I);

	checkCUDAError("extract");

	//======================================================================================================================================================150
	// 	COMPUTATION
	//======================================================================================================================================================150

	printf("Iterations Progress: ");

	// execute main loop
	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

	printf("%d ", iter);
	fflush(NULL);

		// execute square kernel
		prepare<<<blocks, threads>>>(	Ne,
										d_I,
										d_sums,
										d_sums2);

		checkCUDAError("prepare");

		// performs subsequent reductions of sums
		blocks2.x = blocks.x;												// original number of blocks
		blocks2.y = blocks.y;
		no = Ne;															// original number of sum elements
		mul = 1;															// original multiplier

		while(blocks2.x != 0){

			checkCUDAError("before reduce");

			// run kernel
			reduce<<<blocks2, threads>>>(	Ne,
											no,
											mul,
											d_sums, 
											d_sums2);

			checkCUDAError("reduce");

			// update execution parameters
			no = blocks2.x;													// get current number of elements
			if(blocks2.x == 1){
				blocks2.x = 0;
			}
			else{
				mul = mul * NUMBER_THREADS;									// update the increment
				blocks_x = blocks2.x/threads.x;								// number of blocks
				if (blocks2.x % threads.x != 0){							// compensate for division remainder above by adding one grid
					blocks_x = blocks_x + 1;
				}
				blocks2.x = blocks_x;
				blocks2.y = 1;
			}

			checkCUDAError("after reduce");

		}

		checkCUDAError("before copy sum");

		// copy total sums to device
		mem_size_single = sizeof(fp) * 1;
		cudaMemcpy(	&total, 
					d_sums, 
					mem_size_single, 
					cudaMemcpyDeviceToHost);
		cudaMemcpy(	&total2, 
					d_sums2, 
					mem_size_single, 
					cudaMemcpyDeviceToHost);

		checkCUDAError("copy sum");

		// calculate statistics
		meanROI	= total / fp(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / fp(NeROI)) - meanROI2;						// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		// execute srad kernel
		srad<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									d_iN,									// indices of North surrounding pixels
									d_iS,									// indices of South surrounding pixels
									d_jE,									// indices of East surrounding pixels
									d_jW,									// indices of West surrounding pixels
									d_dN,									// North derivative
									d_dS,									// South derivative
									d_dW,									// West derivative
									d_dE,									// East derivative
									q0sqr,									// standard deviation of ROI 
									d_c,									// diffusion coefficient
									d_I);									// output image

		checkCUDAError("srad");

		// execute srad2 kernel
		srad2<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									d_iN,									// indices of North surrounding pixels
									d_iS,									// indices of South surrounding pixels
									d_jE,									// indices of East surrounding pixels
									d_jW,									// indices of West surrounding pixels
									d_dN,									// North derivative
									d_dS,									// South derivative
									d_dW,									// West derivative
									d_dE,									// East derivative
									d_c,									// diffusion coefficient
									d_I);									// output image

		checkCUDAError("srad2");

	}

	printf("\n");

	//======================================================================================================================================================150
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//======================================================================================================================================================150

	compress<<<blocks, threads>>>(	Ne,
									d_I);

	checkCUDAError("compress");

	//======================================================================================================================================================150
	// 	COPY RESULTS BACK TO CPU
	//======================================================================================================================================================150

	cudaMemcpy(image, d_I, mem_size, cudaMemcpyDeviceToHost);

	checkCUDAError("copy back");

	// int i;
	// for(i=0; i<100; i++){
		// printf("%f ", image[i]);
	// }

	//======================================================================================================================================================150
	// 	FREE MEMORY
	//======================================================================================================================================================150

	cudaFree(d_I);
	cudaFree(d_c);
	cudaFree(d_iN);
	cudaFree(d_iS);
	cudaFree(d_jE);
	cudaFree(d_jW);
	cudaFree(d_dN);
	cudaFree(d_dS);
	cudaFree(d_dE);
	cudaFree(d_dW);
	cudaFree(d_sums);
	cudaFree(d_sums2);

	//======================================================================================================================================================150
	// 	End
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
