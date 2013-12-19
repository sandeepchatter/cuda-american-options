#	****** EXAMPLE ****** 

# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

#	CONSTANTS

#	****** CUDA ****** 

#	NVIDIA CUDA header/library [NVIDIA Toolkit]

CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := -L$(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
	ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
		CUDA_LIB_DIR := -L$(CUDA_DIR)/lib64
	endif
endif

CUDA_LIB =	-lcuda \
			-lcudart \
			-lcurand
CUDA_FLAG = -arch sm_20

#	END

#	****** OPENCL ****** 

#	NVIDIA OpenCL header/library [NVIDIA SDK]

# OCL_DIR =/af14/lgs9a/Programs/NVIDIA/SDK
# OCL_INC_DIR = -I$(OCL_DIR)/OpenCL/common/inc
# OCL_LIB_DIR = -L$(OCL_DIR)/OpenCL/common/lib

#	AMD OpenCL header/library [AMD SDK]

# OCL_DIR = /af14/lgs9a/Programs/AMD/SDK
# OCL_INC_DIR = -I$(OCL_DIR)/include/ 
# OCL_LIB_DIR = -L$(OCL_DIR)/lib/x86/
# ifeq ($(shell uname -m), x86_64)
    # ifeq ($(shell if test -d $(OCL_DIR)/lib/x86_64/; then echo T; else echo F; fi), T)
    	# OCL_LIB_DIR = -L$(OCL_DIR)/lib/x86_64/
    # endif
# endif

#	END

#	COMPILER

#	ICC

# C_C = icc
# OMP_FLAG = -openmp

#	GCC

# C_C = gcc
# OMP_LIB = -lgomp
# OMP_FLAG = -fopenmp
# OCL_LIB = -lOpenCL

#	****** NVCC ******

CUD_C = /usr/local/cuda-5.5/bin/nvcc
# OMP_FLAG = 	-Xcompiler paste_one_here

#	PGCC

# C_C = pgcc
# OMP_FLAG = -mp
# ACC_FLAG = 	-ta=nvidia \
			# -Minfo \
			# -Mbounds

#	END

#	END

#	****** EXECUTABLES (LINK OBJECTS TOGETHER INTO BINARY) ****** 

./ao.exe:	./main.o \
			./stock_simulation/stock_simulation.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./kernel/option_kernel.o \
			./util/device/device.o \
			./util/timer/timer.o \
			./util/regress/regress_CPU.o \
			./util/regress/LinearAlgebraSubroutines.o \
			./util/FileIO/FileIO.o
	$(CUD_C)	./main.o \
				./stock_simulation/stock_simulation.o \
				./kernel/kernel_gpu_cuda_wrapper.o \
				./kernel/option_kernel.o \
				./util/device/device.o \
				./util/timer/timer.o \
				./util/regress/regress_CPU.o \
				./util/regress/LinearAlgebraSubroutines.o \
				./util/FileIO/FileIO.o \
				-lm -g\
				$(CUDA_LIB_DIR) \
				$(CUDA_LIB) \
				$(OMP_LIB)

#	******  OBJECTS (COMPILE SOURCE FILES INTO OBJECTS) ****** 

#	MAIN FUNCTION

./main.o:	./main.h \
			./stock_simulation/stock_simulation.h \
			./main.cpp
	$(CUD_C)	./main.cpp \
				-c -g\
				-o ./main.o \
				-O3

#	KERNELS

./kernel/kernel_gpu_cuda_wrapper.o:	./kernel/kernel_gpu_cuda_wrapper.h \
									./kernel/kernel_gpu_cuda_wrapper.cu
	$(CUD_C)	./kernel/kernel_gpu_cuda_wrapper.cu \
				-c -g\
				-o ./kernel/kernel_gpu_cuda_wrapper.o \
				-O3 
				$(CUDA_FLAG)

./kernel/option_kernel.o: ./kernel/option_kernel.h \
                          ./kernel/option_kernel.cu
	$(CUD_C)	./kernel/option_kernel.cu \
				-c -g \
				-o ./kernel/option_kernel.o \
				-O3 \
				$(CUDA_FLAG)

#	UTILITIES

./util/device/device.o:	./util/device/device.h \
						./util/device/device.cu
	$(CUD_C)	./util/device/device.cu \
				-c -g\
				-o ./util/device/device.o \
				-O3 \
				$(CUDA_FLAG)

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.cpp
	$(CUD_C)	./util/timer/timer.cpp \
				-c -g\
				-o ./util/timer/timer.o \
				-O3

./util/regress/regress_CPU.o:	./util/regress/regress_CPU.h \
								./util/regress/LinearAlgebraSubroutines.h \
								./util/regress/LinearAlgebraSubroutines.cpp \
								./util/regress/regress_CPU.cpp
	$(CUD_C)	./util/regress/regress_CPU.cpp \
				-c -g\
				-o ./util/regress/regress_CPU.o \
				-O3

#./util/regress/LinearAlgebraSubroutines.o:	./util/regress/LinearAlgebraSubroutines.h \
#											./util/regress/LinearAlgebraSubroutines.cpp
#	$(CUD_C)	./util/regress/LinearAlgebraSubroutines.cpp \
#				-c \
#				-o ./util/regress/LinearAlgebraSubroutines.o \
#				-O3

./util/FileIO/FileIO.o:	./util/FileIO/FileIO.h \
						./util/FileIO/FileIO.cpp
	$(CUD_C)	./util/FileIO/FileIO.cpp \
				-c -g\
				-o ./util/FileIO/FileIO.o \
				-O3


# PACKAGES

./stock_simulation/stock_simulation.o:	./util/random/random_normal.h \
										./util/regress/regress_CPU.cpp \
										./stock_simulation/stock_simulation.h \
										./stock_simulation/stock_simulation.cpp
	$(CUD_C)	./stock_simulation/stock_simulation.cpp \
				-c -g\
				-o ./stock_simulation/stock_simulation.o \
				-O3
#	END

#	DELETE

clean:
	rm	./*.o ./*.out \
		./kernel/*.o \
		./util/device/*.o \
		./util/timer/*.o \
		./util/regress/regress_CPU.o \
		./stock_simulation/stock_simulation.o \

#	END
