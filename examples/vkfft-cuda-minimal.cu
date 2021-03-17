/* Example showing the use of vkFFT vs cuFFT for a single FFT transform
This should compile (assuming vkfft.h is in the include paths) with e.g.:
CUDA_FLAGS="-O3 -DNDEBUG -L/usr/local/cuda/lib -DVKFFT_BACKEND=1
            -gencode arch=compute_35,code=compute_35 -gencode arch=compute_60,code=compute_60
            -gencode arch=compute_70,code=compute_70 -std=c++11"

nvcc $CUDA_FLAGS -w vkfft-cuda.cu -lcufft -lnvrtc -lcuda -o vkfft-cuda-minimal

*/

// We are using the CUDA backend
#define VKFFT_BACKEND 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include <cufft.h>
#include "vkFFT.h"

// Complex data type
typedef float2 Complex;

void runTest(int argc, char** argv);

#define ARRAY_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test using cuFFT & vkFFT APIs
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
    printf("vkFFT-cuda is starting...\n");

    // Allocate host arrays
    Complex* arr0 = (Complex*)malloc(sizeof(Complex) * ARRAY_SIZE);
    Complex* arr1 = (Complex*)malloc(sizeof(Complex) * ARRAY_SIZE);
    Complex* arr2 = (Complex*)malloc(sizeof(Complex) * ARRAY_SIZE);

    // Init arrays
    for (unsigned int i = 0; i < ARRAY_SIZE; ++i) {
        arr0[i].x = rand() / (float)RAND_MAX - 0.5f + sin((float)i * 2 * 3.141592653589f / ARRAY_SIZE * 7.7);
        arr0[i].y = 0;
        arr1[i].x = arr0[i].x;
        arr1[i].y = 0;
        arr2[i].x = arr0[i].x;
        arr2[i].y = 0;
    }
    int mem_size = sizeof(Complex) * ARRAY_SIZE;
    // Allocate device memory
    Complex* d1;
    cudaMalloc((void**)&d1, mem_size);
    Complex* d2;
    cudaMalloc((void**)&d2, mem_size);
    // Copy host memory to device
    cudaMemcpy(d1, arr1, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d2, arr2, mem_size, cudaMemcpyHostToDevice);

    //////////////////////////////////////////////// CUFFT ///////////////////////////////////////
    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, ARRAY_SIZE, CUFFT_C2C, 1);

    // Transform signal and kernel
    printf("Using cuFFT\n");
    cufftExecC2C(plan, (cufftComplex *)d1, (cufftComplex *)d1, CUFFT_FORWARD);

    //Destroy CUFFT context
    cufftDestroy(plan);


    //////////////////////////////////////////////// vkFFT ///////////////////////////////////////
    printf("Using vkFFT (CUDA backend)\n");

    // Perform the same FFT using vkFFT
    VkFFTConfiguration configuration = {};

    configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
    configuration.size[0] = ARRAY_SIZE; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.
    configuration.size[1] = 1;
    configuration.size[2] = 1;

  	std::cout<<cuInit(0)<<std::endl; // Should not be necessary ?
    int ctdev;cuDeviceGetCount(&ctdev);
	  cudaSetDevice(0);
	  CUdevice dev;
	  std::cout << cuDeviceGet(&dev, 0) <<" "<<ctdev<<std::endl;
	  configuration.device = &dev;

    //CUcontext context;
	  //cuCtxCreate(&vkGPU->context, 0, vkGPU->device);

    configuration.buffer = (void**)&d2;
    uint64_t bufferSize = mem_size;
    configuration.bufferSize = &bufferSize;

		//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.
    VkFFTApplication app = {};
		uint32_t res = initializeVkFFT(&app, configuration);  // TODO
		std::cout <<res<< std::endl;

		VkFFTAppend(&app, -1, NULL);

  	cudaDeviceSynchronize(); // necessary ?
    deleteVkFFT(&app);

    //////////////////////////////////////////////// vkFFT End ///////////////////////////////////

    // Copy device memory to host
    cudaMemcpy(arr1, d1, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(arr2, d2, mem_size, cudaMemcpyDeviceToHost);

    //
    std::ofstream out("results.dat");
    for (unsigned int i = 0; i < ARRAY_SIZE; ++i)
    {
        out << arr0[i].x <<"+"<<arr0[i].x<<"j "
            << arr1[i].x <<"+"<<arr1[i].x<<"j "
            << arr2[i].x <<"+"<<arr2[i].x<<"j "<<std::endl;
    }

    // cleanup memory
    free(arr0);
    free(arr1);
    free(arr2);
    cudaFree(d1);
    cudaFree(d2);
}
