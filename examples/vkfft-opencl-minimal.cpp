/* Example showing the use of vkFFT vs cuFFT for a single FFT transform
This should compile (assuming vkfft.h is in the include paths) with e.g.:

* macOS:
clang++ -std=c++11  -Wl,-framework,OpenCL -I../src/ vkfft-opencl-minimal.cpp

* linux:
g++ -I../src/ -g -lOpenCL vkfft-opencl-minimal.cpp
*/

// We are using the OpenCL backend
#define VKFFT_BACKEND 3

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "vkFFT.h"

// Complex data type
typedef cl_float2 Complex;

void runTest(int argc, char** argv);

#define ARRAY_SIZE 2048

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
    //printf("vkFFT-opencl is starting...\n");

    // Allocate host array
    Complex* arr0 = (Complex*)malloc(sizeof(Complex) * ARRAY_SIZE * ARRAY_SIZE);

    // Init arrays
    for (unsigned int i = 0; i < ARRAY_SIZE *ARRAY_SIZE; ++i) {
        arr0[i]= { (float)(rand() / (float)RAND_MAX - 0.5f + sin((float)i * 2 * 3.141592653589f / ARRAY_SIZE * 7.7)) , 0};
    }
    int mem_size = sizeof(Complex) * ARRAY_SIZE*ARRAY_SIZE;

    // Init OpenCL
  	cl_int res = CL_SUCCESS;
  	cl_platform_id platform;
		cl_device_id device;
		cl_context context ;
		cl_command_queue commandQueue;

  	res = clGetPlatformIDs(1, &platform, NULL);

//    char platform_name[128];
//    size_t ret_param_size = 0;
//    res = clGetPlatformInfo(platform, CL_PLATFORM_NAME,
//            sizeof(platform_name), platform_name,
//            &ret_param_size);
//    printf("Platform: %s\n", platform_name);


    res = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

//    char device_name[128];
//    res = clGetDeviceInfo(device, CL_DEVICE_NAME,
//            sizeof(device_name), device_name,
//            &ret_param_size);
//    printf("Device: %s\n", device_name);


    context = clCreateContext(NULL, 1, &device, NULL, NULL, &res);
//  	cout<<"clCreateContext:"<<res<<endl;
    commandQueue = clCreateCommandQueue(context, device, 0, &res);
//  	cout<<"clCreateCommandQueue:"<<res<<endl;

    // Allocate device memory
    cl_mem buffer;
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, 0, &res);

    //////////////////////////////////////////////// vkFFT ///////////////////////////////////////
//    printf("Using vkFFT (OpenCL backend)\n");

    // Perform the same FFT using vkFFT
    VkFFTConfiguration configuration = {};

    configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
    configuration.size[0] = ARRAY_SIZE;
    configuration.size[1] = ARRAY_SIZE;
    configuration.size[2] = 1;

    configuration.platform = &platform;
    configuration.context = &context;
    configuration.device = &device;


    configuration.buffer = &buffer;
    uint64_t bufferSize = mem_size;
    configuration.bufferSize = &bufferSize;
//    cout << CL_INVALID_COMMAND_QUEUE <<" "<<
//            CL_INVALID_CONTEXT <<" "<<
//            CL_INVALID_MEM_OBJECT <<" "<<
//            CL_INVALID_VALUE <<" "<<
//            CL_INVALID_EVENT_WAIT_LIST <<" "<<
//            CL_MEM_OBJECT_ALLOCATION_FAILURE <<" "<<
//            CL_OUT_OF_HOST_MEMORY <<" "<<endl;

		cout <<"clEnqueueWriteBuffer:"<< clEnqueueWriteBuffer(commandQueue, buffer, CL_TRUE, 0, bufferSize, arr0, 0, NULL, NULL)<<endl;

		//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.
    VkFFTApplication app = {};
		res = initializeVkFFT(&app, configuration);
//		std::cout <<"initializeVkFFT:"<<res<< std::endl;

    VkFFTLaunchParams par;
    par.commandQueue = &commandQueue;
    par.buffer =  &buffer;
    //par.inputBuffer = app->configuration.inputBuffer;
    //par.outputBuffer = app->configuration.outputBuffer;

		VkFFTAppend(&app, -1, &par);
		VkFFTAppend(&app, 1, &par);

    deleteVkFFT(&app);

    //////////////////////////////////////////////// vkFFT End ///////////////////////////////////

    // cleanup
    free(arr0);
    clReleaseMemObject(buffer);
    cout<<"Finished"<<endl;
}
