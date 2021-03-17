/* PyVkFFT
   (c) 2021- : ESRF-European Synchrotron Radiation Facility
       authors:
         Vincent Favre-Nicolin, favre@esrf.fr
*/

// We use the CUDA backend
#define VKFFT_BACKEND 1

#include <iostream>
#include <fstream>
#include <memory>
#include "vkFFT.h"
typedef float2 Complex;

using namespace std;

extern "C"{
VkFFTConfiguration* make_config(const int, const int, const int, const int, void*);

VkFFTApplication* init_app(const VkFFTConfiguration*);

void fft(VkFFTApplication* app);

void ifft(VkFFTApplication* app);

void free_app(VkFFTApplication* app);

void free_config(VkFFTConfiguration *config);

int test_vkfft_cuda(int);
}

/** Create the VkFFTConfiguration from the array parameters
*
* \param nx, ny, nz: dimensions of the array. The fast axis is x. In the corresponding numpy array,
* this corresponds to a shape of (nz, ny, nx)
* \param fftdim: the dimension of the transform. If nz>1 and fftdim=2, the transform is only made
* on the x and y axes
* \param buffer: pointer to the GPU data array.
* \return: the pointer to the newly created VkFFTConfiguration
*/
VkFFTConfiguration* make_config(const int nx, const int ny, const int nz, const int fftdim, void *buffer)
{
  VkFFTConfiguration *config = new VkFFTConfiguration({});
  config->FFTdim = fftdim;
  config->size[0] = nx;
  config->size[1] = ny;
  config->size[2] = nz;

  cudaSetDevice(0);
  CUdevice *dev = new CUdevice;
	cuDeviceGet(dev, 0);
  config->device = dev;

  // TODO: free pbuf and psize when the config is destroyed (auto_ptr?)

  void ** pbuf = new void*;
  *pbuf = buffer;
  config->buffer = pbuf;

  uint64_t* psize = new uint64_t;
  *psize = (uint64_t)(nx * ny * nz * 8);
  config->bufferSize = psize;

  /*
  cout << "make_config: "<<config<<" "<<endl<< config->buffer<<", "<< *(config->buffer)<<", "
       << config->size[0] << " " << config->size[1] << " " << config->size[2] << " "<< config->FFTdim
       << " " << *(config->bufferSize) << endl;
  */
  return config;
}

/** Initialise the VkFFTApplication from the given configuration.
*
* \param config: the pointer to the VkFFTConfiguration
* \return: the pointer to the newly created VkFFTApplication
*/
VkFFTApplication* init_app(const VkFFTConfiguration* config)
{
  VkFFTApplication* app = new VkFFTApplication({});
  const int res = initializeVkFFT(app, *config);
  /*
  cout << "init_app: "<<config<<endl<< config->buffer<<", "<< *(config->buffer)<<", "
       << config->size[0] << " " << config->size[1] << " " << config->size[2] << " "<< config->FFTdim
       << " " << *(config->bufferSize) << endl<<endl;
  cout<<res<<endl<<endl;
  */
  if(res!=0)
  {
    std::cout << "VkFFTApplication initialisation failed: " << res << std::endl;
    delete app;
    return 0;
  }
  return app;
}

void fft(VkFFTApplication* app)
{
  VkFFTAppend(app, -1, NULL);
}

void ifft(VkFFTApplication* app)
{
  VkFFTAppend(app, 1, NULL);
}

/** Free memory allocated during make_config()
*
*/
void free_app(VkFFTApplication* app)
{
  if(app != NULL)
  {
    deleteVkFFT(app);
    free(app);
  }
}

/** Free memory associated to the vkFFT app
*
*/
void free_config(VkFFTConfiguration *config)
{
  free(config->device);
  // Only frees the pointer to the buffer pointer, not the buffer itself.
  free(config->buffer);
  free(config->bufferSize);
  free(config);
}

/** Basic test function
*
*/
int test_vkfft_cuda(const int size)
{
  printf("vkFFT-cuda is starting...\n");

  // Allocate host arrays
  Complex* arr0 = (Complex*)malloc(sizeof(Complex) * size);
  Complex* arr1 = (Complex*)malloc(sizeof(Complex) * size);
  Complex* arr2 = (Complex*)malloc(sizeof(Complex) * size);

  // Init arrays
  for (unsigned int i = 0; i < size; ++i) {
      arr0[i].x = rand() / (float)RAND_MAX - 0.5f + sin((float)i * 2 * 3.141592653589f / size * 7.7);
      arr0[i].y = 0;
      arr1[i].x = arr0[i].x;
      arr1[i].y = 0;
      arr2[i].x = arr0[i].x;
      arr2[i].y = 0;
  }
  int mem_size = sizeof(Complex) * size;
  // Allocate device memory
  Complex* d1;
  cudaMalloc((void**)&d1, mem_size);
  Complex* d2;
  cudaMalloc((void**)&d2, mem_size);
  // Copy host memory to device
  cudaMemcpy(d1, arr1, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d2, arr2, mem_size, cudaMemcpyHostToDevice);

  //////////////////////////////////////////////// vkFFT ///////////////////////////////////////
  printf("Using vkFFT (CUDA backend)\n");

  // Perform the same FFT using vkFFT
  VkFFTConfiguration configuration = {};

  configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
  configuration.size[0] = size; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.
  configuration.size[1] = 1;
  configuration.size[2] = 1;

  cuInit(0); // Should not be necessary ?
  int ctdev;cuDeviceGetCount(&ctdev);
  cudaSetDevice(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  configuration.device = &dev;

  // CUcontext context;
  // cuCtxCreate(&vkGPU->context, 0, vkGPU->device);

  configuration.buffer = (void**)&d2;
  uint64_t bufferSize = mem_size;
  configuration.bufferSize = &bufferSize;

  // Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.
  VkFFTApplication app = {};
  uint32_t res = initializeVkFFT(&app, configuration);  // TODO
  if(res!=0)
  {
    free(arr0);
    free(arr1);
    free(arr2);
    cudaFree(d1);
    cudaFree(d2);
    std::cout << "Something went wrong intialising the VkFFTApplication !" << std::endl;
    return res;
  }

  VkFFTAppend(&app, -1, NULL);

  cudaDeviceSynchronize(); // necessary ?
  deleteVkFFT(&app);

  //////////////////////////////////////////////// vkFFT End ///////////////////////////////////

  // Copy device memory to host
  cudaMemcpy(arr1, d1, mem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(arr2, d2, mem_size, cudaMemcpyDeviceToHost);

  /*
  std::ofstream out("results.dat");
  for (unsigned int i = 0; i < size; ++i)
  {
      out << arr0[i].x <<"+"<<arr0[i].x<<"j "
          << arr1[i].x <<"+"<<arr1[i].x<<"j "
          << arr2[i].x <<"+"<<arr2[i].x<<"j "<<std::endl;
  }
  */

  // cleanup memory
  free(arr0);
  free(arr1);
  free(arr2);
  cudaFree(d1);
  cudaFree(d2);
  std::cout << "Finished VkFFT basic test"<<std::endl;
  return 0;
}
