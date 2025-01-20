/* PyVkFFT
   (c) 2021- : ESRF-European Synchrotron Radiation Facility
       authors:
         Vincent Favre-Nicolin, favre@esrf.fr
*/

// We use the Vulkan backend
#define VKFFT_BACKEND 0

#include <iostream>
#include <fstream>
#include <memory>
using namespace std;
#include "vkFFT.h"
//typedef float2 Complex;

#ifdef _WIN32
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"

extern "C"{

double __exp_glibc_2_2_5(double x);
double __exp2_glibc_2_2_5(double x);
double __log_glibc_2_2_5(double x);
double __log2_glibc_2_2_5(double x);
double __pow_glibc_2_2_5(double x, double y);

asm(".symver __exp_glibc_2_2_5, exp@GLIBC_2.2.5");
asm(".symver __exp2_glibc_2_2_5, exp2@GLIBC_2.2.5");
asm(".symver __log_glibc_2_2_5, log@GLIBC_2.2.5");
asm(".symver __log2_glibc_2_2_5, log2@GLIBC_2.2.5");
asm(".symver __pow_glibc_2_2_5, pow@GLIBC_2.2.5");

double __wrap_exp(double x)
{
    return __exp_glibc_2_2_5(x); 
}

double __wrap_exp2(double x)
{
    return __exp2_glibc_2_2_5(x); 
}

double __wrap_log(double x)
{
    return __log_glibc_2_2_5(x); 
}

double __wrap_log2(double x)
{
    return __log2_glibc_2_2_5(x); 
}

double __wrap_pow(double x, double y)
{
    return __pow_glibc_2_2_5(x, y); 
}

}
#endif

LIBRARY_API VkFFTConfiguration* make_config(const long*, const size_t, VkBuffer, VkBuffer, 
                                VkPhysicalDevice*, VkDevice*, VkQueue*,
                                VkCommandPool*, VkFence*, uint64_t,
                                const int, const size_t, const int, const int, const int, const int,
                                const int, const int, const size_t, const long*,
                                const int, const int, const int, const int, const int, const int, const int,
                                const long*);

LIBRARY_API VkFFTApplication* init_app(const VkFFTConfiguration*, int*);

LIBRARY_API int fft(VkFFTApplication* app, VkCommandBuffer* cmd_buffer, VkBuffer* in, VkBuffer* out);

LIBRARY_API int ifft(VkFFTApplication* app, VkCommandBuffer* cmd_buffer, VkBuffer* in, VkBuffer* out);

LIBRARY_API int copy_test(char *buf);

LIBRARY_API int get_dev_props(VkPhysicalDevice* physicalDevice, char *buf);

LIBRARY_API int get_dev_props2(const VkFFTConfiguration* config, char *buf);

LIBRARY_API int get_buf_size(VkBuffer buffer, VkDevice* dev);

LIBRARY_API int sync_app(VkFFTApplication* app);

LIBRARY_API void free_app(VkFFTApplication* app);

LIBRARY_API void free_config(VkFFTConfiguration *config);

LIBRARY_API uint32_t vkfft_version();

LIBRARY_API uint32_t vkfft_max_fft_dimensions();

// LIBRARY_API int cuda_runtime_version();

// LIBRARY_API int cuda_driver_version();

// LIBRARY_API int cuda_compile_version();


class PyVkFFT
{
  public:
    PyVkFFT(const int nx, const int ny, const int nz, const int fftdim, void* hstream,
            const int norm, const int precision, const int r2c)
    {

    };
  private:
    VkFFTConfiguration mConf;
    VkFFTApplication mApp;
    VkFFTApplication mLaunchParams;
};


/** Create the VkFFTConfiguration from the array parameters
*
* \param nx, ny, nz: dimensions of the array. The fast axis is x. In the corresponding numpy array,
* this corresponds to a shape of (nz, ny, nx)
* \param fftdim: the dimension of the transform. If nz>1 and fftdim=2, the transform is only made
* on the x and y axes
* \param buffer, buffer_out: pointer to the GPU data source and destination arrays. These
*  can be fake and the actual buffers supplied in fft() and ifft. However buffer should be non-zero,
*  and buffer_out should be non-zero only for an out-of-place transform.
* \param hstream: the stream handle (CUstream)
* \param norm: 0, the L2 norm is multiplied by the size on each transform, 1, the inverse transform
*   divides the L2 norm by the size.
* \param precision: number of bits per float, 16=half, 32=single, 64=double precision
* \return: the pointer to the newly created VkFFTConfiguration, or 0 if an error occurred.
*/

int copy_test(char *buf){
    char buf2[256] = "Hello!!";
    memmove((void*) buf, (void*) buf2, 256);
    return 123;
}    

int get_dev_props(VkPhysicalDevice* physicalDevice, char *buf){
	VkPhysicalDeviceProperties physicalDeviceProperties = { 0 };
	vkGetPhysicalDeviceProperties(physicalDevice[0], &physicalDeviceProperties);
    memmove((void*) buf, (void*) physicalDeviceProperties.deviceName, 256);
    return 0;
};


int get_dev_props2(const VkFFTConfiguration* config, char *buf){
    // same as above, but uses config
	VkPhysicalDeviceProperties physicalDeviceProperties = { 0 };
    
    VkFFTApplication* app = new VkFFTApplication({});
    VkFFTConfiguration inputLaunchConfiguration = *config;
    app->configuration.physicalDevice = inputLaunchConfiguration.physicalDevice;
    
    
	vkGetPhysicalDeviceProperties(app->configuration.physicalDevice[0], &physicalDeviceProperties);
    memmove((void*) buf, (void*) physicalDeviceProperties.deviceName, 256);
    return 0;
};

int sync_app(VkFFTApplication* app){
  VkFFTResult resFFT;
  resFFT = VkFFTSync(app);
  return resFFT;
    
};

int get_buf_size(VkBuffer buffer, VkDevice* dev){
    
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(*dev, buffer, &mem_req);  
    return mem_req.size;
    
};

ofstream myfile;

VkFFTConfiguration* make_config(const long* size, const size_t fftdim,
                                VkBuffer buffer, VkBuffer buffer_out, 
                                VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* queue,
                                VkCommandPool* commandPool, VkFence* fence, uint64_t isCompilerInitialized,
                                const int norm, const size_t precision, const int r2c, const int dct,
                                const int disableReorderFourStep, const int registerBoost,
                                const int useLUT, const int keepShaderCode, const size_t n_batch,
                                const long* skip,
                                const int coalescedMemory, const int numSharedBanks,
                                const int aimThreads, const int performBandwidthBoost,
                                const int registerBoostNonPow2, const int registerBoost4Step,
                                const int warpSize, const long* grouped_batch)
{
  VkFFTConfiguration *config = new VkFFTConfiguration({});
  

  myfile.open ("outputtt.txt");
  myfile << "Debug file.\n";
  
  
  
  config->FFTdim = fftdim;
  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++) config->size[i] = size[i];
  config->numberBatches = n_batch;

  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++) config->omitDimension[i] = skip[i];
  
  config->normalize = norm;
  config->performR2C = r2c;
  config->performDCT = dct;

  if(disableReorderFourStep>=0)
    config->disableReorderFourStep = disableReorderFourStep;

  if(registerBoost>=0)
    config->registerBoost = registerBoost;

  if(useLUT>=0)
    config->useLUT = useLUT;

  if(keepShaderCode>=0)
    config->keepShaderCode = keepShaderCode;

  if(coalescedMemory>=0)
    config->coalescedMemory = coalescedMemory;

  if(numSharedBanks>=0)
    config->numSharedBanks = numSharedBanks;

  if(aimThreads>=0)
    config->aimThreads = aimThreads;

  if(performBandwidthBoost>=0)
    config->performBandwidthBoost = performBandwidthBoost;

  if(registerBoostNonPow2>=0)
    config->registerBoostNonPow2 = registerBoostNonPow2;

  if(registerBoost4Step>=0)
    config->registerBoost4Step = registerBoost4Step;

  if(warpSize>=0)
    config->warpSize = warpSize;

  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++)
     if(grouped_batch[i]>0) config->groupedBatch[i] = grouped_batch[i];

  switch(precision)
  {
      case 2 : config->halfPrecision = 1;
      case 8 : config->doublePrecision = 1;
  };

  config->physicalDevice = physicalDevice;
  config->device = device;
  config->queue = queue;
  config->commandPool = commandPool;
  config->fence = fence;
  //config->isCompilerInitialized = isCompilerInitialized;
  
  VkBuffer * pbuf = new VkBuffer;
  *pbuf = buffer;

  uint64_t* psize = new uint64_t;
  uint64_t* psizein = psize;

  int s =  size[0];
  for(int i=1; i<VKFFT_MAX_FFT_DIMENSIONS; i++) s *= size[i];

  if(r2c)
  {
    *psize = (uint64_t)((s / 2 +1) * precision * (size_t)2);
    if(buffer_out != NULL)
    {
      psizein = new uint64_t;
      *psizein = (uint64_t)(s * precision);
      //config->isInputFormatted = 0;
      config->inverseReturnToInputBuffer = 1;
	  config->inputBufferStride[0] = size[0];
      for(int i=1; i<VKFFT_MAX_FFT_DIMENSIONS; i++)
        config->inputBufferStride[i] = size[i] * config->inputBufferStride[i-1];
    }
  }
  else
  {
    if(dct) *psize = (uint64_t)(s * precision);
    else *psize = (uint64_t)(s * precision * (size_t)2);
  }

  config->bufferSize = psize;
  
  //DvdB: Now we remove the padding size from the fast FT dimension:
  //if(r2c) config->size[0] -= 2;
  
// Calculations are made in buffer, so with buffer != inputBuffer we keep the original data
  if(buffer_out != NULL)
  {

    VkBuffer * pbufout = new VkBuffer;
    *pbufout = buffer_out;

    config->buffer = pbufout;
    config->inputBuffer = pbuf;

    config->inputBufferSize = psizein;

    config->isInputFormatted = 1;
  }
  else
  {
    config->buffer = pbuf;
  }
  cout << "Hello everyone! Please get yourself comfortable while the Config is being made!\n";

  myfile << "make_config: "<<config<<" "<<endl
       << config->buffer<<", "<< *(config->buffer)<< endl
       << size[0] << " " << size[1] << " " << size[2] << " " << size[3]<< ", FFTdim: " << config->FFTdim << endl
       << *(config->bufferSize) << endl 
       << *(config->inputBufferSize) << endl;
  
  
  myfile << "\n End of debug file.\n";
  myfile.close();

  return config;
}

/** Initialise the VkFFTApplication from the given configuration.
*
* \param config: the pointer to the VkFFTConfiguration
* \return: the pointer to the newly created VkFFTApplication
*/
VkFFTApplication* init_app(const VkFFTConfiguration* config, int *res)
{
    
  //cout << "Hello everyone! Please get yourself comfortable while the Config is being made!\n";

  VkFFTApplication* app = new VkFFTApplication({});
  *res = initializeVkFFT(app, *config);
  /*
  cout << "init_app: "<<config<<endl<< config->buffer<<", "<< *(config->buffer)<<", "
       << config->size[0] << " " << config->size[1] << " " << config->size[2] << " "<< config->FFTdim
       << " " << *(config->bufferSize) << endl<<endl;
  cout<<res<<endl<<endl;
  */
  if(*res!=0)
  {
    delete app;
    return 0;
  }
  return app;
}

int fft(VkFFTApplication* app, VkCommandBuffer* cmd_buffer, VkBuffer* in, VkBuffer* out)
{

  (app->configuration.buffer) = out;
  (app->configuration.inputBuffer) = in;
  //(app->configuration.outputBuffer) = out;

  VkFFTLaunchParams par = {};
  par.buffer =  app->configuration.buffer;
  par.inputBuffer = app->configuration.inputBuffer;
  //par.outputBuffer = app->configuration.outputBuffer;
  par.commandBuffer = cmd_buffer;
 
  return VkFFTAppend(app, -1, &par);
}

int ifft(VkFFTApplication* app, VkCommandBuffer* cmd_buffer,  VkBuffer* in, VkBuffer* out)
{

  (app->configuration.buffer) = out;
  (app->configuration.inputBuffer) = in;
  //(app->configuration.outputBuffer) = out;

  VkFFTLaunchParams par = {};
  par.buffer =  app->configuration.buffer;
  par.inputBuffer = app->configuration.inputBuffer;
  //par.outputBuffer = app->configuration.outputBuffer;
  par.commandBuffer = cmd_buffer;

  return VkFFTAppend(app, 1, &par);
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
  // Only frees the pointer to the buffer pointer, not the buffer itself.
  free(config->buffer);
  free(config->bufferSize);

  if((config->outputBuffer != NULL) && (config->buffer != config->outputBuffer)) free(config->outputBuffer);
  if((config->inputBuffer != NULL) && (config->buffer != config->inputBuffer)
     && (config->outputBuffer != config->inputBuffer)) free(config->inputBuffer);

  if((config->inputBufferSize != NULL) && (config->inputBufferSize != config->bufferSize))
    free(config->inputBufferSize);
  if((config->outputBufferSize != NULL) && (config->outputBufferSize != config->bufferSize)
     && (config->outputBufferSize != config->inputBufferSize)) free(config->outputBufferSize);

  //****
  //if(config->stream != 0) free(config->stream);
  
  // if (config->physicalDevice) free(config->physicalDevice);
  // if (config->device) free(config->device);
  // if (config->queue) free(config->queue);
  // if (config->commandPool) free(config->commandPool);
  // if (config->fence) free(config->fence);
  free(config);
}

/// Get VkFFT version
uint32_t vkfft_version()
{
  return VkFFTGetVersion();
};

/// Get VKFFT_MAX_FFT_DIMENSIONS
uint32_t vkfft_max_fft_dimensions()
{
  return VKFFT_MAX_FFT_DIMENSIONS;
};

// /// CUDA runtime version
// int cuda_runtime_version()
// {
  // int v=0;
  // const cudaError_t err = cudaRuntimeGetVersion(&v);
  // if(err==cudaSuccess) return v;
  // return 0;
// };

// /// CUDA driver version
// int cuda_driver_version()
// {
  // int v=0;
  // const CUresult err = cuDriverGetVersion(&v);
  // if(err==CUDA_SUCCESS) return v;
  // return 0;
// };

// /// CUDA version against which pyvkfft was compiled
// int cuda_compile_version()
// {
  // return (int)CUDA_VERSION;
// };
