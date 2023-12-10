/* PyVkFFT
   (c) 2021- : ESRF-European Synchrotron Radiation Facility
       authors:
         Vincent Favre-Nicolin, favre@esrf.fr
*/

// We use the OpenCL backend
#define VKFFT_BACKEND 3

#include <iostream>
#include <fstream>
#include <memory>
#include <iostream>
using namespace std;
#include "vkFFT.h"

#ifdef _WIN32
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"
#endif

LIBRARY_API VkFFTConfiguration* make_config(const long*,
                                            const size_t, void*, void*, void*, void*, void*,
                                            const int, const size_t, const int,
                                            const int, const int,
                                            const int, const int, const int, const int,
                                            const size_t, const long*,
                                            const int, const int, const int, const int,
                                            const int, const int, const int, const long*,
                                            const int);

LIBRARY_API VkFFTApplication* init_app(const VkFFTConfiguration*, void*, int*,
                                       size_t*, long*, long*);

LIBRARY_API int fft(VkFFTApplication* app, void*, void*, void*);

LIBRARY_API int ifft(VkFFTApplication* app, void*, void*, void*);

LIBRARY_API void free_app(VkFFTApplication* app);

LIBRARY_API void free_config(VkFFTConfiguration *config);

LIBRARY_API uint32_t vkfft_version();

LIBRARY_API uint32_t vkfft_max_fft_dimensions();

/** Create the VkFFTConfiguration from the array parameters
*
* \param size: pointer to the size of the data array along each dimension - this should hold
*    VKFFT_MAX_FFT_DIMENSIONS values, starting from the fastest - e.g. if a numpy array has a shape
*    (nz, ny, nx), then this would hold [nx, ny, nz, 1, 1..]
*    Note that if r2c is True
* \param fftdim: number of dimensions of the transform.
* \param buffer, buffer_out: pointer to the GPU data source and destination arrays. These
*  can be fake and the actual buffers supplied in fft() and ifft. However buffer should be non-zero,
*  and buffer_out should be non-zero only for an out-of-place transform.
* \param platform: the cl_platform
* \param device: the cl_device
* \param ctx: the cl_context
* \param norm: 0, the L2 norm is multiplied by the size on each transform, 1, the inverse transform
*   divides the L2 norm by the size.
* \param precision: number of bits per float, 16=half, 32=single, 64=double precision
* \param r2c: if True, create a configuration for a real<->complex transform
* \return: the pointer to the newly created VkFFTConfiguration, or 0 if an error occurred.
*/
VkFFTConfiguration* make_config(const long* size, const size_t fftdim,
                                void *buffer, void *buffer_out,
                                void* platform, void* device, void* ctx,
                                const int norm, const size_t precision, const int r2c,
                                const int dct, const int dst,
                                const int disableReorderFourStep, const int registerBoost,
                                const int useLUT, const int keepShaderCode, const size_t n_batch,
                                const long *skip,
                                const int coalescedMemory, const int numSharedBanks,
                                const int aimThreads, const int performBandwidthBoost,
                                const int registerBoostNonPow2, const int registerBoost4Step,
                                const int warpSize, const long* grouped_batch,
                                const int forceCallbackVersionRealTransforms)
{
  VkFFTConfiguration *config = new VkFFTConfiguration({});
  config->FFTdim = fftdim;
  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++) config->size[i] = size[i];
  config->numberBatches = n_batch;

  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++) config->omitDimension[i] = skip[i];

  config->normalize = norm;
  config->performR2C = r2c;
  config->performDCT = dct;
  config->performDST = dst;

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

  cl_device_id *pdev = new cl_device_id;
  *pdev = (cl_device_id)device;
  config->device = pdev;

  cl_platform_id *pplatform = new cl_platform_id;
  *pplatform = (cl_platform_id)platform;
  config->platform = pplatform;

  cl_context * pctx = new cl_context;
  *pctx = (cl_context) ctx;
  config->context = pctx;

  void ** pbuf = new void*;
  *pbuf = buffer;

  uint64_t* psize = new uint64_t;
  uint64_t* psizein = psize;

  int s =  size[0];
  for(int i=1; i<VKFFT_MAX_FFT_DIMENSIONS; i++) s *= size[i];

  // forceCallbackVersionRealTransforms is normally automatically set by VkFFT
  // for odd-length R2C/DCT/DST - this can be used to force it for even lengths
  if((r2c || dst || dct) && forceCallbackVersionRealTransforms>0)
    config->forceCallbackVersionRealTransforms = 1;

  if(r2c)
  {
    *psize = (uint64_t)((s / 2 +1) * precision * (size_t)2);
    if(buffer_out != NULL)
    {
      psizein = new uint64_t;
      *psizein = (uint64_t)(s * precision);
      config->inverseReturnToInputBuffer = 1;
			config->inputBufferStride[0] = size[0];
      for(int i=1; i<VKFFT_MAX_FFT_DIMENSIONS; i++)
        config->inputBufferStride[i] = size[i] * config->inputBufferStride[i-1];
    }
  }
  else
  {
    if(dct || dst) *psize = (uint64_t)(s * precision);
    else *psize = (uint64_t)(s * precision * (size_t)2);
  }

  config->bufferSize = psize;

  if(buffer_out != NULL)
  {
    // Calculations are made in buffer, so with buffer != inputBuffer we keep the original data
    void ** pbufout = new void*;
    *pbufout = buffer_out;

    config->buffer = (cl_mem*)pbufout;
    config->inputBuffer = (cl_mem*)pbuf;

    config->inputBufferSize = psizein;

    config->isInputFormatted = 1;
  }
  else
  {
    config->buffer = (cl_mem*)pbuf;
  }

  return config;
}

/** Initialise the VkFFTApplication from the given configuration.
*
* \param config: the pointer to the VkFFTConfiguration
* \param queue: the cl_command_queue
* \return: the pointer to the newly created VkFFTApplication
*/
VkFFTApplication* init_app(const VkFFTConfiguration* config, void *queue, int *res,
                           size_t *tmp_buffer_nbytes, long *use_bluestein_fft,
                           long *num_axis_upload)
{
  VkFFTApplication* app = new VkFFTApplication({});
  *res = initializeVkFFT(app, *config);

  if(*res!=0)
  {
    delete app;
    return 0;
  }
  // Did VkFFT allocate a temporary buffer ?
  if(app->configuration.allocateTempBuffer)
    *tmp_buffer_nbytes = app->configuration.tempBufferSize[0];
   else *tmp_buffer_nbytes = 0;

  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++) use_bluestein_fft[i] = app->useBluesteinFFT[i];
  for(int i=0; i<VKFFT_MAX_FFT_DIMENSIONS; i++) num_axis_upload[i] = app->localFFTPlan->numAxisUploads[i];

  return app;
}

int fft(VkFFTApplication* app, void *in, void *out, void* queue)
{
  cl_command_queue q = (cl_command_queue) queue;

  // Modify the original app only to avoid allocating
  // new buffer pointers in memory
  *(app->configuration.buffer) = (cl_mem)out;
  *(app->configuration.inputBuffer) = (cl_mem)in;
  *(app->configuration.outputBuffer) = (cl_mem)out;
  app->configuration.commandQueue = &q;

  VkFFTLaunchParams par = {};
  par.commandQueue = &q;
  par.buffer =  app->configuration.buffer;
  par.inputBuffer = app->configuration.inputBuffer;
  par.outputBuffer = app->configuration.outputBuffer;

  return VkFFTAppend(app, -1, &par);
}

int ifft(VkFFTApplication* app, void *in, void *out, void* queue)
{
  cl_command_queue q = (cl_command_queue) queue;

  // Modify the original app only to avoid allocating
  // new buffer pointers in memory
  *(app->configuration.buffer) = (cl_mem)out;
  *(app->configuration.inputBuffer) = (cl_mem)in;
  *(app->configuration.outputBuffer) = (cl_mem)out;
  app->configuration.commandQueue = &q;

  VkFFTLaunchParams par = {};
  par.commandQueue = &q;
  par.buffer =  app->configuration.buffer;
  par.inputBuffer = app->configuration.inputBuffer;
  par.outputBuffer = app->configuration.outputBuffer;

  return VkFFTAppend(app, 1, &par);
}

/** Free memory associated to the vkFFT app
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

/** Free memory allocated during make_config()
*
*/
void free_config(VkFFTConfiguration *config)
{
  free(config->platform);
  free(config->device);
  free(config->context);
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
