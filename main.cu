#include <gd.h>
#include <stdio.h>
#include <stdlib.h>

#include <fenv.h>
#include <math.h>
#include <errno.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

// ------------------------
// TO BE CUSTOMIZED BY USER
// ------------------------

// ------------------------

// See the bottom of this code for a discussion of some output possibilities.
char*   filenameF =   "output.txt";

__device__
double slog(double in) {
  double result = 0;

  while (in > 1) {
    result += 1;
    in = log(in);
  }

  return result - 1 + in;
}

__global__
void fillColor(int n, int H, int W, double* color, double reStart, double reEnd, double imStart, double imEnd, int maxIter, volatile int *progress) {

  int T = blockIdx.x*blockDim.x + threadIdx.x;
  if (T >= n) return;

  int x = T % H;
  int y = T / H;
  double re = reStart + ((double) x / W * (reEnd - reStart));
  double im = imEnd - ((double) y / H * (imEnd - imStart));

  double nextRe, nextIm, logRe, logIm, powerRe, powerIm;

  int toggleOverflow = 0;                                          
  int numberOfIterations = 0;                                      
  if (re == 0 && im == 0){
    color[T] = maxIter;
  }
  else {
    logRe = .5*log(re*re + im*im);
    logIm = atan2(im, re);
    nextRe = re;
    nextIm = im;
    while (numberOfIterations < maxIter && toggleOverflow == 0)
    {
        powerRe = (nextRe * logRe - nextIm * logIm);
        powerIm = (nextRe * logIm + nextIm * logRe);

        if (powerRe > 10) {
            toggleOverflow = 1;
        }

        nextRe = exp(powerRe) * cos(powerIm);
        nextIm = exp(powerRe) * sin(powerIm);
        
        numberOfIterations += 1;
    }
  }

  double it = numberOfIterations == maxIter ? maxIter : numberOfIterations + 1 - slog(powerRe);
  color[T] = it;
  if (!(threadIdx.x || threadIdx.y)){
    atomicAdd((int *)progress, 1);
    __threadfence_system();
  }
}

extern "C" {
double *create_frame(int sharpness, double centerRe, double centerIm, double epsilon, int maxIter, double *res) {
  FILE*       outfile;                      // defined in stdio
  int         T;                            // array subscripts
  

  double reStart = centerRe - epsilon;
  double reEnd = centerRe + epsilon;
  double imStart = centerIm - epsilon;
  double imEnd = centerIm + epsilon;

  printf("start = %f + %fi\n", reStart, imStart);
  printf("end = %f + %fi\n", reEnd, imEnd);

  int pngWidth = sharpness;
  int pngHeight = pngWidth * (imEnd - imStart) / (reEnd - reStart);
  int N = pngWidth * pngHeight;

  res = (double*) malloc(N*sizeof(double));
  double* d_color;

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  cudaMalloc(&d_color, N*sizeof(double));

  //Progress variables
  volatile int *d_data, *h_data;
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaHostAlloc((void **)&h_data, sizeof(int), cudaHostAllocMapped);
  cudaHostGetDevicePointer((int **)&d_data, (int *)h_data, 0);
  *h_data = 0;

  // Calculate power tower convergence / divergence
  int num_blocks = (pngWidth*pngHeight+255)/256;
  fillColor<<<num_blocks, 256>>>(N, pngHeight, pngWidth, d_color, reStart, reEnd, imStart, imEnd, maxIter, d_data);
  cudaEventRecord(stop);

  // Measure progress
  float my_progress = 0.0f;
  int value = 0;
  printf("Progress:\n");
  do{
    cudaEventQuery(stop);  // may help WDDM scenario
    int value1 = *h_data;
    float kern_progress = (float)value1 / (float)num_blocks;
    if ((kern_progress - my_progress) > 0.1f) {
      printf("percent complete = %2.1f\n", (kern_progress*100));
      my_progress = kern_progress;}}
  while (my_progress < 0.9f);

  // Wait for all threads to finish
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();

  cudaMemcpy(res, d_color, N*sizeof(double), cudaMemcpyDeviceToHost);

  // Free 2D array
  cudaFree(d_color);

  printf("Calculation done.\n");

  return res;
}
}