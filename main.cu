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
#define BASE 1
#define PI 3.14159265359

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
void fillColor(int n, int H, int W, double* color, double reStart, double reEnd, double imStart, double imEnd, int maxIter, volatile int *progress, bool greyscale) {

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

    // Do one iteration with the base number in the exponent
    powerRe = logRe * BASE;
    powerIm = logIm * BASE;

    nextRe = exp(powerRe) * cos(powerIm);
    nextIm = exp(powerRe) * sin(powerIm);

    // if (powerRe > 700) {
    //   toggleOverflow = 1;
    // }

    while (numberOfIterations < maxIter && toggleOverflow == 0)
    {
        powerRe = (nextRe * logRe - nextIm * logIm);
        powerIm = (nextRe * logIm + nextIm * logRe);

        if (powerRe > 700) {
            toggleOverflow = 1;
        }

        nextRe = exp(powerRe) * cos(powerIm);
        nextIm = exp(powerRe) * sin(powerIm);
        
        numberOfIterations += 1;
    }
  }

  double smoothening_factor = greyscale ? 0 : 1 - slog(powerRe);
  double it = numberOfIterations == maxIter ? maxIter : numberOfIterations + smoothening_factor;

  color[T] = it;
  if (!(threadIdx.x || threadIdx.y)){
    atomicAdd((int *)progress, 1);
    __threadfence_system();
  }
}

typedef struct RgbColor
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
} RgbColor;

typedef struct HsvColor
{
  unsigned char h;
  unsigned char s;
  unsigned char v;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv)
{
  RgbColor rgb;
  unsigned char region, remainder, p, q, t;

  if (hsv.s == 0)
  {
    rgb.r = hsv.v;
    rgb.g = hsv.v;
    rgb.b = hsv.v;
    return rgb;
  }

  region = hsv.h / 43;
  remainder = (hsv.h - (region * 43)) * 6; 

  p = (hsv.v * (255 - hsv.s)) >> 8;
  q = (hsv.v * (255 - ((hsv.s * remainder) >> 8))) >> 8;
  t = (hsv.v * (255 - ((hsv.s * (255 - remainder)) >> 8))) >> 8;

  switch (region)
  {
    case 0:
      rgb.r = hsv.v; rgb.g = t; rgb.b = p;
      break;
    case 1:
      rgb.r = q; rgb.g = hsv.v; rgb.b = p;
      break;
    case 2:
      rgb.r = p; rgb.g = hsv.v; rgb.b = t;
      break;
    case 3:
      rgb.r = p; rgb.g = q; rgb.b = hsv.v;
      break;
    case 4:
      rgb.r = t; rgb.g = p; rgb.b = hsv.v;
      break;
    default:
      rgb.r = hsv.v; rgb.g = p; rgb.b = q;
      break;
  }

  return rgb;
}

HsvColor RgbToHsv(RgbColor rgb)
{
  HsvColor hsv;
  unsigned char rgbMin, rgbMax;

  rgbMin = rgb.r < rgb.g ? (rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
  rgbMax = rgb.r > rgb.g ? (rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

  hsv.v = rgbMax;
  if (hsv.v == 0)
  {
    hsv.h = 0;
    hsv.s = 0;
    return hsv;
  }

  hsv.s = 255 * long(rgbMax - rgbMin) / hsv.v;
  if (hsv.s == 0)
  {
    hsv.h = 0;
    return hsv;
  }

  if (rgbMax == rgb.r)
    hsv.h = 0 + 43 * (rgb.g - rgb.b) / (rgbMax - rgbMin);
  else if (rgbMax == rgb.g)
    hsv.h = 85 + 43 * (rgb.b - rgb.r) / (rgbMax - rgbMin);
  else
    hsv.h = 171 + 43 * (rgb.r - rgb.g) / (rgbMax - rgbMin);

  return hsv;
}

double linear_interpolation(double color1, double color2, double t) {
  return color1 * (1 - t) + color2 * t;
}

char*   filenameF =   "preview.png";

void set_palette(gdImagePtr image, int *palette, int *black, bool greyscale) {
  HsvColor    col_hsv;
  RgbColor    col_rgb;
  
  if (!greyscale) {
    *black = gdImageColorAllocate(image, 0, 0, 0);

    for (int i=0; i<255; i++){
      col_hsv.h = i;
      col_hsv.s = 255;
      col_hsv.v = (i == 255 ? 0 : 255);
      col_rgb = HsvToRgb(col_hsv);
      palette[i] = gdImageColorAllocate(image, col_rgb.r, col_rgb.g, col_rgb.b);
    }
  
    palette[255] = gdImageColorAllocate(image, 0, 0, 0);
  }
  else {
    *black = gdImageColorAllocate(image, 255, 255, 255);

    for (int i=0; i<255; i++){
      palette[i] = gdImageColorAllocate(image, 0, 0, 0);
    }
  
    palette[255] = gdImageColorAllocate(image, 255, 255, 255);
  }
}

extern "C" {
double *create_frame(int sharpness, double centerRe, double centerIm, double epsilon, int maxIter, bool greyscale, double *res) {
  FILE*       outfile;                      // defined in stdio
  int         T, i, x, y;                   // array subscripts
  gdImagePtr  image;                        // a GD image object
  char        filename[80];
  int         black, palette[256];          // red, all possible shades of palette

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
  double *color = (double*) malloc(N*sizeof(double));
  double *d_color;

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  cudaMalloc(&d_color, N*sizeof(double));

  image = gdImageCreate(pngWidth, pngHeight);


  set_palette(image, palette, &black, greyscale);

  //Progress variables
  volatile int *d_data, *h_data;
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaHostAlloc((void **)&h_data, sizeof(int), cudaHostAllocMapped);
  cudaHostGetDevicePointer((int **)&d_data, (int *)h_data, 0);
  *h_data = 0;

  // Calculate power tower convergence / divergence
  int num_blocks = (pngWidth*pngHeight+255)/256;
  fillColor<<<num_blocks, 256>>>(N, pngHeight, pngWidth, d_color, reStart, reEnd, imStart, imEnd, maxIter, d_data, greyscale);
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

  cudaMemcpy(color, d_color, N*sizeof(double), cudaMemcpyDeviceToHost);

  // Create lookup table of hues
  int *hues = (int*) calloc((maxIter + 1), sizeof(int));
  for (int T = 0; T < N; T++) {
    if (color[T] < maxIter) {
      int index = (int) color[T];
      hues[(int) color[T]]++;
    }
  }

  int total = 0;
  for (int i = 0; i < maxIter; i++) {
    total += hues[i];
  }

  hues[0] *= 255;
  for (int i = 1; i < maxIter; i++) {
    hues[i] = hues[i - 1] + 255 * hues[i];
  }
  hues[maxIter] = -1;

  if (total > 0) {
    for (int i = 0; i < maxIter; i++) {
      hues[i] /= total;
    }
  }

  // Now create the result array consisting of the actual colors
  for (int T = 0; T < N; T++) {
    res[T] = linear_interpolation(hues[(int) color[T]], hues[(int) (color[T] + .5)], color[T] - (int) color[T]);
    x = T % pngHeight;
    y = T / pngHeight;
    gdImageSetPixel(image, x, y, res[T] > 0 ? res[T] : black);  }

  // Free 2D array
  cudaFree(d_color);
  free(hues);
  free(color);
  // printf("Creating output file '%s'.\n", filenameF);
  outfile = fopen(filenameF, "wb");
  gdImagePng(image, outfile);
  fclose(outfile);

  return res;
}
}