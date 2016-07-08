#ifndef _FLUID2DGPU_CUH_
#define _FLUID2DGPU_CUH_

#include <stdlib.h>
#include <stdio.h>

#include <windows.h>
#include <mmsystem.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <cuda.h>
#include <device_functions.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_math.h>

#include "config.h"

__global__ void addSource_K(int size, float *d, float *s);
__global__ void addConstantSource_K(int size, float* x, int i, int j, float value, float dt);
__global__ void texture_K(int size, uchar4 *surface, float *dens);
__global__ void advect_K(int size, float *d, float *d0, float *u, float *v, bool isBackward);
__global__ void MacCormack_K(int N, float* dest, float* d0, float* MC1, float* MC2, float* u, float* v, float dt);
__global__ void redGauss_K(int size, float *x, float *x0, float a, float c);
__global__ void blackGauss_K(int size, float *x, float *x0, float a, float c);
__global__ void divergence_K(int size, float *u, float *v, float *p, float *div);
__global__ void subtractGradient_K(int size, float *u, float *v, float *p);
__global__ void curl_K(int size, float *u, float *v, float *curl);
__global__ void vorticity_K(int size, float *u, float *v, float *curl, float vort_str, float dt);
__global__ void buoyancy_K(int size, float *d, float *s, float kappa, float sigma);
__global__ void set_bnd_K(int size, int b, float *x);

extern uchar4 *d_textureBufferData;

//velocity and pressure
float *d_u, *d_v;
float *d_u0, *d_v0;
float *d_curl;
//divergence of velocity
float *d_div;

//density
float *d_d, *d_d0;
//MacCormack
float *d_MC_b, *d_MC_f;

cudaStream_t streams[2];

#endif