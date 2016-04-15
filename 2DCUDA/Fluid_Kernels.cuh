#ifndef _FLUID2DGPU_CUH_
#define _FLUID2DGPU_CUH_

#include "config.h"

typedef struct {
    int width;
    int height;
    int steps;
    float timestep;    // Timestep in Seconds
} Dimensions;

__constant__ Dimensions ddim;

__global__ void addSource_K(float *d, float *s);
__global__ void advect_K(float *d, float *d0, float *u, float *v);
__global__ void redGauss_K(float *x, float *x0, float a, float c);
__global__ void blackGauss_K(float *x, float *x0, float a, float c);
__global__ void divergence_K(float *u, float *v, float *p, float *div);
__global__ void subtractGradient_K(float *u, float *v, float *p);
__global__ void set_bnd_K(int b, float *x);

#endif