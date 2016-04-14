#ifndef _FLUID2DGPU_CUH_
#define _FLUID2DGPU_CUH_

#include "config.h"

typedef struct {
    int width;
    int height;
    int steps;
    float timestep;    // Timestep in Seconds
    float dissipation; // Mass Dissipation Constant (how our "ink" dissipates over time). Set to 1 for no dissipation, 0.995 tends to look nice though.
} Dimensions;

__constant__ Dimensions ddim;

__global__ void AddSource_k(float *d, float *s, int size);
__global__ void Advect_k(float *dold, float *d, float *u, float *v, float md,int size);
__global__ void Divergence_k(float *u, float *v, float *div, int size);
__global__ void SetBnd_k(float *u, float *v, float *p, int size);
__global__ void RedGaussSeidel_K(float* x, float *x0, float diff, int size);
__global__ void BlackGaussSeidel_K(float *x, float *x0, float diff, int size);
__global__ void Project_K(float *u, float *v);

#endif