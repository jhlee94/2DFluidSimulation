#ifndef _FLUID2DGPU_CUH_
#define _FLUID2DGPU_CUH_

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>
#include "config.h"


typedef struct {
	int width;
	int steps;
	float timestep;    // Timestep in Seconds
} Dimensions;

__constant__ Dimensions ddim;

__global__ void addSource_K(int size, float *d, float *s);
__global__ void advect_K(int size, float *d, float *d0, float *u, float *v);
__global__ void redGauss_K(int size, float *x, float *x0, float a, float c);
__global__ void blackGauss_K(int size, float *x, float *x0, float a, float c);
__global__ void divergence_K(int size, float *u, float *v, float *p, float *div);
__global__ void subtractGradient_K(int size, float *u, float *v, float *p);
__global__ void set_bnd_K(int size, int b, float *x);


void initCUDA(int size);
void freeCUDA();
void step(int size, float dt, float viscosity, float diffusion, int iteration, float *sd, float *su, float *sv);

void diffuse(int size, int b, float *x, float *x0, float diff, int iteration, float dt);
void advect(int size, int b, float *d, float *d0, float *u, float *v, float dt);
void project(int size, float *u, float *v, float *p, float *div, int iteration);
#endif