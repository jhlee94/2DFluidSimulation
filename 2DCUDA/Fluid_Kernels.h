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

__global__ void add_source_k(float *d, float *s);
__global__ void advect_k(float *dold, float *d, float *u, float *v, float md);
__global__ void divergence_k(float *u, float *v, float *div);
__global__ void pressure_k(float *u, float *v, float *p, float *pold, float *div);
__global__ void set_bnd_k(float *u, float *v, float *p);
__global__ void velocity_bc_k(float *u, float *v);
__global__ void pressure_bc_k(float *p);

#endif