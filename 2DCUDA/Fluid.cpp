// OpenGL 
#include <GL/glew.h>
// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "config.h"
#include "Fluid_Kernels.h"

// Global Variable Init
//velocity and pressure
float *u, *v, *p;
float *uold, *vold, *pold;
float *utemp, *vtemp, *ptemp; // temp pointers, DO NOT INITIALIZE

//divergence of velocity
float *div;

//density
float *d, *dold, *dtemp;
float *map;

//sources
float *sd, *su, *sv;


// Cuda Kernels
extern "C" void add_source(float *d, float *s);
extern "C" void advect(float *dold, float *d, float *u, float *v, float md);
extern "C" void divergence(float *u, float *v, float *div);
extern "C" void pressure(float *u, float *v, float *p, float *pold, float *div);
extern "C" void set_bnd(float *u, float *v, float *p);
extern "C" void velocity_bc(float *u, float *v);
extern "C" void pressure_bc(float *p);

int main(void)
{


	return 0;
}