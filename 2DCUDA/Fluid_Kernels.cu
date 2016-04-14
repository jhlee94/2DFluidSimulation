#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Fluid_Kernels.cuh"

//velocity and pressure
float *u, *v, *p;
float *uold, *vold, *pold;
float *utemp, *vtemp, *ptemp; // temp pointers, DO NOT INITIALIZE

//divergence of velocity
float *divg;

//density
float *d, *dold, *dtemp;
float *map;

//sources
float *sd, *su, *sv;

__global__ void add_source_k(float *d, float *s) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	// Skip Boundary values
	if (px > 0 && py > 0 && px < width - 1 && py < height - 1)
	{
		// Add source each timestep
		d[i] += ddim.timestep * s[i];
	}
}

__global__ void advect_k(float *dold, float *d, float *u, float *v, float md) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	int px0, py0, px1, py1;
	float x, y, dx0, dx1, dy0, dy1;

	// Skip Boundary values
	if (px != 0 && py != 0 && px != width - 1 && py != height - 1)
	{
		// Move "backwards" in time
		x = px - ddim.timestep * (width - 2)  * u[i]; // Multiply by the width of the grid not including the boundary
		y = py - ddim.timestep * (height - 2) * v[i]; // Multiply by the height of the grid not including the boundary

		// Clamp to Edges, that is, if the velocity goes over the edge, clamp it to the boundary value
		if (x < 0.5) x = 0.5; if (x > width - 1.5)  x = width - 1.5;
		if (y < 0.5) y = 0.5; if (y > height - 1.5) y = height - 1.5;

		// Setup bilinear interpolation "corner points"
		px0 = (int)x; px1 = px0 + 1; dx1 = x - px0; dx0 = 1 - dx1;
		py0 = (int)y; py1 = py0 + 1; dy1 = y - py0; dy0 = 1 - dy1;

		// Perform a bilinear interpolation
		d[i] = dx0 * (dy0 * dold[px0 + width*py0] + dy1 * dold[px0 + width*py1]) +
			dx1 * (dy0 * dold[px1 + width*py0] + dy1 * dold[px1 + width*py1]);

		// Multiply by the mass dissipation constant
		d[i] *= md;
	}

	return;
}

__global__ void divergence_k(float *u, float *v, float *div) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	// Skip Boundary values
	if (px > 0 && py > 0 && px < width - 1 && py < height - 1)
	{
		float u_l = u[i - 1];
		float u_r = u[i + 1];
		float v_t = v[px + (py - 1)*width];
		float v_b = v[px + (py + 1)*width];

		// Calculate divergence using finite difference method
		// We multiply by -1 here to reduce the number of negative multiplications in the pressure calculation
		div[i] = -0.5 * ((u_r - u_l) / (width - 2) + (v_b - v_t) / (height - 2));
	}
}

__global__ void set_bnd_k(float *u, float *v, float *p) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	// Skip Boundary values
	if (px > 0 && py > 0 && px < width - 1 && py < height - 1) {

		// left, right, top, bottom neighbors
		float p_l = p[i - 1];
		float p_r = p[i + 1];
		float p_t = p[px + (py - 1)*width];
		float p_b = p[px + (py + 1)*width];

		// Find the gradient and perform the correction/projection
		u[i] -= 0.5*(p_r - p_l)*(width - 2);
		v[i] -= 0.5*(p_b - p_t)*(height - 2);
	}
}

extern "C" 
void add_source(float *d, float *s)
{
	dim3 blocks;
	dim3 tids(TIDSX, TIDSY);

	add_source_k <<<blocks, tids >>>(d, s);
}

extern "C" 
void advect(float *dold, float *d, float *u, float *v, float md)
{

}

extern "C" 
void divergence(float *u, float *v, float *div)
{

}

extern "C" 
void pressure(float *u, float *v, float *p, float *pold, float *div)
{

}

extern "C" 
void set_bnd(float *u, float *v, float *p)
{

}

extern "C" 
void velocity_bc(float *u, float *v)
{

}

extern "C" 
void pressure_bc(float *p)
{

}

int main()
{

	return 0;
}