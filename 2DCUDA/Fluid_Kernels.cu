#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "Fluid_Kernels.cuh"
#include "device_launch_parameters.h"


__global__ void add_source_K(float *d, float *s) {
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

__global__ void advect_K(float *dold, float *d, float *u, float *v, float md) {
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

__global__ void divergence_K(float *u, float *v, float *div) {
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

__global__ void pressure_K(float *u, float *v, float *p, float *pold, float *div) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	// Skip Boundary values
	if (px > 0 && py > 0 && px < width - 1 && py < height - 1)
	{
		// left, right, top, bottom neighbors
		float x_l = p[i - 1];
		float x_r = p[i + 1];
		float x_t = p[px + (py - 1)*width];
		float x_b = p[px + (py + 1)*width];
		float b = div[i];

		// Jacobi method for solving the pressure Poisson equation
		pold[i] = (x_l + x_r + x_t + x_b + b) * 0.25; // Here b is positive because of the extra negative sign in the divergence calculation
	}
}

__global__ void set_bnd_K(float *u, float *v, float *p) {
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

__global__ void velocity_bc_K(float *u, float *v) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	// Skip Inner Values
	if (px == 0)
	{
		u[i] = -u[i + 1];
		v[i] = v[i + 1];
	}
	else if (py == 0)
	{
		u[i] = u[px + (py + 1)*width];
		v[i] = -v[px + (py + 1)*width];
	}
	else if (px == width - 1)
	{
		u[i] = -u[i - 1];
		v[i] = v[i - 1];
	}
	else if (py == height - 1)
	{
		u[i] = u[px + (py - 1)*width];
		v[i] = -v[px + (py - 1)*width];
	}
}

__global__ void pressure_bc_K(float *p) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int px = i % width;
	int py = i / width;

	// Skip Inner Values
	if (px == 0)
	{
		p[i] = p[i + 1];
	}
	else if (py == 0)
	{
		p[i] = p[px + (py + 1)*width];
	}
	else if (px == width - 1)
	{
		p[i] = p[i - 1];
	}
	else if (py == height - 1)
	{
		p[i] = p[px + (py - 1)*width];
	}
}

extern "C" 
void add_source(float *d, float *s, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	add_source_K <<<blocks, THREADS_PER_BLOCK >>>(d, s);
}

extern "C" 
void advect(float *dold, float *d, float *u, float *v, float md, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	advect_K <<<blocks, THREADS_PER_BLOCK >>> (dold, d, u, v, md);
}

extern "C" 
void divergence(float *u, float *v, float *div, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	divergence_K<<<blocks,THREADS_PER_BLOCK>>>(u, v, div);
}

extern "C" 
void pressure(float *u, float *v, float *p, float *pold, float *div, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	pressure_K <<<blocks, THREADS_PER_BLOCK >>>(u, v, p, pold, div);
}

extern "C" 
void set_bnd(float *u, float *v, float *p, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	set_bnd_K<<<blocks,THREADS_PER_BLOCK>>>(u, v, p);
}

extern "C" 
void velocity_bc(float *u, float *v, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	velocity_bc_K<<<blocks,THREADS_PER_BLOCK>>>(u, v);
}

extern "C"
void pressure_bc(float *p, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	pressure_bc_K <<<blocks, THREADS_PER_BLOCK >>>(p);
}