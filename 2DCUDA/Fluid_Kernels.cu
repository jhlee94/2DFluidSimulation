#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "Fluid_Kernels.cuh"
#include "device_launch_parameters.h"

#define index(i,j) ((i) * (ddim.width) *(j))
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

//velocity and pressure
float *d_u, *d_v;
float *d_u0, *d_v0;

//divergence of velocity
float *d_div;

//density
float *d_d, *d_d0;


__global__ void add_source_K(float *d, float *s) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width - 2;
	int height = ddim.height - 2;
	int i = gtidx % ddim.width;
	int j = gtidx / ddim.width;

	// Skip Boundary values
	if (i > 0 && j > 0 && i < width-2 && j < height-2)
	{
		// Add source each timestep
		d[i] += ddim.timestep * s[i];
	}
}

__global__ void advect_K(float *d, float *d0, float *u, float *v) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int size = ddim.width * ddim.height;
	int j = (int)gtidx / (ddim.width);
	int i = (int)gtidx - (j*ddim.width);
	int N = (ddim.width - 2);
	
	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	dt0 = (ddim.timestep*N);
	if (i<1 || i>N || j<1 || j>N) return;

	if (x<0.5) x = 0.5;
	if (x>N + 0.5) x = N + 0.5;

	i0 = (int)x;
	i1 = i0 + 1;

	if (y<0.5) y = 0.5;
	if (y>N + 0.5) y = N + 0.5;

	j0 = (int)y;
	j1 = j0 + 1;

	s1 = x - i0;
	s0 = 1 - s1;
	t1 = y - j0;
	t0 = 1 - t1;
	d[index(i, j)] = s0 * (t0*d0[index(i0, j0)] + t1*d0[index(i0, j1)]) +
		s1 * (t0*d0[index(i1, j0)] + t1*d0[index(i1, j1)]);
}

__global__ void redGauss_K(float *x, float *x0, float a, float c)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int size = ddim.width * ddim.height;
	int j = (int)gtidx / (ddim.width);
	int i = (int)gtidx - (j*ddim.width);
	float invC = 1.f / c;
	int N = (ddim.width - 2);

	if (i<1 || i>N || j<1 || j>N) return;

	if ((i + j) % 2 == 0)
	{
		x[index(i, j)] =
			(x0[index(i, j)] +
			a * (x[index(i - 1, j)] +
			x[index(i + 1, j)] +
			x[index(i, j - 1)] +
			x[index(i, j + 1)])) * invC;
	}
}

__global__ void blackGauss_K(float *x, float *x0, float a, float c)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int size = ddim.width * ddim.height;
	int j = (int)gtidx / (ddim.width);
	int i = (int)gtidx - (j*ddim.width);
	float invC = 1.f / c;
	int N = (ddim.width - 2);

	if (i<1 || i>N || j<1 || j>N) return;

	if ((i + j) % 2 != 0)
	{
		x[index(i, j)] =
			(x0[index(i, j)] +
			a * (x[index(i - 1, j)] +
			x[index(i + 1, j)] +
			x[index(i, j - 1)] +
			x[index(i, j + 1)])) * invC;
	}
}

__global__ void divergence_K(float* u, float* v, float* p, float* div) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int i = gtidx % width;
	int j = gtidx / width;

	// Skip Boundary values
	if (i > 0 && j > 0 && i < width - 2 && j < height - 2)
	{
		float h = 1.0f / (width-2);
		// Calculate divergence using finite difference method
		// We multiply by -1 here to reduce the number of negative multiplications in the pressure calculation
		div[index(i, j)] = -0.5f*h*(u[index(i + 1, j)] - u[index(i - 1, j)] + v[index(i, j + 1)] - v[index(i, j - 1)]);
		p[index(i, j)] = 0;
	}
}

__global__ void subtractGradient_K(float *u, float *v, float *p)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int width = ddim.width;
	int height = ddim.height;
	int i = gtidx % width;
	int j = gtidx / width;

	// Skip Boundary values
	if (i > 0 && j > 0 && i < width - 2 && j < height - 2)
	{
		float h = 1.0f / (width - 2);
		// Calculate divergence using finite difference method
		// We multiply by -1 here to reduce the number of negative multiplications in the pressure calculation
		u[index(i, j)] -= 0.5*(p[index(i + 1, j)] - p[index(i - 1, j)]) / h;
		v[index(i, j)] -= 0.5*(p[index(i, j + 1)] - p[index(i, j - 1)]) / h;
	}
}

__global__ void set_bnd_K(int b, float *x) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = gtidx + 1;
	int N = ddim.width-2;

	if (i <= N){
		x[index(0, i)] = b == 1 ? -x[index(1, i)] : x[index(1, i)];
		x[index(N + 1, i)] = b == 1 ? -x[index(N, i)] : x[index(N, i)];
		x[index(i, 0)] = b == 2 ? -x[index(i, 1)] : x[index(i, 1)];
		x[index(i, N + 1)] = b == 2 ? -x[index(i, N)] : x[index(i, N)];

		if (i == 1)
		{
			x[index(0, 0)] = 0.5f*(x[index(1, 0)] + x[index(0, 1)]);
			x[index(0, N + 1)] = 0.5f*(x[index(1, N + 1)] + x[index(0, N)]);
			x[index(N + 1, 0)] = 0.5f*(x[index(N, 0)] + x[index(N + 1, 1)]);
			x[index(N + 1, N + 1)] = 0.5f*(x[index(N, N + 1)] + x[index(N + 1, N)]);
		}
	}
}


extern "C"
void initCUDA(int size)
{
	cudaMalloc((void**)&d_div, size * sizeof(float));
	cudaMalloc((void**)&d_d, size * sizeof(float));
	cudaMalloc((void**)&d_d0, size * sizeof(float));
	cudaMalloc((void**)&d_u, size * sizeof(float));
	cudaMalloc((void**)&d_u0, size * sizeof(float));
	cudaMalloc((void**)&d_v, size * sizeof(float));
	cudaMalloc((void**)&d_v0, size * sizeof(float));

	// Initialize our "previous" values of density and velocity to be all zero
	cudaMemset(d_u, 0, size * sizeof(float));
	cudaMemset(d_v, 0, size * sizeof(float));
	cudaMemset(d_d, 0, size * sizeof(float));
	cudaMemset(d_u0, 0, size * sizeof(float));
	cudaMemset(d_v0, 0, size * sizeof(float));
	cudaMemset(d_d0, 0, size * sizeof(float));
	cudaMemset(d_div, 0, size * sizeof(float));
}

extern "C"
void freeCUDA()
{
	cudaFree(d_d);
	cudaFree(d_d0);
	cudaFree(d_u);
	cudaFree(d_u0);
	cudaFree(d_v);
	cudaFree(d_v0);
	cudaFree(d_div);
}
extern "C"
void diffuse(int b, float *x, float *x0, float diff, int iteration)
{
	int N = (ddim.width - 2);
	float a = ddim.timestep * diff * (float) N * (float) N;

	for (int i = 0; i < iteration; i++)
	{
		redGauss_K<<<BLOCKS, THREADS>>>(x, x0, x, a, (1 + 4 * a));
		cudaDeviceSynchronize();
		blackGauss_K<<<BLOCKS, THREADS>>>(x, x0, x, a, (1 + 4 * a));
	}

	cudaDeviceSynchronize();
	set_bnd_K<<<1, N>>>(b, x);
	cudaDeviceSynchronize();
}
extern "C"
void advect(int b, float *d, float *d0, float *u, float *v)
{
	int N = (ddim.width - 2);

	advect_K<<<BLOCKS, THREADS >>>(d, d0, u, v);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(b, d);
	cudaDeviceSynchronize();
}
extern "C"
void project(float *u, float *v, float *p, float *div)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int size = ddim.width * ddim.height;
	int j = (int)gtidx / (ddim.width);
	int i = (int)gtidx - (j*ddim.width);
	int N = (ddim.width - 2);

	divergence_K<<<BLOCKS, THREADS >>>(u, v, p, div);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(0, div);
	set_bnd_K<<<1, N >>>(0, p);
	cudaDeviceSynchronize();

	// Linear Solve
	redGauss_K<<<BLOCKS, THREADS >>>(p, div, p, 1, 4);
	cudaDeviceSynchronize();
	blackGauss_K<<<BLOCKS, THREADS >>>(p, div, p, 1, 4);
	cudaDeviceSynchronize();
	set_bnd_K<<< 1, N >>>(0, p);

	subtractGradient_K <<<BLOCKS, THREADS >>>(u, v, p);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(1, u);
	set_bnd_K<<<1, N >>>(2, v);
	cudaDeviceSynchronize();
}

extern "C"
void step(int size, float dt, float viscosity, float diffusion, int iteration)
{
	if (dt != ddim.timestep)
		ddim.timestep = dt;
	// Vel step
	// Add Velocity Source

	SWAP(d_u0, d_u);
	diffuse(1, d_u, d_u0, viscosity, iteration);
	SWAP(d_v0, d_v);
	diffuse(2, d_v, d_v0, viscosity, iteration);

	project(d_u, d_v, d_u0, d_v0);

	SWAP(d_u0, d_u);
	SWAP(d_v0, d_v);
	advect(1, d_u, d_u0, d_u0, d_v0);
	advect(1, d_v, d_v0, d_u0, d_v0);

	project(d_u, d_v, d_u0, d_v0);

	// Reset for next step
	cudaMemset(d_u0, 0, size * sizeof(float));
	cudaMemset(d_v0, 0, size * sizeof(float));

	// Density step
	// Add Density Source
	SWAP(d_d0, d_d);
	diffuse(0, d_d, d_d0, diffusion, iteration);
	SWAP(d_d0, d_d);
	advect(0, d_d, d_d0, d_u, d_v);
	
	// Reset for next Step
	cudaMemset(d_d0, 0, size * sizeof(float));

	return;
}