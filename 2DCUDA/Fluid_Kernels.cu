#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "Fluid_Kernels.cuh"
#include "device_launch_parameters.h"
#include <iostream>

#define index(i,j) ((i) + (DIM) *(j))
#define SWAP(a0, a) {float *tmp = a0; a0 = a; a = tmp;}

//velocity and pressure
float *d_u, *d_v;
float *d_u0, *d_v0;

//divergence of velocity
float *d_div;

//density
float *d_d, *d_d0;

__global__ void addSource_K(int size, float *d, float *s, float dt) {
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int i =  gtidx % size;
	int j = gtidx / size;
	int N = (256 - 2);

	// Skip Boundary values
	if (i<1 || i>N || j<1 || j>N) return;
	// Add source each timestep
	d[gtidx] += dt * s[gtidx];
}

__global__ void advect_K(int size, float *d, float *d0, float *u, float *v, float dt) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int j = (int)gtidx / (size);
	int i = (int)gtidx - (j*size);
	int N = (size - 2);
	
	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	float dx = 1.0f / N;
	dx = 1 / dx;
	dt0 = (dt*dx)/1;
	if (i<1 || i>N || j<1 || j>N) return;

	x = i - dt0*u[index(i, j)];
	y = j - dt0*v[index(i, j)];

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

__global__ void redGauss_K(int size, float *x, float *x0, float a, float c)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int j = (int)gtidx / (size);
	int i = (int)gtidx - (j*size);
	float invC = 1.f / c;
	int N = (size - 2);

	if (i<1 || i>N || j<1 || j>N) return;

	if ((i + j) % 2 == 0)
	{
		x[index(i, j)] = (x0[index(i, j)] +	a * (x[index(i - 1, j)] + x[index(i + 1, j)] + x[index(i, j - 1)] +	x[index(i, j + 1)])) * invC;
	}
}

__global__ void blackGauss_K(int size, float *x, float *x0, float a, float c)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int j = (int)gtidx / (size);
	int i = (int)gtidx - (j*size);
	float invC = 1.f / c;
	int N = (size - 2);

	if (i<1 || i>N || j<1 || j>N) return;

	if ((i + j) % 2 != 0)
	{
		x[index(i, j)] = (x0[index(i, j)] +	a * (x[index(i - 1, j)] + x[index(i + 1, j)] + x[index(i, j - 1)] +	x[index(i, j + 1)])) * invC;
	}
}

__global__ void divergence_K(int size, float* u, float* v, float* p, float* div) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int width = size;
	int i = gtidx % width;
	int j = gtidx / width;

	int N = (size - 2);

	if (i<1 || i>N || j<1 || j>N) return;
	
		float h = 1.0f / N;
		// Calculate divergence using finite difference method
		// We multiply by -1 here to reduce the number of negative multiplications in the pressure calculation
		div[index(i, j)] = -0.5f*h*(u[index(i + 1, j)] - u[index(i - 1, j)] + v[index(i, j + 1)] - v[index(i, j - 1)]);
		p[index(i, j)] = 0;
	
}

__global__ void subtractGradient_K(int size, float *u, float *v, float *p)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int width = size;
	int i = gtidx % width;
	int j = gtidx / width;

	// Skip Boundary values

	int N = (size - 2);

	if (i<1 || i>N || j<1 || j>N) return;
		float h = 1.0f / N;
		// Calculate divergence using finite difference method
		// We multiply by -1 here to reduce the number of negative multiplications in the pressure calculation
		u[index(i, j)] -= 0.5*(p[index(i + 1, j)] - p[index(i - 1, j)]) / h;
		v[index(i, j)] -= 0.5*(p[index(i, j + 1)] - p[index(i, j - 1)]) / h;
	
}

__global__ void set_bnd_K(int size, int b, float *x) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = gtidx + 1;
	int N = size-2;

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
void initCUDA(int dim)
{
	cudaSetDevice(0);
	cudaMalloc((void**)&d_div, dim * sizeof(float));
	cudaMalloc((void**)&d_d, dim * sizeof(float));
	cudaMalloc((void**)&d_d0, dim * sizeof(float));
	cudaMalloc((void**)&d_u, dim * sizeof(float));
	cudaMalloc((void**)&d_u0, dim * sizeof(float));
	cudaMalloc((void**)&d_v, dim * sizeof(float));
	cudaMalloc((void**)&d_v0, dim * sizeof(float));

	// Initialize our "previous" values of density and velocity to be all zero
	cudaMemset(d_u, 0, dim * sizeof(float));
	cudaMemset(d_v, 0, dim * sizeof(float));
	cudaMemset(d_d, 0, dim * sizeof(float));
	cudaMemset(d_u0, 0, dim * sizeof(float));
	cudaMemset(d_v0, 0, dim * sizeof(float));
	cudaMemset(d_d0, 0, dim * sizeof(float));
	cudaMemset(d_div, 0, dim * sizeof(float));
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

	cudaDeviceReset();
}

void diffuse(int size, int b, float *x, float *x0, float diff, int iteration)
{
	int N = (size - 2);
	float a = 0.01f * diff * (float) N * (float) N;
	float c = 1.f + 4.f *a;
	for (int i = 0; i < iteration; i++)
	{
		redGauss_K<<<BLOCKS, THREADS>>>(size, x, x0, a, c);
		cudaDeviceSynchronize();
		blackGauss_K<<<BLOCKS, THREADS>>>(size, x, x0, a, c);
	}

	cudaDeviceSynchronize();
	set_bnd_K<<<BLOCKS, THREADS>>>(size, b, x);
	cudaDeviceSynchronize();
}

void advect(int size, int b, float *d, float *d0, float *u, float *v, float dt)
{
	int N = (size - 2);

	advect_K<<<BLOCKS, THREADS >>>(size, d, d0, u, v, dt);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(size, b, d);
	cudaDeviceSynchronize();
}

void project(int size, float *u, float *v, float *p, float *div, int iteration)
{
	int N = (size - 2);

	divergence_K<<<BLOCKS, THREADS >>>(size, u, v, p, div);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(size, 0, div);
	set_bnd_K<<<1, N >>>(size, 0, p);
	cudaDeviceSynchronize();

	for (int k = 0; k < iteration; k++){
		// Linear Solve
		redGauss_K << <BLOCKS, THREADS >> >(size, p, div, 1, 4);
		cudaDeviceSynchronize();
		blackGauss_K << <BLOCKS, THREADS >> >(size, p, div, 1, 4);
		cudaDeviceSynchronize();
		set_bnd_K << < BLOCKS, THREADS >> >(size, 0, p);
		cudaDeviceSynchronize();
	}

	subtractGradient_K << <BLOCKS, THREADS >> >(size, u, v, p);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(size, 1, u);
	set_bnd_K<<<1, N >>>(size, 2, v);
	cudaDeviceSynchronize();
}

extern "C"
void step(int size, float dt, float viscosity, float diffusion, int iteration, float *sd, float *su, float *sv)
{
	// Vel step
	// Add Velocity Source
	cudaMemcpy(d_u0, su, (size*size)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v0, sv, (size*size)*sizeof(float), cudaMemcpyHostToDevice);
	addSource_K <<<BLOCKS, THREADS>>>(size, d_u, d_u0, dt);
	addSource_K <<<BLOCKS, THREADS>>>(size, d_v, d_v0, dt);
	cudaDeviceSynchronize();

	SWAP(d_u0, d_u);
	diffuse(size, 1, d_u, d_u0, viscosity, iteration);
	SWAP(d_v0, d_v);
	diffuse(size, 2, d_v, d_v0, viscosity, iteration);

	project(size, d_u, d_v, d_u0, d_v0, iteration);

	SWAP(d_u0, d_u);
	SWAP(d_v0, d_v);
	advect(size, 1, d_u, d_u0, d_u0, d_v0, dt);
	advect(size, 1, d_v, d_v0, d_u0, d_v0, dt);

	project(size, d_u, d_v, d_u0, d_v0, iteration);
	
	// Density step
	// Add Density Source
	cudaMemcpy(d_d0, sd, (size*size)*sizeof(float), cudaMemcpyHostToDevice);
	addSource_K <<<BLOCKS, THREADS>>>(size, d_d, d_d0, dt);
	cudaDeviceSynchronize();

	SWAP(d_d0, d_d);
	diffuse(size, 0, d_d, d_d0, diffusion, iteration);
	SWAP(d_d0, d_d);
	advect(size, 0, d_d, d_d0, d_u, d_v, dt);

	cudaMemcpy(sd, d_d, (size*size)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(su, d_u, (size*size)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(sv, d_v, (size*size)*sizeof(float), cudaMemcpyDeviceToHost);
	return;
}