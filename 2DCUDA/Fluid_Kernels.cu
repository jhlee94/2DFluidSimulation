#pragma once
#include "Fluid_Kernels.cuh"

#define index(i,j) ((i) + (DIM) *(j))
#define SWAP(a0, a) {float *tmp = a0; a0 = a; a = tmp;}
#define CLAMP(v, a, b) (a + (v - a) / (b - a))

extern uchar4 *d_textureBufferData;

//velocity and pressure
float *d_u, *d_v;
float *d_u0, *d_v0;
float *d_curl;
//divergence of velocity
float *d_div;

//density
float *d_d, *d_d0;

__global__ void addSource_K(int size, float *d, float *s, float dt) {
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = gtidx % size;
	int j = gtidx / size;
	int N = (size - 2);
	
	// Skip Boundary values
	if (i<1 || i>N || j<1 || j>N) return;
	// Add source each timestep
	d[gtidx] += dt * s[gtidx];
}

__global__ void texture_K(int size, uchar4 *surface, float *dens)
{
	int gtidx = (int) (threadIdx.x + blockIdx.x * blockDim.x);
	int i = gtidx % size;
	int j = gtidx / size;
	int N = (size - 2);

	const float treshold1 = 1.;
	const float treshold2 = 4.;
	const float treshold3 = 10.;

	// Skip Boundary values
	if (i<1 || i>N || j<1 || j>N) {
		surface[index(i, j)].w = 255;
		surface[index(i, j)].x = 0;
		surface[index(i, j)].y = 0;
		surface[index(i, j)].z = 0;
		return;
	}
	else
	{
		float pvalue = dens[index(i, j)];
		uchar4 color;
		/* red */
		if (pvalue < treshold1) {
			color.w = 255;
			color.x = 255 * CLAMP(pvalue, 0., treshold1);
			color.y = 0;
			color.z = 0;
		}
		/* yellow */
		else if (pvalue < treshold2) {
			color.w = 255;
			color.x = 255;
			color.y = 255 * (CLAMP(pvalue, treshold1, treshold2) - treshold1);
			color.z = 0;
		}
		/* white */
		else if (pvalue < treshold3){
			color.w = 255;
			color.x = 255;
			color.y = 255;
			color.z = 255 * (CLAMP(pvalue, treshold2, treshold3) - treshold2);
		}
		else{
			color.w = 255;
			color.x = 255;
			color.y = 255;
			color.z = 255;
		}

		if (pvalue > 0) {
			// populate it
			surface[index(i, j)].w = color.w;
			surface[index(i, j)].x = color.x;
			surface[index(i, j)].y = color.y;
			surface[index(i, j)].z = color.z;
		}
		else {
			surface[index(i, j)].w = 255;
			surface[index(i, j)].x = 0;
			surface[index(i, j)].y = 0;
			surface[index(i, j)].z = 0;
		}
	}
}


__global__ void addConstantSource_K(int size, float* x, int i, int j, float value, float dt)
{
	int N = (size - 2);

	// Skip Boundary values
	if (i<1 || i>N || j<1 || j>N) return;

	x[index(i, j)] += value *dt;

	x[index(i+1, j)] += value *dt;
	x[index(i-1, j)] += value *dt;
	x[index(i + 2, j)] += value *dt;
	x[index(i - 2, j)] += value *dt;
}

__global__ void advect_K(int size, float *d, float *d0, float *u, float *v, float dt) {
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = gtidx % size;
	int j = gtidx / size;
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
	int i = gtidx % size;
	int j = gtidx / size;
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
	int i = gtidx % size;
	int j = gtidx / size;
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
	int i = gtidx % size;
	int j = gtidx / size;

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
	int i = gtidx % size;
	int j = gtidx / size;

	// Skip Boundary values

	int N = (size - 2);

	if (i<1 || i>N || j<1 || j>N) return;
		float h = 1.0f / N;
		// Calculate divergence using finite difference method
		// We multiply by -1 here to reduce the number of negative multiplications in the pressure calculation
		u[index(i, j)] -= 0.5*(p[index(i + 1, j)] - p[index(i - 1, j)]) / h;
		v[index(i, j)] -= 0.5*(p[index(i, j + 1)] - p[index(i, j - 1)]) / h;
	
}

__global__ void curl_K(int size, float *u, float *v, float *curl)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = gtidx % size;
	int j = gtidx / size;

	// Skip Boundary values
	int N = (size - 2);
	if (i<1 || i>N || j<1 || j>N) return;

	float h = 1.0f / N;
	float du_dy;
	float dv_dx;

	du_dy = (u[index(i, j + 1)] - u[index(i, j - 1)]) /h * 0.5f;
	dv_dx = (v[index(i + 1, j)] - v[index(i - 1, j)]) /h * 0.5f;

	curl[index(i, j)] = (dv_dx - du_dy);
}

__global__ void vorticity_K(int size, float *u, float *v, float *curl, float vort_str, float dt)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = gtidx % size;
	int j = gtidx / size;

	// Skip Boundary values
	int N = (size - 2);
	if (i<1 || i>N || j<1 || j>N) return;

	float h = 1.0f / N;

	float vort;

	float omegaT = curl[index(i, j - 1)];
	float omegaB = curl[index(i, j + 1)];
	float omegaR = curl[index(i + 1, j)];
	float omegaL = curl[index(i - 1, j)];

	float dw_dx = (omegaR - omegaL) * 0.5f;
	float dw_dy = (omegaT - omegaB) * 0.5f;
	float2 force; force.x = dw_dy; force.y = dw_dx; force /= h;
	force /= (length(force) + 0.000001f);
	
	float2 newVec;
	newVec.x = -curl[index(i, j)] * force.y;
	newVec.y = curl[index(i, j)] * force.x;
	newVec *= vort_str;
	u[index(i, j)] += newVec.x * dt;
	v[index(i, j)] += newVec.y * dt;

}

__global__ void buoyancy_K(int size, float *d, float *s, float kappa, float sigma)
{
	int gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = (int)gtidx % size;
	int j = (int)gtidx / size;
	int N = (size - 2);

	// Skip Boundary values
	if (i<1 || i>N || j<1 || j>N) return;

	d[index(i, j)] = sigma*s[index(i, j)] + -kappa* s[index(i, j)];
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
	cudaMalloc((void**)&d_curl, dim * sizeof(float));

	// Initialize our "previous" values of density and velocity to be all zero
	cudaMemset(d_u, 0, dim * sizeof(float));
	cudaMemset(d_v, 0, dim * sizeof(float));
	cudaMemset(d_d, 0, dim * sizeof(float));
	cudaMemset(d_u0, 0, dim * sizeof(float));
	cudaMemset(d_v0, 0, dim * sizeof(float));
	cudaMemset(d_d0, 0, dim * sizeof(float));
	cudaMemset(d_div, 0, dim * sizeof(float));
	cudaMemset(d_curl, 0, dim * sizeof(float));
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
	cudaFree(d_curl);
	cudaFree(d_textureBufferData);
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
	set_bnd_K<<<1, N>>>(size, b, x);
	cudaDeviceSynchronize();
}

void advect(int size, int b, float *d, float *d0, float *u, float *v, float dt)
{
	int N = (size - 2);

	advect_K<<<BLOCKS, THREADS>>>(size, d, d0, u, v, dt);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(size, b, d);
	cudaDeviceSynchronize();
}

void project(int size, float *u, float *v, float *p, float *div, int iteration)
{
	int N = (size - 2);

	divergence_K<<<BLOCKS, THREADS>>>(size, u, v, p, div);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(size, 0, div);
	set_bnd_K<<<1, N >>>(size, 0, p);
	cudaDeviceSynchronize();

	for (int k = 0; k < iteration; k++){
		// Linear Solve
		redGauss_K<<<BLOCKS, THREADS>>>(size, p, div, 1, 4);
		cudaDeviceSynchronize();
		blackGauss_K<<<BLOCKS, THREADS>>>(size, p, div, 1, 4);
		cudaDeviceSynchronize();
		set_bnd_K<<<1, N>>>(size, 0, p);
		cudaDeviceSynchronize();
	}

	subtractGradient_K<<<BLOCKS, THREADS>>>(size, u, v, p);
	cudaDeviceSynchronize();
	set_bnd_K<<<1, N >>>(size, 1, u);
	set_bnd_K<<<1, N >>>(size, 2, v);
	cudaDeviceSynchronize();
}

extern "C"
void step(int size,
		  float dt, 
		  float viscosity, 
		  float diffusion, 
		  float kappa, 
		  float sigma, 
		  int iteration,
		  float *sd,
		  float s_v_i,
		  float s_v_j,
		  float s_d_i,
		  float s_d_j,
		  float s_d_val,
		  float s_u_val,
		  float s_v_val)
{
	int N = (size - 2);
	// Vel step
	// Add Velocity Source
	addConstantSource_K<<<1, 1>>>(size, d_u, s_v_i, s_v_j, s_u_val, dt);
	addConstantSource_K<<<1, 1>>>(size, d_v, s_v_i, s_v_j, s_v_val, dt);
	cudaDeviceSynchronize();

	//Vorticity
	if (true) {
		curl_K<<<BLOCKS, THREADS>>>(size, d_u, d_v, d_curl);
		cudaDeviceSynchronize();
		vorticity_K<<<BLOCKS, THREADS>>>(size, d_u, d_v, d_curl, 0.10f, dt);
		cudaDeviceSynchronize();
		set_bnd_K<<<1, N >>>(size, 1, d_u);
		set_bnd_K<<<1, N >>>(size, 2, d_v);
		cudaDeviceSynchronize();
	}

	// Buoyancy
	if (true) {
		buoyancy_K<<<BLOCKS, THREADS>>>(size, d_v0, d_d, kappa, sigma);
		cudaDeviceSynchronize();
		addSource_K<<<BLOCKS, THREADS>>>(size, d_v, d_v0, dt);
		cudaDeviceSynchronize();
		set_bnd_K<<<1, N>>>(size, 2, d_v);
		cudaDeviceSynchronize();
	}
	
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
	addConstantSource_K<<<1, 1>>>(size, d_d, s_d_i, s_d_j, s_d_val, dt);
	addConstantSource_K<<<1, 1>>>(size, d_d, 128, 248, 100, dt);
	cudaDeviceSynchronize();

	SWAP(d_d0, d_d);
	diffuse(size, 0, d_d, d_d0, diffusion, iteration);
	SWAP(d_d0, d_d);
	advect(size, 0, d_d, d_d0, d_u, d_v, dt);

	//cudaMemcpy(sd, d_d, (size*size)*sizeof(float), cudaMemcpyDeviceToHost);

	// Reset for next step
	cudaMemset(d_u0, 0, (size*size) * sizeof(float));
	cudaMemset(d_v0, 0, (size*size) * sizeof(float));
	cudaMemset(d_d0, 0, (size*size) * sizeof(float));
	return;
}

extern "C"
void createTexture(int size, uchar4* d_texture)
{
	texture_K<<<BLOCKS,THREADS>>>(size, d_texture, d_d);
}