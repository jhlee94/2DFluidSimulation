#pragma once
#include <math.h>
#include "Config.h"

class Fluid2DCPU
{
public:
	int bits, dim, dim2;
	unsigned int m_scale;
	float m_diffusion, m_viscosity;
	float	*u, *u_prev;		// velocity x
	float	*v, *v_prev;	// velocity y 
	float	*dens, *dens_prev; // Density
	float	*m_curl; // Curl Data
	int iteration = 10; // Poisson Iteration *Gauss-Seidel method
	bool vorticity = true, buoyancy = true;
	Fluid2DCPU(int N,
		float viscosity = 0.0001f, 
		float diffusion = 0.0002f,
		unsigned int scale = 2)
		: m_viscosity(viscosity), m_diffusion(diffusion), m_scale(scale)
	{
		dim = N;
		dim2 = (dim + 2)*(dim + 2);

		u = new float[dim2];
		v = new float[dim2];
		dens = new float[dim2];

		u_prev = new float[dim2];
		v_prev = new float[dim2];
		dens_prev = new float[dim2];
		m_curl = new float[dim2];

		//Initialise all to zero
		for (unsigned int i = 0; i < dim2; i++)
		{
			u[i] = 0.f;
			v[i] = 0.f;
			dens[i] = 0.f; 
			u_prev[i] = 0.f;
			v_prev[i] = 0.f;
			dens_prev[i] = 0.f;
		}

	}

	~Fluid2DCPU() {
		delete[] u;
		delete[] u_prev;
		delete[] v;
		delete[] v_prev;
		delete[] dens;
		delete[] dens_prev;
		delete[] m_curl;
	}

	int index(int i, int j);
	void set_bnd(int b, float *x);
	void addSource(float *x, float *s, float dt);
	void linearSolve(int b, float* x, float* x0, float a, float c);
	void diffuse(int b, float *x, float *x0, float diff, float dt);
	void advect(int b, float *d, float *d0, float *u, float *v, float dt);
	float curl(int i, int j);
	void vort_conf(float *u, float *v);
	void project(float *u, float *v, float *p, float *div);
	void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt);
	void vel_step(float* u, float *v, float *u0, float *v0, float viscosity, float dt);
	void step(float dt);
};
