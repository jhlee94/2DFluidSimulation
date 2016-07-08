#ifndef _FLUID2DCPU_H_
#define _FLUID2DCPU_H_

// Includes
#include <SFML\Graphics.hpp>
#include <math.h>
#include "Config.h"

class Fluid2DCPU
{
public:
	struct Parameters
	{
		int		iterations;// Poisson Iteration *Gauss-Seidel method
		float	dt;
		float	kappa;
		float	sigma;
		float	vort_str;
		float	diffusion;
		float	viscosity;
		bool	vorticity;
		bool	buoyancy;
		bool	velocity;
		bool	grid;
	} m_parameters;

	int dim;
	int	dim2;
	float	*u, *u_prev;		// velocity x Field
	float	*v, *v_prev;	// velocity y  Field
	float	*dens, *dens_prev; // Density Field
	float	*m_curl; // Curl Data
	float	*aux;
	sf::Uint8* pixels;

public:
	Fluid2DCPU() {};

	~Fluid2DCPU() {
		delete[] u;
		delete[] u_prev;
		delete[] v;
		delete[] v_prev;
		delete[] dens;
		delete[] dens_prev;
		delete[] m_curl;
		delete[] pixels;
	}

	// Class Functions
	void Initialise(unsigned int N);
	void UpdateTexture();
	void ApplyColour(sf::Uint8 *pixels, float x, int i);

	// Solver Functions
	int index(int i, int j);
	void set_bnd(int b, float *x);
	void addSource(float *d, float *s, float dt);
	void jacobi(int b, float* aux, float* x, float* x0, float a, float c);
	void gaussSeidel(int b, float* x, float* x0, float a, float c);
	void diffuse(int b, float *x, float *x0, float diff, float dt);
	void advect(int b, float *d, float *d0, float *u, float *v, float dt);
	float curl(int i, int j);
	void vort_conf(float *u, float *v, float vort_str, float dt);
	void buoyancy(float *d, float *s, float kappa, float sigma);
	void project(float *u, float *v, float *p, float *div);
	void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt);
	void vel_step(float* u, float *v, float *u0, float *v0, float viscosity, float kappa, float sigma, float dt);
	void step();
};
#endif // !_FLUID2DCPU_H_