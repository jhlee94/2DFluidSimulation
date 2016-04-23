#include "Fluid.h"

#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

void Fluid2DCPU::Initialise(unsigned int N)
{
	dim = N;
	dim2 = (N + 2) * (N + 2);
	u = new float[dim2];
	v = new float[dim2];
	dens = new float[dim2];

	u_prev = new float[dim2];
	v_prev = new float[dim2];
	dens_prev = new float[dim2];
	m_curl = new float[dim2];

	m_parameters.iterations = 10;
	m_parameters.dt = 0.01f;
	m_parameters.kappa = 0.3f;
	m_parameters.sigma = 0.f;
	m_parameters.diffusion = 0.f;
	m_parameters.viscosity = 0.f;
	m_parameters.vorticity = true;
	m_parameters.buoyancy = true;

	//Initialise all to zero
	for (unsigned int i = 0; i < dim2; i++)
	{
		u[i] = 0.f;
		v[i] = 0.f;
		dens[i] = 0.f;
		u_prev[i] = 0.f;
		v_prev[i] = 0.f;
		dens_prev[i] = 0.f;
		m_curl[i] = 0.f;
	}

}

int Fluid2DCPU::index(int i, int j)
{
	return i + (dim + 2)*j;
}

void Fluid2DCPU::set_bnd(int b, float *x)
{
	int i;
	for (i = 1; i <= dim; i++) {
		x[index(0, i)] = (b == 1) ? -x[index(1, i)] : x[index(1, i)];
		x[index(dim + 1, i)] = (b == 1) ? -x[index(dim, i)] : x[index(dim, i)];
		x[index(i, 0)] = (b == 2) ? -x[index(i, 1)] : x[index(i, 1)];
		x[index(i, dim + 1)] = (b == 2) ? -x[index(i, dim)] : x[index(i, dim)];
	}
	x[index(0, 0)] = 0.5*(x[index(1, 0)] + x[index(0, 1)]);
	x[index(0, dim + 1)] = 0.5*(x[index(1, dim + 1)] + x[index(0, dim)]);
	x[index(dim + 1, 0)] = 0.5*(x[index(dim, 0)] + x[index(dim + 1, 1)]);
	x[index(dim + 1, dim + 1)] = 0.5*(x[index(dim, dim + 1)] + x[index(dim + 1, dim)]);
}

void Fluid2DCPU::addSource(float *x, float *s, float dt)
{
	int i;
	for (i = 0; i < dim2; i++){
		x[i] += dt*s[i];
	}
}

void Fluid2DCPU::linearSolve(int b, float* x, float* x0, float a, float c)
{
	int i, j, k;
	float invC = 1.f / c;
	for (k = 0; k < m_parameters.iterations; k++) {
		for (i = 1; i <= dim; i++) {
			for (j = 1; j <= dim; j++) {
				x[index(i, j)] =
					(x0[index(i, j)] +
					a * (x[index(i - 1, j)] +
					x[index(i + 1, j)] +
					x[index(i, j - 1)] +
					x[index(i, j + 1)])) * invC;
			}
		}
		set_bnd(b, x);
	}
}

float Fluid2DCPU::curl(int i, int j)
{
	float h = 1.0f / dim;
	float du_dy = (u[index(i, j + 1)] - u[index(i, j - 1)]) /h * 0.5f;
	float dv_dx = (v[index(i + 1, j)] - v[index(i - 1, j)]) /h * 0.5f;

	return dv_dx - du_dy;
}

void Fluid2DCPU::vort_conf(float *u, float *v, float vort_str, float dt)
{
	float dw_dx, dw_dy;
	float length;
	float vortx, vorty;
	float h = 1.0f / dim;

	for (int i = 0; i < dim2; i++)
	{
		int x = i % (dim + 2);
		int y = i / (dim + 2);
		if (x<1 || x>dim || y<1 || y>dim) {}
		else{
			m_curl[i] = curl(x, y);
		}
	}/*
	for (int i = 1; i <= dim; i++)
	{
		for (int j = 1; j <= dim; j++)
		{
			m_curl[i] = curl(i, j);
		}
	}*/

	for (int i = 1; i < dim; i++)
	{
		for (int j = 1; j < dim; j++)
		{

			// Find derivative of the magnitude (n = del |w|)
			float omegaT = m_curl[index(i, j - 1)];
			float omegaB = m_curl[index(i, j + 1)];
			float omegaR = m_curl[index(i + 1, j)];
			float omegaL = m_curl[index(i - 1, j)];

			float dw_dx = ((omegaR - omegaL) * 0.5f) /h;
			float dw_dy = ((omegaT - omegaB) * 0.5f) /h;


			// Calculate vector length. (|n|)
			// Add small factor to prevent divide by zeros.
			length = (float) sqrt(dw_dx * dw_dx + dw_dy * dw_dy)
				+ 0.000001f;

			// N = ( n/|n| )
			dw_dx /= length;
			dw_dy /= length;

			vortx = -curl(i, j) * dw_dx * vort_str;
			vorty = curl(i, j) * dw_dy * vort_str;

			// N x w
			u[index(i, j)] += vortx * dt;
			v[index(i, j)] += vorty * dt;
		}
	}
	set_bnd(1, u);
	set_bnd(2, v);
}

void Fluid2DCPU::buoyancy(float *d, float *s, float kappa, float sigma)
{
	unsigned int i, j;

	for (i = 1; i <= dim; i++) {
		for (j = 1; j <= dim; j++) {
			d[index(i, j)] = sigma*s[index(i, j)] + -kappa*s[index(i, j)];
		}
	}
}

void Fluid2DCPU::diffuse(int b, float *x, float *x0, float diff, float dt)
{
	float a = dt*diff*dim*dim;
	linearSolve(b, x, x0, a, 1.f + 4.f * a);
}

//Semi-Lagrangian Advection Method
void Fluid2DCPU::advect(int b, float *d, float *d0, float *u, float *v, float dt)
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	dt0 = dt*dim;
	for (i = 1; i <= dim; i++) {
		for (j = 1; j <= dim; j++) {
			x = i - dt0*u[index(i, j)];
			y = j - dt0*v[index(i, j)];
			if (x<0.5) x = 0.5;
			if (x>dim + 0.5) x = dim + 0.5;

			i0 = (int)x;
			i1 = i0 + 1;

			if (y<0.5) y = 0.5;
			if (y>dim + 0.5) y = dim + 0.5;

			j0 = (int)y;
			j1 = j0 + 1;

			s1 = x - i0;
			s0 = 1 - s1;
			t1 = y - j0;
			t0 = 1 - t1;
			d[index(i, j)] = s0 * (t0*d0[index(i0, j0)] + t1*d0[index(i0, j1)]) +
							 s1 * (t0*d0[index(i1, j0)] + t1*d0[index(i1, j1)]);
		}
	}
	set_bnd(b, d);
}

void Fluid2DCPU::project(float *u, float *v, float *p, float *div)
{
	int i, j, k;

	// Divergence
	float h;
	h = 1.0f / dim;
	for (i = 1; i <= dim; i++) {
		for (j = 1; j <= dim; j++) {
			div[index(i, j)] = -0.5*h*(u[index(i + 1, j)] - u[index(i - 1, j)] + v[index(i, j + 1)] - v[index(i, j - 1)]);

			p[index(i, j)] = 0;
		}
	}

	set_bnd(0, div);
	set_bnd(0, p);

	linearSolve(0, p, div, 1, 4);

	// Subtract Gradient
	for (i = 1; i <= dim; i++) {
		for (j = 1; j <= dim; j++) {
			u[index(i, j)] -= 0.5*(p[index(i + 1, j)] - p[index(i - 1, j)]) / h;
			v[index(i, j)] -= 0.5*(p[index(i, j + 1)] - p[index(i, j - 1)]) / h;
		}
	}
	set_bnd(1, u);
	set_bnd(2, v);

}

void Fluid2DCPU::dens_step(float *x, float *x0, float *u, float *v, float diff, float dt)
{
	addSource(x, x0, dt);
	SWAP(x0, x);
	diffuse(0, x, x0, diff, dt);
	SWAP(x0, x);
	advect(0, x, x0, u, v, dt);

}

void Fluid2DCPU::vel_step(float* u, float *v, float *u0, float *v0, float viscosity, float kappa, float sigma, float dt)
{
	addSource(u, u0, dt);
	addSource(v, v0, dt);

	if (m_parameters.vorticity)
	{
		vort_conf(u, v, 0.01f, dt);
	}

	if (m_parameters.buoyancy)
	{
		buoyancy(v0, dens, kappa, sigma);
		addSource(v, v0, dt);
	}

	SWAP(u0, u); 
	diffuse(1, u, u0, viscosity, dt);
	SWAP(v0, v); 
	diffuse(2, v, v0, viscosity, dt);

	project(u, v, u0, v0);

	SWAP(u0, u);
	SWAP(v0, v);
	advect(1, u, u0, u0, v0, dt);
	advect(2, v, v0, u0, v0, dt);


	project(u, v, u0, v0);
}


void Fluid2DCPU::step()
{
	vel_step(u, v, u_prev, v_prev, m_parameters.viscosity, m_parameters.kappa, m_parameters.sigma, m_parameters.dt);
	dens_step(dens, dens_prev, u, v, m_parameters.diffusion, m_parameters.dt);

	// Reset for next step
	for (unsigned int i = 0; i < dim2; i++)
	{
		dens_prev[i] = 0.f;
		u_prev[i] = 0.f;
		v_prev[i] = 0.f;
	}
}