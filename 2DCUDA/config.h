#ifndef _CONFIG_H_
#define _CONFIG_H_

#define WIDTH 512 // Window width
#define HEIGHT 512 // Window height
#define DIM 256 // Grid Dimension
#define DS (DIM * DIM)

#define THREADS 128
#define BLOCKS (DS/THREADS)


#define TILE_SIZE_X (float) (WIDTH/DIM) / WIDTH
#define TILE_SIZE_Y (float) (HEIGHT/DIM) / HEIGHT

#define CLAMP(v, a, b) (a + (v - a) / (b - a))
#define index(i,j) ((i) + (DIM) *(j))

struct Parameters {
	int		iterations;// Poisson Iteration *Gauss-Seidel method
	float	dt;
	float	kappa;
	float	sigma;
	float	vort_str;
	float	diffusion;
	float	viscosity;
	bool	vorticity;
	bool	buoyancy;
	bool	grid;
	bool	isMaccormack;
};

#endif // !_CONFIG_H_
