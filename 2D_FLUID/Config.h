struct Vector2F
{
	float x, y;
};

#define WIDTH 1024
#define HEIGHT 768

#define GRID_WIDTH 510
#define GRID_HEIGHT 510
#define DIM 126
#define DS ((DIM+2) * (DIM+2))
#define TILE_SIZE_X (float) (GRID_WIDTH/DIM) //
#define TILE_SIZE_Y (float) (GRID_HEIGHT/DIM) //