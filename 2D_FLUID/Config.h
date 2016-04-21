struct Vector2F
{
	float x, y;
};

#define WIDTH 512
#define HEIGHT 512
#define DIM 512
#define DS (DIM * DIM)
#define TILE_DIM 256
#define TILE_SIZE_X (float) (WIDTH/TILE_DIM) / WIDTH
#define TILE_SIZE_Y (float) (HEIGHT/TILE_DIM) / HEIGHT