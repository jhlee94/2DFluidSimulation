struct Vector2F
{
	float x, y;
};

#define WIDTH 512
#define HEIGHT 512
#define DIM 256
#define DS (DIM * DIM)
#define TILE_SIZE_X (float) (WIDTH/DIM) / WIDTH
#define TILE_SIZE_Y (float) (HEIGHT/DIM) / HEIGHT